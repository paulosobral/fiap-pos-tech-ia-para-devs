from __future__ import annotations

import os
from typing import Any

KANBAN_STATES = (
    "greeting",
    "elicitation",
    "intent",
    "qualification",
    "recommendation",
    "scheduling",
    "handoff",
    "followup",
)


def cognito_config() -> dict[str, str] | None:
    """Lê a configuração do User Pool/App Client injetada pelo Terraform
    (env vars do task definition, infra/ecs.tf). Sem isso, não há como logar."""
    pool_id = os.environ.get("COGNITO_USER_POOL_ID", "").strip()
    client_id = os.environ.get("COGNITO_CLIENT_ID", "").strip()
    region = os.environ.get("AWS_REGION", "").strip()
    if not (pool_id and client_id and region):
        return None
    return {"pool_id": pool_id, "client_id": client_id, "region": region}


def cognito_login(
    username: str, password: str, config: dict[str, str], client: Any = None
) -> tuple[dict[str, Any] | None, str | None]:
    """`InitiateAuth` (USER_PASSWORD_AUTH) direto no User Pool — sem Hosted UI/
    redirect, porque o dashboard-ui roda em ECS com IP público efêmero (sem
    domínio/ALB para um callback OAuth estável). Retorna (resultado, erro);
    resultado pode ser `{"tokens": AuthenticationResult}` ou, no primeiro
    login de usuário criado pelo admin, `{"challenge": "NEW_PASSWORD_REQUIRED",
    "session": ...}`."""
    if client is None:
        import boto3

        client = boto3.client("cognito-idp", region_name=config["region"])
    try:
        resp = client.initiate_auth(
            ClientId=config["client_id"],
            AuthFlow="USER_PASSWORD_AUTH",
            AuthParameters={"USERNAME": username, "PASSWORD": password},
        )
    except Exception as exc:
        return None, f"Login falhou: {exc}"
    if resp.get("ChallengeName") == "NEW_PASSWORD_REQUIRED":
        return {"challenge": "NEW_PASSWORD_REQUIRED", "session": resp["Session"]}, None
    return {"tokens": resp["AuthenticationResult"]}, None


def cognito_respond_new_password(
    username: str, new_password: str, session: str, config: dict[str, str], client: Any = None
) -> tuple[dict[str, Any] | None, str | None]:
    """Completa o desafio de 1º login (usuário criado via `admin-create-user`
    sempre nasce em FORCE_CHANGE_PASSWORD)."""
    if client is None:
        import boto3

        client = boto3.client("cognito-idp", region_name=config["region"])
    try:
        resp = client.respond_to_auth_challenge(
            ClientId=config["client_id"],
            ChallengeName="NEW_PASSWORD_REQUIRED",
            Session=session,
            ChallengeResponses={"USERNAME": username, "NEW_PASSWORD": new_password},
        )
    except Exception as exc:
        return None, f"Troca de senha falhou: {exc}"
    return {"tokens": resp["AuthenticationResult"]}, None


def fetch_kpis(
    api_url: str, token: str | None, timeout: int = 10, http: Any = None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not api_url:
        return None, {"status": None, "message": "DASHBOARD_API_URL não configurada"}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        if http is None:
            import requests

            http = requests
        response = http.get(f"{api_url.rstrip('/')}/api/kpis", timeout=timeout, headers=headers)
    except Exception as exc:
        return None, {"status": None, "message": f"API indisponível: {exc}"}
    if response.status_code == 401:
        return None, {"status": 401, "message": "Sessão expirada — faça login novamente"}
    if response.status_code != 200:
        return None, {"status": response.status_code, "message": "Erro ao carregar KPIs (toast)"}
    try:
        return response.json(), None
    except ValueError:
        return None, {"status": 200, "message": "Resposta inválida da API"}


def _render_error(error: dict[str, Any], st: Any) -> None:
    st.error(error.get("message", "Erro inesperado"))


def render(kpis: dict[str, Any] | None, error: dict[str, Any] | None, api_url: str) -> None:
    import streamlit as st

    st.set_page_config(page_title="Dashboard SDR Imobiliário", layout="wide")
    st.title("Dashboard SDR Imobiliário")
    if error:
        _render_error(error, st)
        return
    if not kpis:
        st.info("Sem dados de KPI no momento.")
        return

    st.subheader("Métricas de negócio (FR7.2)")
    columns = st.columns(7)
    metrics = (
        ("Leads hoje", kpis.get("leads_today", 0)),
        ("Leads na semana", kpis.get("leads_week", 0)),
        ("1ª resposta p90 (s)", kpis.get("response_time_p90", 0.0)),
        ("Taxa de qualificação", f"{float(kpis.get('qualification_rate', 0.0)) * 100:.1f}%"),
        ("Agendamentos", kpis.get("scheduled_visits", 0)),
        ("Anomalias 24h", kpis.get("anomalies_count", 0)),
        ("Custo mês (US$)", kpis.get("cost_monthly", 0.0)),
    )
    for column, (label, value) in zip(columns, metrics):
        column.metric(label, value)

    st.subheader("Esteira Kanban (FR7.3)")
    funnel = kpis.get("funnel", {})
    kanban = st.columns(len(KANBAN_STATES))
    for column, state in zip(kanban, KANBAN_STATES):
        column.metric(state, funnel.get(state, 0))

    chart_left, chart_right = st.columns(2)
    chart_left.bar_chart(kpis.get("intents", {}), x_label="intenção", y_label="leads")
    chart_right.bar_chart(kpis.get("route_distribution", {}), x_label="roleta", y_label="leads")

    st.subheader("Alertas de anomalias — últimas 24h (FR7.4)")
    alerts = kpis.get("alerts", [])
    if alerts:
        st.dataframe(alerts, use_container_width=True)
    else:
        st.info("Nenhum alerta nas últimas 24h.")


def _render_login(st: Any, config: dict[str, str]) -> None:
    st.title("Dashboard SDR Imobiliário — login")
    challenge = st.session_state.get("_cognito_challenge")
    if challenge:
        st.info("Primeiro acesso: defina uma nova senha.")
        with st.form("new_password_form"):
            new_password = st.text_input("Nova senha", type="password")
            submitted = st.form_submit_button("Definir senha e entrar")
        if submitted:
            result, error = cognito_respond_new_password(
                challenge["username"], new_password, challenge["session"], config
            )
            if error:
                st.error(error)
            else:
                st.session_state["id_token"] = result["tokens"]["IdToken"]
                st.session_state.pop("_cognito_challenge", None)
                st.rerun()
        return
    with st.form("login_form"):
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        submitted = st.form_submit_button("Entrar")
    if submitted:
        result, error = cognito_login(username, password, config)
        if error:
            st.error(error)
        elif result.get("challenge") == "NEW_PASSWORD_REQUIRED":
            st.session_state["_cognito_challenge"] = {
                "username": username,
                "session": result["session"],
            }
            st.rerun()
        else:
            st.session_state["id_token"] = result["tokens"]["IdToken"]
            st.rerun()


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Dashboard SDR Imobiliário", layout="wide")
    api_url = os.environ.get("DASHBOARD_API_URL", "").strip()
    if not api_url:
        st.error("Configure DASHBOARD_API_URL (endpoint do DashAPI).")
        st.stop()
    config = cognito_config()
    if config is None:
        st.error("Configure COGNITO_USER_POOL_ID, COGNITO_CLIENT_ID e AWS_REGION.")
        st.stop()
    if not st.session_state.get("id_token"):
        _render_login(st, config)
        return
    kpis, error = fetch_kpis(api_url, st.session_state["id_token"])
    if error is not None and error.get("status") == 401:
        st.session_state.pop("id_token", None)
        st.rerun()
        return
    render(kpis, error, api_url)
    if st.button("Sair"):
        st.session_state.clear()
        st.rerun()


if __name__ == "__main__":
    main()
