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


def _auth_headers() -> dict[str, str]:
    """Placeholder de autenticação Cognito (fase de infra): o JWT emitido no
    login entra via `DASHBOARD_API_TOKEN` e viaja como bearer no `GET /api/kpis`
    (Contrato 2). Sem o env, a chamada segue sem header e a API responde 401."""
    token = os.environ.get("DASHBOARD_API_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def cognito_login_url() -> str | None:
    return os.environ.get("COGNITO_LOGIN_URL", "").strip() or None


def fetch_kpis(api_url: str, timeout: int = 10, http: Any = None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not api_url:
        return None, {"status": None, "message": "DASHBOARD_API_URL não configurada"}
    try:
        if http is None:
            import requests

            http = requests
        response = http.get(
            f"{api_url.rstrip('/')}/api/kpis", timeout=timeout, headers=_auth_headers()
        )
    except Exception as exc:
        return None, {"status": None, "message": f"API indisponível: {exc}"}
    if response.status_code == 401:
        return None, {"status": 401, "message": "Não autenticado — faça login via Cognito"}
    if response.status_code != 200:
        return None, {"status": response.status_code, "message": "Erro ao carregar KPIs (toast)"}
    try:
        return response.json(), None
    except ValueError:
        return None, {"status": 200, "message": "Resposta inválida da API"}


def _render_error(error: dict[str, Any], st: Any) -> None:
    st.error(error.get("message", "Erro inesperado"))
    if error.get("status") == 401:
        login_url = cognito_login_url()
        if login_url:
            st.link_button("Entrar com Cognito", login_url)
        else:
            st.info("Login Cognito será habilitado na fase de infra (COGNITO_LOGIN_URL).")


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


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Dashboard SDR Imobiliário", layout="wide")
    api_url = os.environ.get("DASHBOARD_API_URL", "").strip()
    if not api_url:
        st.error("Configure DASHBOARD_API_URL (endpoint do DashAPI).")
        st.stop()
    kpis, error = fetch_kpis(api_url)
    render(kpis, error, api_url)


if __name__ == "__main__":
    main()
