from __future__ import annotations

import base64
import json
import os
import secrets
import threading
import time
from typing import Any, Callable

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

# Rótulos em português. A API e o banco guardam os códigos em inglês (estáveis); só a tela traduz.
# Código desconhecido aparece como veio, para nunca esconder um valor novo.
STATE_LABELS = {
    "greeting": "Saudação",
    "elicitation": "Consentimento",
    "intent": "Intenção",
    "discovery": "Descoberta",
    "qualification": "Qualificação",
    "recommendation": "Recomendação",
    "conversation": "Em conversa",
    "scheduling": "Agendamento",
    "handoff": "Encaminhado ao corretor",
    "followup": "Acompanhamento",
    "outros": "Outros",
}
INTENT_LABELS = {"purchase": "Compra", "rent": "Locação", "investment": "Investimento"}
URGENCY_LABELS = {"high": "Alta", "medium": "Média", "low": "Baixa"}
ALERT_TYPE_LABELS = {
    "high_volume": "Volume alto de mensagens",
    "long_messages": "Mensagens muito longas",
    "negative_sentiment": "Sentimento negativo",
    "atypical_hours": "Fora do horário comercial",
    "composite": "Vários sinais combinados",
    "ml_ensemble": "Modelo estatístico (Isolation Forest + PCA)",
}
ALERT_STATUS_LABELS = {"open": "Aberto", "resolved": "Resolvido"}
ALERT_ACTION_LABELS = {
    "alert_issued": "Alerta emitido",
    "schedule_restricted": "Agendamento bloqueado",
}


def pt(mapping: dict[str, str], value: Any) -> Any:
    """Traduz um código para o rótulo em português; vazio fica vazio e desconhecido fica como veio."""
    if value in (None, ""):
        return value
    return mapping.get(str(value), value)


def translate_counts(mapping: dict[str, str], counts: dict[str, Any] | None) -> dict[str, Any]:
    """Traduz as chaves de um gráfico somando as que caem no mesmo rótulo (ex.: 'purchase' e 'Compra')."""
    out: dict[str, Any] = {}
    for key, value in (counts or {}).items():
        label = str(pt(mapping, key))
        out[label] = out.get(label, 0) + value
    return out


def alert_rows(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for alert in alerts:
        confidence = alert.get("confidence")
        rows.append(
            {
                "Tipo": pt(ALERT_TYPE_LABELS, alert.get("type")),
                "Confiança": f"{float(confidence) * 100:.0f}%" if isinstance(confidence, (int, float)) else confidence,
                "Situação": pt(ALERT_STATUS_LABELS, alert.get("status")),
                "Ação tomada": pt(ALERT_ACTION_LABELS, alert.get("action_taken")),
                "Detectado em": alert.get("detected_at"),
                "Lead": alert.get("lead_id"),
                "ID do alerta": alert.get("anomaly_id"),
            }
        )
    return rows


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


LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")
PAGE_TITLE = "Dashboard SDR Imobiliário"
BRAND_GOLD = "#D8A858"  # mesma cor de primaryColor em .streamlit/config.toml
SESSION_COOKIE = "sdr_session"
SESSION_TTL_SECONDS = 12 * 3600


class SessionVault:
    """Sessão que sobrevive ao F5. O refresh token do Cognito fica AQUI, no servidor; o
    navegador guarda só um id aleatório (cookie), então roubar o cookie não entrega o token
    do Cognito. Memória do processo: reiniciar o container (scale-in da noite) pede novo login."""

    def __init__(self, ttl_seconds: float = SESSION_TTL_SECONDS, now_fn: Callable[[], float] = time.time) -> None:
        self._ttl = ttl_seconds
        self._now = now_fn
        self._data: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def create(self, refresh_token: str) -> str:
        sid = secrets.token_urlsafe(32)
        with self._lock:
            now = self._now()
            self._data = {k: v for k, v in self._data.items() if v[1] > now}
            self._data[sid] = (refresh_token, now + self._ttl)
        return sid

    def get(self, sid: str | None) -> str | None:
        if not sid:
            return None
        with self._lock:
            entry = self._data.get(sid)
            if entry is None:
                return None
            if entry[1] <= self._now():
                self._data.pop(sid, None)
                return None
            return entry[0]

    def drop(self, sid: str | None) -> None:
        if sid:
            with self._lock:
                self._data.pop(sid, None)


def cognito_refresh(
    refresh_token: str, config: dict[str, str], client: Any = None
) -> tuple[str | None, str | None]:
    """Novo IdToken a partir do refresh token (o id token dura 1 h)."""
    if client is None:
        import boto3

        client = boto3.client("cognito-idp", region_name=config["region"])
    try:
        resp = client.initiate_auth(
            ClientId=config["client_id"],
            AuthFlow="REFRESH_TOKEN_AUTH",
            AuthParameters={"REFRESH_TOKEN": refresh_token},
        )
    except Exception as exc:
        return None, f"Sessão expirada: {exc}"
    return resp["AuthenticationResult"]["IdToken"], None


def jwt_claims(token: str | None) -> dict[str, Any]:
    """Só para EXIBIR quem está logado — a validação de verdade é do authorizer do API Gateway."""
    try:
        payload = (token or "").split(".")[1]
        return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except Exception:
        return {}


def user_label(token: str | None) -> str:
    claims = jwt_claims(token)
    return str(claims.get("email") or claims.get("cognito:username") or "usuário")


def cookie_script(value: str, max_age: int) -> str:
    """JS (em iframe sem altura) que grava/apaga o cookie de sessão no documento pai."""
    assert all(c.isalnum() or c in "-_" for c in value), "valor de cookie inesperado"
    return (
        "<script>window.parent.document.cookie = "
        f"'{SESSION_COOKIE}={value}; path=/; max-age={max_age}; SameSite=Strict';</script>"
    )


def page_icon() -> Any:
    """Favicon da aba: o logo (Streamlit usa o ícone dele se não for informado). Miniatura em
    memória — o arquivo do logo não é alterado; sem o arquivo, cai num emoji."""
    try:
        from PIL import Image

        icon = Image.open(LOGO_PATH)
        icon.thumbnail((128, 128))
        return icon
    except Exception:
        return "🏢"


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


def fetch_leads(
    api_url: str, token: str | None, timeout: int = 20, http: Any = None
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    if not api_url:
        return None, {"status": None, "message": "DASHBOARD_API_URL não configurada"}
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        if http is None:
            import requests

            http = requests
        response = http.get(f"{api_url.rstrip('/')}/api/leads", timeout=timeout, headers=headers)
    except Exception as exc:
        return None, {"status": None, "message": f"API indisponível: {exc}"}
    if response.status_code == 401:
        return None, {"status": 401, "message": "Sessão expirada — faça login novamente"}
    if response.status_code != 200:
        return None, {"status": response.status_code, "message": "Erro ao carregar leads"}
    try:
        return response.json().get("leads", []), None
    except ValueError:
        return None, {"status": 200, "message": "Resposta inválida da API"}


def send_lead_to_crm(
    api_url: str, token: str | None, lead_id: str, timeout: int = 20, http: Any = None
) -> tuple[bool, str]:
    """Reenvia o lead ao CRM (fila do handoff → crm-adapter → HubSpot)."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        if http is None:
            import requests

            http = requests
        response = http.post(
            f"{api_url.rstrip('/')}/api/leads/{lead_id}/crm", timeout=timeout, headers=headers
        )
    except Exception as exc:
        return False, f"API indisponível: {exc}"
    if response.status_code == 202:
        return True, "Lead enviado ao CRM (HubSpot)."
    if response.status_code == 404:
        return False, "Lead não encontrado."
    return False, f"Falha ao enviar ao CRM (HTTP {response.status_code})."


_LEAD_COLUMNS = (
    ("name", "Nome"),
    ("phone", "Telefone"),
    ("email", "E-mail"),
    ("score", "Score"),
    ("urgency", "Urgência"),
    ("intent", "Intenção"),
    ("budget", "Orçamento"),
    ("area", "Área"),
    ("region", "Região"),
    ("deadline", "Prazo"),
    ("property", "Imóvel escolhido"),
    ("property_price", "Valor do imóvel"),
    ("state", "Estado"),
    ("updated_at", "Atualizado"),
)


_LEAD_TRANSLATIONS = {"state": STATE_LABELS, "intent": INTENT_LABELS, "urgency": URGENCY_LABELS}


def render_leads(leads: list[dict[str, Any]] | None, error: dict[str, Any] | None, api_url: str, token: str | None) -> None:
    import streamlit as st

    st.subheader("Leads")
    if error:
        _render_error(error, st)
        return
    if not leads:
        st.info("Nenhum lead ainda.")
        return
    st.dataframe(
        [{label: pt(_LEAD_TRANSLATIONS.get(key, {}), lead.get(key)) for key, label in _LEAD_COLUMNS} for lead in leads],
        use_container_width=True,
        hide_index=True,
    )
    by_label = {f"{lead.get('name')} — {lead.get('phone') or lead.get('email') or 'sem contato'}": lead["lead_id"] for lead in leads}
    choice = st.selectbox("Lead", list(by_label))
    if st.button("Enviar ao HubSpot"):
        ok, message = send_lead_to_crm(api_url, token, by_label[choice])
        (st.success if ok else st.error)(message)


def _bar_chart(container: Any, data: dict[str, Any] | None, x_label: str) -> None:
    """Gráfico de barras na cor da marca. Sem dados, o st.bar_chart com `color` levanta exceção
    (lista de cores x colunas vazia) e derrubava a tela inteira num ambiente recém-criado."""
    if not data:
        container.caption(f"Sem dados de {x_label} ainda.")
        return
    container.bar_chart(data, x_label=x_label, y_label="leads", color=BRAND_GOLD)


def _render_error(error: dict[str, Any], st: Any) -> None:
    st.error(error.get("message", "Erro inesperado"))


def render(kpis: dict[str, Any] | None, error: dict[str, Any] | None, api_url: str) -> None:
    import streamlit as st

    st.set_page_config(page_title=PAGE_TITLE, page_icon=page_icon(), layout="wide")
    if error:
        _render_error(error, st)
        return
    if not kpis:
        st.info("Sem dados de KPI no momento.")
        return

    columns = st.columns(5)
    metrics = (
        ("Leads hoje", kpis.get("leads_today", 0)),
        ("Leads na semana", kpis.get("leads_week", 0)),
        ("1ª resposta p90", f"{float(kpis.get('response_time_p90', 0.0)):.0f}s"),
        ("Taxa de qualificação", f"{float(kpis.get('qualification_rate', 0.0)) * 100:.0f}%"),
        ("Agendamentos", kpis.get("scheduled_visits", 0)),
    )
    for column, (label, value) in zip(columns, metrics):
        column.metric(label, value)

    with st.expander("Esteira por estado da conversa"):
        funnel = kpis.get("funnel", {})
        kanban = st.columns(len(KANBAN_STATES))
        for column, state in zip(kanban, KANBAN_STATES):
            column.metric(pt(STATE_LABELS, state), funnel.get(state, 0))

    chart_left, chart_right = st.columns(2)
    _bar_chart(chart_left, translate_counts(INTENT_LABELS, kpis.get("intents")), "intenção")
    _bar_chart(chart_right, kpis.get("route_distribution"), "roleta")

    st.subheader(f"Anomalias (24h) — {len(kpis.get('alerts', []))} alerta(s)")
    alerts = kpis.get("alerts", [])
    if alerts:
        st.dataframe(alert_rows(alerts), use_container_width=True)
    else:
        st.info("Nenhum alerta nas últimas 24h.")


@__import__("functools").lru_cache(maxsize=1)
def _fallback_vault() -> SessionVault:
    return SessionVault()


def _vault(st: Any) -> SessionVault:
    """Um cofre por processo, compartilhado entre sessões do Streamlit."""
    try:
        return st.cache_resource(_fallback_vault)()
    except Exception:
        return _fallback_vault()


def _start_session(st: Any, tokens: dict[str, Any]) -> None:
    st.session_state["id_token"] = tokens["IdToken"]
    refresh = tokens.get("RefreshToken")
    if refresh:
        sid = _vault(st).create(refresh)
        st.session_state["_sid"] = sid
        st.session_state["_set_cookie"] = sid


def _browser_sid(st: Any) -> str | None:
    try:
        return st.context.cookies.get(SESSION_COOKIE)
    except Exception:
        return None


def _refresh_session(st: Any, config: dict[str, str]) -> bool:
    """Tenta renovar o id token a partir do cookie de sessão (F5 ou expiração de 1 h)."""
    sid = st.session_state.get("_sid") or _browser_sid(st)
    refresh = _vault(st).get(sid)
    if not refresh:
        return False
    token, _error = cognito_refresh(refresh, config)
    if not token:
        _vault(st).drop(sid)
        return False
    st.session_state["id_token"] = token
    st.session_state["_sid"] = sid
    return True


def _emit_cookie(st: Any, value: str, max_age: int) -> None:
    import streamlit.components.v1 as components

    components.html(cookie_script(value, max_age), height=0)


def _render_header(st: Any) -> None:
    logo, left, mid, right = st.columns([1, 5, 3, 1], vertical_alignment="center")
    if os.path.exists(LOGO_PATH):
        logo.image(LOGO_PATH, width=64)
    left.markdown("### W Levitt — Dashboard SDR")
    mid.caption(f"👤 {user_label(st.session_state.get('id_token'))}")
    if right.button("Sair"):
        _vault(st).drop(st.session_state.get("_sid") or _browser_sid(st))
        st.session_state.clear()
        st.session_state["_clear_cookie"] = True
        st.rerun()


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
                _start_session(st, result["tokens"])
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
            _start_session(st, result["tokens"])
            st.rerun()


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title=PAGE_TITLE, page_icon=page_icon(), layout="wide")
    api_url = os.environ.get("DASHBOARD_API_URL", "").strip()
    if not api_url:
        st.error("Configure DASHBOARD_API_URL (endpoint do DashAPI).")
        st.stop()
    config = cognito_config()
    if config is None:
        st.error("Configure COGNITO_USER_POOL_ID, COGNITO_CLIENT_ID e AWS_REGION.")
        st.stop()
    if st.session_state.pop("_clear_cookie", False):
        _emit_cookie(st, "x", 0)
    if not st.session_state.get("id_token"):
        _refresh_session(st, config)
    if not st.session_state.get("id_token"):
        _render_login(st, config)
        return
    pending_cookie = st.session_state.pop("_set_cookie", None)
    if pending_cookie:
        _emit_cookie(st, pending_cookie, SESSION_TTL_SECONDS)
    _render_header(st)
    kpis, error = fetch_kpis(api_url, st.session_state["id_token"])
    if error is not None and error.get("status") == 401 and _refresh_session(st, config):
        kpis, error = fetch_kpis(api_url, st.session_state["id_token"])
    if error is not None and error.get("status") == 401:
        st.session_state.pop("id_token", None)
        st.rerun()
        return
    render(kpis, error, api_url)
    leads, leads_error = fetch_leads(api_url, st.session_state["id_token"])
    render_leads(leads, leads_error, api_url, st.session_state["id_token"])


if __name__ == "__main__":
    main()
