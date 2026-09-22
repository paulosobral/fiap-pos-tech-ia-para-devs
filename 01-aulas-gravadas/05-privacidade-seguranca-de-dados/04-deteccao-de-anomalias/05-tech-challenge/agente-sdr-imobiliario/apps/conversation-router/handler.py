from __future__ import annotations

import json
import logging
import os
from typing import Any

from infra.session_store import SessionStore
from service.entities import KANBAN_STAGE_ORDER, KANBAN_STATUS_MAP, utc_now_iso
from service.flow.lead_qualifier import LeadQualifier
from service.flow.sales_flow import SalesFlow
from service.restriction import DynamoRestrictionCheck
from service.security_layer import FALLBACK_MESSAGE, KmsPiiRegistry, SecurityLayer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

try:
    from service.llm import classify_intent as _llm_classify_intent
    from service.llm import generate_reply as _llm_generate_reply
    from service.properties_catalog import search_properties as _search_properties

    _HAS_LLM = True
except ImportError:  # pragma: no cover - módulos ausentes só em env sem os arquivos
    _HAS_LLM = False

_INTERNAL_INBOUND_PATH = "/internal/inbound-text"
_INTERNAL_CRM_STATUS_PATH = "/internal/crm-status"


def _env(name: str) -> str:
    """Env obrigatória (padrão dos handlers): ausente em produção = falha alta e cedo."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"env var obrigatória ausente: {name}")
    return value


def _llm_api_key() -> str | None:
    """Chave OpenRouter: env override (dev/testes) → Secrets Manager (`LLM_API_SECRET_ID`).

    None = sem IA; o fluxo cai no classificador por regex.
    """
    if key := os.environ.get("LLM_API_KEY"):
        return key
    secret_id = os.environ.get("LLM_API_SECRET_ID")
    if not secret_id:
        return None
    try:
        import boto3

        value = boto3.client("secretsmanager").get_secret_value(SecretId=secret_id)
        return value.get("SecretString") or None
    except Exception:
        logger.warning("Falha ao ler a chave LLM no Secrets Manager (%s); IA desativada", secret_id, exc_info=True)
        return None


def _crm_lead_name(contact: dict[str, list[str]], lead: Any) -> str:
    """Nome do handoff (Contrato 4): NOME do registro de PII; sem NOME, representação
    segura do decisor — nunca a flag "yes"/"no" de `decision_maker`."""
    names = contact.get("NOME") or []
    if names:
        return names[0]
    if lead.decision_maker == "yes":
        return "Decisor (nome não informado)"
    return "Lead (nome não informado)"


class TelegramApi:
    """Cliente mínimo da Bot API do Telegram para envio de respostas."""

    def __init__(self, token: str, timeout: int = 10) -> None:
        self._url = f"https://api.telegram.org/bot{token}/sendMessage"
        self._timeout = timeout

    def send_message(self, chat_id: Any, text: str) -> None:
        import urllib.parse
        import urllib.request

        data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
        urllib.request.urlopen(urllib.request.Request(self._url, data=data), timeout=self._timeout)


class ConversationRouter:
    def __init__(
        self,
        store: SessionStore,
        security_layer: SecurityLayer,
        sales_flow: SalesFlow,
        sqs_client: Any | None = None,
        telegram_client: Any | None = None,
        voice_queue_url: str | None = None,
        crm_queue_url: str | None = None,
        secret_token: str | None = None,
        pii_store: Any | None = None,
        internal_secret_token: str | None = None,
    ) -> None:
        self.store = store
        self.security = security_layer
        self.flow = sales_flow
        self.sqs = sqs_client
        self.telegram = telegram_client
        self.voice_queue_url = voice_queue_url
        self.crm_queue_url = crm_queue_url
        self.secret_token = secret_token
        self.pii_store = pii_store
        self.internal_secret_token = internal_secret_token

    def validate_secret(self, headers: dict[str, str] | None, expected: str | None = None) -> bool:
        expected = expected or self.secret_token or os.environ.get("TELEGRAM_SECRET_TOKEN")
        if not expected:
            return False
        if not headers:
            return False
        normalized = {k.lower(): v for k, v in headers.items()}
        return normalized.get("x-telegram-bot-api-secret-token") == expected

    @property
    def internal_secret(self) -> str | None:
        return self.internal_secret_token or os.environ.get("INTERNAL_SECRET_TOKEN")

    @staticmethod
    def _path_of(event: dict[str, Any]) -> str:
        raw = (
            event.get("path")
            or event.get("rawPath")
            or (event.get("requestContext") or {}).get("http", {}).get("path", "")
            or ""
        )
        return raw.split("?", 1)[0].rstrip("/") or "/"

    def _internal_auth_error(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """401 (secret inválido), 405 (método) e 503 (sem INTERNAL_SECRET_TOKEN configurada)."""
        if not self.internal_secret:
            logger.warning("Internal endpoint unavailable: INTERNAL_SECRET_TOKEN not configured")
            return {"statusCode": 503, "body": json.dumps({"error": "internal endpoint not configured"})}
        method = event.get("httpMethod") or (event.get("requestContext") or {}).get("http", {}).get("method") or "POST"
        if method != "POST":
            return {"statusCode": 405, "body": json.dumps({"error": "method not allowed"})}
        normalized = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
        if normalized.get("x-internal-secret") != self.internal_secret:
            logger.warning("Internal endpoint rejected: invalid secret")
            return {"statusCode": 401, "body": json.dumps({"error": "unauthorized"})}
        return None

    def _parse_json_body(self, event: dict[str, Any]) -> tuple[bool, Any]:
        body = event.get("body")
        try:
            parsed = json.loads(body) if isinstance(body, str) else body
        except json.JSONDecodeError:
            return False, None
        return isinstance(parsed, dict), parsed

    def handle(self, event: dict[str, Any]) -> dict[str, Any]:
        path = self._path_of(event)
        if path == "/health":
            return {"statusCode": 200, "body": json.dumps({"status": "ok"})}
        if path == _INTERNAL_INBOUND_PATH:
            return self.handle_internal_inbound_text(event)
        if path == _INTERNAL_CRM_STATUS_PATH:
            return self.handle_internal_crm_status(event)

        headers = event.get("headers") or {}
        if not self.validate_secret(headers):
            logger.warning("Webhook rejected: invalid secret")
            return {"statusCode": 401, "body": json.dumps({"error": "unauthorized"})}

        ok, update = self._parse_json_body(event)
        if not ok or not update.get("message"):
            return {"statusCode": 400, "body": json.dumps({"error": "invalid payload"})}
        message = update["message"]

        sender = message.get("from") or {}
        chat = message.get("chat") or {}
        telegram_user_id = sender.get("id")
        text = message.get("text", "")
        if telegram_user_id is None:
            return {"statusCode": 400, "body": json.dumps({"error": "invalid payload"})}

        lead, conversation, _ = self.store.get_or_create(telegram_user_id)
        response, state = self._process_lead_message(lead, conversation, text)

        conversation.current_state = state
        conversation.pii_masked = True
        conversation.messages.append({"role": "agent", "text": response, "at": utc_now_iso()})
        self.store.save(lead, conversation)

        if message.get("voice") and self.sqs and self.voice_queue_url:
            self.sqs.send_message(
                QueueUrl=self.voice_queue_url,
                MessageBody=json.dumps(
                    {
                        "message_id": message.get("message_id"),
                        "telegram_user_id": telegram_user_id,
                        "voice_file_id": message["voice"].get("file_id"),
                        "session_id": conversation.session_id,
                        "timestamp": utc_now_iso(),
                    }
                ),
            )

        if self.telegram:
            self.telegram.send_message(chat.get("id"), response)

        return {"statusCode": 200, "body": json.dumps({"ok": True, "state": conversation.current_state})}

    def handle_internal_inbound_text(self, event: dict[str, Any]) -> dict[str, Any]:
        """Re-injeção de texto (u2/u4): POST {telegram_user_id, session_id, text}.

        Recupera a sessão por `telegram_user_id` (GSI telegram-user-index) e usa a
        `session_id` enviada quando houver linha CONV# correspondente; alimenta o
        fluxo como mensagem do lead (mesmo pipeline do webhook, sem envio ao Telegram).
        """
        auth_error = self._internal_auth_error(event)
        if auth_error:
            return auth_error
        ok, body = self._parse_json_body(event)
        if not ok:
            return {"statusCode": 400, "body": json.dumps({"error": "invalid payload"})}
        telegram_user_id = body.get("telegram_user_id")
        session_id = body.get("session_id")
        text = body.get("text")
        valid = (
            isinstance(telegram_user_id, int)
            and not isinstance(telegram_user_id, bool)
            and isinstance(session_id, str)
            and bool(session_id.strip())
            and isinstance(text, str)
            and bool(text.strip())
        )
        if not valid:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {"error": "invalid body: telegram_user_id (int), session_id (str) e text (str) obrigatórios"}
                ),
            }

        lead, conversation = self.store.get_by_telegram_user(telegram_user_id)
        if lead is not None and conversation is not None and conversation.session_id != session_id:
            stored = self.store.get_conversation(lead.lead_id, session_id)
            if stored is not None:
                conversation = stored
            else:
                logger.info(
                    "internal inbound-text: session_id=%s not found; using stored session", session_id
                )
        if lead is None or conversation is None:
            lead, conversation, _ = self.store.get_or_create(telegram_user_id)

        response, state = self._process_lead_message(lead, conversation, text)
        conversation.current_state = state
        conversation.pii_masked = True
        conversation.messages.append({"role": "agent", "text": response, "at": utc_now_iso()})
        self.store.save(lead, conversation)
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "ok": True,
                    "lead_id": lead.lead_id,
                    "session_id": conversation.session_id,
                    "state": conversation.current_state,
                    "response": response,
                }
            ),
        }

    def handle_internal_crm_status(self, event: dict[str, Any]) -> dict[str, Any]:
        """Callback da esteira Kanban (u3): POST {lead_id, session_id, stage}.

        Atualiza o estágio do lead de forma MONOTÔNICA (ordem KANBAN_STAGE_ORDER,
        a mesma da U3): estágio de regressão é ignorado e o já avançado é mantido.
        """
        auth_error = self._internal_auth_error(event)
        if auth_error:
            return auth_error
        ok, body = self._parse_json_body(event)
        if not ok:
            return {"statusCode": 400, "body": json.dumps({"error": "invalid payload"})}
        lead_id = body.get("lead_id")
        session_id = body.get("session_id")
        stage = body.get("stage")
        if (
            not isinstance(lead_id, str)
            or not lead_id.strip()
            or not isinstance(session_id, str)
            or not session_id.strip()
            or not isinstance(stage, str)
            or not stage.strip()
        ):
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {"error": "invalid body: lead_id (str), session_id (str) e stage (str) obrigatórios"}
                ),
            }
        if stage not in KANBAN_STAGE_ORDER:
            return {"statusCode": 400, "body": json.dumps({"error": f"invalid stage: {stage}"})}
        lead = self.store.get_lead(lead_id)
        if lead is None:
            return {"statusCode": 404, "body": json.dumps({"error": "lead not found"})}

        current_stage = KANBAN_STATUS_MAP.get(lead.status, lead.status)
        current_idx = KANBAN_STAGE_ORDER.index(current_stage) if current_stage in KANBAN_STAGE_ORDER else -1
        new_idx = KANBAN_STAGE_ORDER.index(stage)
        if new_idx >= current_idx:
            lead.status = stage
            lead.updated_at = utc_now_iso()
            self.store.save_lead(lead)
            applied = stage
        else:
            logger.info(
                "crm-status regression ignored: lead=%s current=%s requested=%s",
                lead_id,
                current_stage,
                stage,
            )
            applied = current_stage
        return {"statusCode": 200, "body": json.dumps({"ok": True, "lead_id": lead_id, "stage": applied})}

    def _process_lead_message(self, lead: Any, conversation: Any, text: str) -> tuple[str, str]:
        """Guard → máscara → fluxo → handoff do CRM. Retorna (resposta, estado)."""
        violation, reason = self.security.guard(text)
        if violation:
            logger.warning("Guardrail violation: %s", reason)
            return FALLBACK_MESSAGE, conversation.current_state
        masked = self.security.mask(text, session_id=conversation.session_id)
        conversation.messages.append({"role": "lead", "text": masked, "at": utc_now_iso()})
        flow_state = {
            "session_id": conversation.session_id,
            "lead_id": lead.lead_id,
            "message": masked,
            "current_state": conversation.current_state,
            "consent_recorded": conversation.consent_recorded,
            "context": conversation.context,
            "lead_info": conversation.context.get("lead_info", {}),
        }
        flow_state = self.flow.invoke(flow_state)
        response = flow_state.get("response", "")
        state = flow_state.get("current_state", conversation.current_state)
        # Consentimento só é gravado a partir da decisão do lead no fluxo (LGPD R6).
        conversation.consent_recorded = flow_state.get("consent_recorded", conversation.consent_recorded)
        lead.intent = flow_state.get("intent", lead.intent)
        score = flow_state.get("score")
        if score is not None:
            lead.score = score
        if flow_state.get("route"):
            lead.route = flow_state["route"]
        if flow_state.get("lead_qualified"):
            lead.status = "qualified"
            self._enqueue_crm(lead, conversation, flow_state)
        leak, _ = self.security.check_output_leak(response)
        if leak:
            logger.error("PII leakage in response; using fallback")
            response = FALLBACK_MESSAGE
        conversation.context = flow_state.get("context", conversation.context)
        return response, state

    def _enqueue_crm(self, lead: Any, conversation: Any, flow_state: dict[str, Any] | None = None) -> None:
        if not self.sqs or not self.crm_queue_url:
            return
        contact: dict[str, list[str]] = {}
        if self.pii_store is not None:
            contact = self.pii_store.load(conversation.session_id)
        context = (flow_state or {}).get("context") or conversation.context or {}
        info = context.get("lead_info") or {}
        # Urgência derivada dos MESMOS fatores do qualifier (R3/ticket) e persistida no Lead.
        urgency = lead.urgency or self.flow.qualifier.urgency(info)
        lead.urgency = urgency
        lead.budget = info.get("budget") or lead.budget
        lead.deadline = info.get("deadline") or lead.deadline
        lead.area = info.get("area") or lead.area
        self.sqs.send_message(
            QueueUrl=self.crm_queue_url,
            MessageBody=json.dumps(
                {
                    "message_id": conversation.session_id,
                    "lead_id": lead.lead_id,
                    "lead_data": {
                        "name": _crm_lead_name(contact, lead),
                        "email": (contact.get("EMAIL") or [None])[0],
                        "phone": (contact.get("TELEFONE") or [None])[0],
                        "score": lead.score,
                        "urgency": urgency,
                        "intent": lead.intent,
                        "budget": info.get("budget"),
                        "deadline": info.get("deadline"),
                        "area": info.get("area"),
                    },
                    "session_id": conversation.session_id,
                    "timestamp": utc_now_iso(),
                }
            ),
        )


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    import boto3

    dynamodb = boto3.client("dynamodb")
    sqs = boto3.client("sqs")
    store = SessionStore(dynamodb, os.environ.get("SESSIONS_TABLE", "sdr-sessions"))

    # Token/valores injetados no deploy via Secrets Manager (placeholders de env); sem token, sem envio.
    telegram = None
    if token := os.environ.get("TELEGRAM_BOT_TOKEN"):
        telegram = TelegramApi(token)

    pii_store = None
    if key_id := os.environ.get("PII_KMS_KEY_ID"):
        pii_store = KmsPiiRegistry(
            dynamodb, os.environ.get("PII_TABLE", "sdr-pii"), boto3.client("kms"), key_id
        )

    security = SecurityLayer(pii_store=pii_store)

    # LLM (OpenRouter) + RAG (catálogo sintético) — o coração do PRD (FR-02/FR-11).
    # Sem chave (env ou secret sdr/llm-api-key), cai no classificador por regex (comportamento POC igual ao anterior).
    llm_key = _llm_api_key() if _HAS_LLM else None
    llm_classifier = None
    if llm_key:
        def llm_classify(message: str) -> tuple[str, float]:
            try:
                return _llm_classify_intent(message, api_key=llm_key, model=os.environ.get("LLM_MODEL"))
            except Exception:
                logger.warning("LLM classify falhou; usando regex como fallback", exc_info=True)
                return SalesFlow._default_classify(message)

        llm_classifier = llm_classify

    llm_reply = None
    if llm_key:
        def llm_reply(
            message: str, canned: str, lead_info: dict[str, Any], properties: list[dict[str, Any]]
        ) -> str:
            try:
                return _llm_generate_reply(
                    message, canned, lead_info, properties,
                    api_key=llm_key, model=os.environ.get("LLM_MODEL"),
                )
            except Exception:
                logger.warning("LLM reply falhou; usando resposta oficial como fallback", exc_info=True)
                return canned

    flow = SalesFlow(
        lead_qualifier=LeadQualifier(),
        llm_classify_intent=llm_classifier,
        reply_generator=llm_reply,
        properties_rag=_search_properties if _HAS_LLM else None,
        specialist_rotation=[s.strip() for s in os.environ.get("SPECIALIST_ROTATION", "").split(",") if s.strip()],
        specialist_fallback=os.environ.get("SPECIALIST_FALLBACK", "diretor"),
        restriction_check=DynamoRestrictionCheck(dynamodb, os.environ.get("ALERTS_TABLE", "sdr-alerts")),
    )
    router = ConversationRouter(
        store=store,
        security_layer=security,
        sales_flow=flow,
        sqs_client=sqs,
        telegram_client=telegram,
        pii_store=pii_store,
        voice_queue_url=os.environ.get("VOICE_QUEUE_URL"),
        crm_queue_url=os.environ.get("CRM_QUEUE_URL"),
        internal_secret_token=_env("INTERNAL_SECRET_TOKEN"),
    )
    return router.handle(event)
