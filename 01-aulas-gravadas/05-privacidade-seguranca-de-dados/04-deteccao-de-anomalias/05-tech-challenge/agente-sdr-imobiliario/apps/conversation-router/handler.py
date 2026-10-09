from __future__ import annotations

import json
import logging
import os
import re
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
    from service.llm import extract_and_route as _llm_extract_and_route
    from service.llm import generate_reply as _llm_generate_reply
    from service.properties_catalog import known_places as _known_places
    from service.properties_catalog import search_properties as _search_properties
    from service.properties_catalog import set_dynamodb_client as _set_properties_dynamo_client

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
        logger.warning(
            "Falha ao ler a chave LLM no Secrets Manager (%s); IA desativada",
            secret_id,
            exc_info=True,
        )
        return None


def _crm_text(value: Any) -> str | None:
    """O contrato do CRM (Contrato 4) só aceita texto ou null; a LLM devolve números (1000)."""
    if value in (None, ""):
        return None
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def _format_phone(raw: str) -> str:
    """Telefone BR legível: (11) 97991-8262. Formato desconhecido volta como veio."""
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) > 11 and digits.startswith("55"):
        digits = digits[2:]
    if len(digits) == 11:
        return f"({digits[:2]}) {digits[2:7]}-{digits[7:]}"
    if len(digits) == 10:
        return f"({digits[:2]}) {digits[2:6]}-{digits[6:]}"
    return raw


def _contact_confirmation(before: dict[str, list[str]], after: dict[str, list[str]]) -> str:
    """Ecoa pro lead o contato que acabou de chegar nesta mensagem, para ele corrigir erro de
    digitação. Feito em código (o número nunca passa pela LLM) depois do check de vazamento."""
    parts: list[str] = []
    new_phones = [v for v in after.get("TELEFONE", []) if v not in before.get("TELEFONE", [])]
    new_emails = [v for v in after.get("EMAIL", []) if v not in before.get("EMAIL", [])]
    if new_phones:
        parts.append(f"telefone {_format_phone(new_phones[-1])}")
    if new_emails:
        parts.append(f"e-mail {new_emails[-1]}")
    if not parts:
        return ""
    return f"Anotei seu {' e '.join(parts)}. Se algo estiver errado, é só me mandar o correto."


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
        self._base_url = f"https://api.telegram.org/bot{token}"
        self._timeout = timeout

    def send_message(self, chat_id: Any, text: str) -> None:
        import urllib.parse
        import urllib.request

        data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
        urllib.request.urlopen(
            urllib.request.Request(f"{self._base_url}/sendMessage", data=data),
            timeout=self._timeout,
        )

    def send_photo(self, chat_id: Any, photo_url: str, caption: str | None = None) -> None:
        """Envia foto por URL remota (Contrato 1 estendido): o Telegram baixa a
        imagem direto do CDN do imóvel — sem precisarmos armazenar/servir o arquivo."""
        import urllib.parse
        import urllib.request

        params = {"chat_id": chat_id, "photo": photo_url}
        if caption:
            params["caption"] = caption[:1024]
        data = urllib.parse.urlencode(params).encode()
        urllib.request.urlopen(
            urllib.request.Request(f"{self._base_url}/sendPhoto", data=data),
            timeout=self._timeout,
        )


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

    def validate_secret(
        self, headers: dict[str, str] | None, expected: str | None = None
    ) -> bool:
        expected = (
            expected
            or self.secret_token
            or os.environ.get("TELEGRAM_SECRET_TOKEN")
            or os.environ.get("INTERNAL_SECRET_TOKEN")
        )
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
            logger.warning(
                "Internal endpoint unavailable: INTERNAL_SECRET_TOKEN not configured"
            )
            return {
                "statusCode": 503,
                "body": json.dumps({"error": "internal endpoint not configured"}),
            }
        method = (
            event.get("httpMethod")
            or (event.get("requestContext") or {}).get("http", {}).get("method")
            or "POST"
        )
        if method != "POST":
            return {
                "statusCode": 405,
                "body": json.dumps({"error": "method not allowed"}),
            }
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

        lead, conversation, created = self.store.get_or_create(telegram_user_id)
        logger.info(
            "sessao: created=%s telegram_user_id=%s lead_id=%s session_id=%s",
            created, telegram_user_id, lead.lead_id, conversation.session_id,
        )
        if message.get("voice"):
            voice = message["voice"]
            if not voice.get("file_id"):
                return {"statusCode": 400, "body": json.dumps({"error": "invalid voice payload"})}
            if not self.sqs or not self.voice_queue_url:
                logger.error("Voice processing unavailable: SQS queue is not configured")
                return {"statusCode": 503, "body": json.dumps({"error": "voice processing unavailable"})}
            self.sqs.send_message(
                QueueUrl=self.voice_queue_url,
                MessageBody=json.dumps(
                    {
                        "message_id": message.get("message_id"),
                        "telegram_user_id": telegram_user_id,
                        "voice_file_id": voice["file_id"],
                        "session_id": conversation.session_id,
                        "timestamp": utc_now_iso(),
                    }
                ),
            )
            if self.telegram:
                try:
                    self.telegram.send_message(
                        chat.get("id"),
                        "Recebi seu áudio e coloquei na fila para transcrição. "
                        "Assim que for processado, retorno por aqui.",
                    )
                except Exception:
                    logger.warning("Voice queued but Telegram acknowledgement failed", exc_info=True)
            return {
                "statusCode": 200,
                "body": json.dumps({"ok": True, "state": conversation.current_state}),
            }

        response, state, response_images = self._process_lead_message(lead, conversation, text)
        self._persist_telegram_profile_name(sender, conversation)

        conversation.current_state = state
        conversation.pii_masked = True
        conversation.messages.append(
            {"role": "agent", "text": response, "at": utc_now_iso()}
        )
        self.store.save(lead, conversation)

        if self.telegram:
            try:
                self.telegram.send_message(chat.get("id"), response)
            except Exception:
                # Estado já foi salvo (linha acima): se a notificação falhar e
                # devolvermos 500, o Telegram reenvia o MESMO update — e o
                # reprocessamento acharia a sessão já avançada (ex.: já passou
                # da saudação), produzindo respostas duplicadas/fora de
                # sincronia. Melhor logar e devolver 200 — processamos certo,
                # só a entrega falhou.
                logger.error(
                    "Falha ao enviar send_message via Telegram; sessão já salva, "
                    "não propagando erro (evita retry/reprocessamento duplicado)",
                    exc_info=True,
                )
            for image_url in response_images:
                try:
                    self.telegram.send_photo(chat.get("id"), image_url)
                except Exception:
                    logger.warning(
                        "Falha ao enviar foto via Telegram (url=%s)",
                        image_url,
                        exc_info=True,
                    )

        return {
            "statusCode": 200,
            "body": json.dumps({"ok": True, "state": conversation.current_state}),
        }

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
                    {
                        "error": "invalid body: telegram_user_id (int), session_id (str) e text (str) obrigatórios"
                    }
                ),
            }

        lead, conversation = self.store.get_by_telegram_user(telegram_user_id)
        if (
            lead is not None
            and conversation is not None
            and conversation.session_id != session_id
        ):
            stored = self.store.get_conversation(lead.lead_id, session_id)
            if stored is not None:
                conversation = stored
            else:
                logger.info(
                    "internal inbound-text: session_id=%s not found; using stored session",
                    session_id,
                )
        if lead is None or conversation is None:
            lead, conversation, _ = self.store.get_or_create(telegram_user_id)

        response, state, response_images = self._process_lead_message(lead, conversation, text)
        conversation.current_state = state
        conversation.pii_masked = True
        conversation.messages.append(
            {"role": "agent", "text": response, "at": utc_now_iso()}
        )
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
                    "response_images": response_images,
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
                    {
                        "error": "invalid body: lead_id (str), session_id (str) e stage (str) obrigatórios"
                    }
                ),
            }
        if stage not in KANBAN_STAGE_ORDER:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"invalid stage: {stage}"}),
            }
        lead = self.store.get_lead(lead_id)
        if lead is None:
            return {"statusCode": 404, "body": json.dumps({"error": "lead not found"})}

        current_stage = KANBAN_STATUS_MAP.get(lead.status, lead.status)
        current_idx = (
            KANBAN_STAGE_ORDER.index(current_stage)
            if current_stage in KANBAN_STAGE_ORDER
            else -1
        )
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
        return {
            "statusCode": 200,
            "body": json.dumps({"ok": True, "lead_id": lead_id, "stage": applied}),
        }

    def _process_lead_message(
        self, lead: Any, conversation: Any, text: str
    ) -> tuple[str, str, list[str]]:
        """Guard → máscara → fluxo → handoff do CRM. Retorna (resposta, estado, fotos)."""
        violation, reason = self.security.guard(text)
        if violation:
            logger.warning("Guardrail violation: %s", reason)
            return FALLBACK_MESSAGE, conversation.current_state, []
        contact_before = (
            self.pii_store.load(conversation.session_id)
            if self.pii_store is not None
            else {}
        )
        masked = self.security.mask(text, session_id=conversation.session_id)
        contact = (
            self.pii_store.load(conversation.session_id)
            if self.pii_store is not None
            else {}
        )
        flow_contact_fields = {
            "name": "NOME",
            "email": "EMAIL",
            "phone": "TELEFONE",
        }
        missing_contact_fields = [
            field
            for field, label in flow_contact_fields.items()
            if not contact.get(label)
        ]
        conversation.messages.append(
            {"role": "lead", "text": masked, "at": utc_now_iso()}
        )
        flow_state = {
            "session_id": conversation.session_id,
            "lead_id": lead.lead_id,
            "message": masked,
            "conversation_history": self._recent_conversation_history(conversation),
            "missing_contact_fields": missing_contact_fields,
            "current_state": conversation.current_state,
            "consent_recorded": conversation.consent_recorded,
            "context": conversation.context,
            "lead_info": conversation.context.get("lead_info", {}),
        }
        logger.info(
            "turno IN: session=%s state_in=%s context_keys=%s lead_info_in=%s message=%r",
            conversation.session_id,
            conversation.current_state,
            sorted((conversation.context or {}).keys()),
            conversation.context.get("lead_info", {}),
            masked[:80],
        )
        flow_state = self.flow.invoke(flow_state)
        response = flow_state.get("response", "")
        state = flow_state.get("current_state", conversation.current_state)
        logger.info(
            "turno OUT: state_out=%s tool=%s response_properties=%d response_images=%d",
            state,
            flow_state.get("_router_tool"),
            len(flow_state.get("response_properties") or []),
            len(flow_state.get("response_images") or []),
        )
        # Consentimento só é gravado a partir da decisão do lead no fluxo (LGPD R6).
        conversation.consent_recorded = flow_state.get(
            "consent_recorded", conversation.consent_recorded
        )
        lead.intent = flow_state.get("intent", lead.intent)
        score = flow_state.get("score")
        if score is not None:
            lead.score = score
        if flow_state.get("route"):
            lead.route = flow_state["route"]
        if flow_state.get("lead_qualified"):
            lead.status = "qualified"
            self._enqueue_crm(lead, conversation, flow_state)
        # Roteamento tool-agent (ADR-011) nunca passa por _node_qualification
        # (fica em "conversation" o turno inteiro) — sem este gatilho, nenhum
        # lead chegava ao CRM nesse fluxo, nem mesmo após agendamento concluído
        # com contato completo. Dispara 1x, só na transição PARA handoff (não
        # repete em toda mensagem seguinte já dentro de handoff).
        if state == "handoff" and conversation.current_state != "handoff":
            lead.status = lead.status or "qualified"
            self._enqueue_crm(lead, conversation, flow_state)
        leak, label = self.security.check_output_leak(response, session_pii=contact)
        if leak:
            # Reescrita da LLM barrada: o texto oficial do fluxo (determinístico) costuma estar
            # limpo — usa ele em vez de largar "Não posso ajudar com isso" no meio da conversa.
            official = flow_state.get("official_response")
            if official and official != response and not self.security.check_output_leak(
                official, session_pii=contact
            )[0]:
                logger.warning("Resposta humanizada barrada (%s); enviando o texto oficial", label)
                response = official
            else:
                logger.error("PII leakage in response (%s); using fallback", label)
                response = FALLBACK_MESSAGE
        if response != FALLBACK_MESSAGE:
            confirmation = _contact_confirmation(contact_before, contact)
            if confirmation:
                response = f"{response}\n\n{confirmation}"
        conversation.context = flow_state.get("context", conversation.context)
        response_images = flow_state.get("response_images") or []
        return response, state, response_images

    def _persist_telegram_profile_name(
        self, sender: dict[str, Any], conversation: Any
    ) -> None:
        if not conversation.consent_recorded or self.pii_store is None:
            return
        name = " ".join(
            part.strip()
            for part in (sender.get("first_name"), sender.get("last_name"))
            if isinstance(part, str) and part.strip()
        )
        if not name:
            return
        stored = self.pii_store.load(conversation.session_id)
        if not stored.get("NOME"):
            self.pii_store.save(conversation.session_id, {"NOME": [name]})

    @staticmethod
    def _recent_conversation_history(conversation: Any) -> list[dict[str, str]]:
        """Retorna turnos anteriores já mascarados, sem duplicar a mensagem atual."""
        if not getattr(conversation, "pii_masked", False):
            return []
        history: list[dict[str, str]] = []
        for turn in (conversation.messages[:-1])[-8:]:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role")
            text = turn.get("text")
            if role not in ("lead", "agent") or not isinstance(text, str) or not text:
                continue
            history.append(
                {
                    "role": "user" if role == "lead" else "assistant",
                    "content": text[:500],
                }
            )
        return history

    def _enqueue_crm(
        self, lead: Any, conversation: Any, flow_state: dict[str, Any] | None = None
    ) -> None:
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
        budget_text = _crm_text(info.get("budget")) or (
            f"a partir de {_crm_text(info.get('budget_min'))}" if info.get("budget_min") else None
        )
        lead.budget = budget_text or lead.budget
        lead.deadline = _crm_text(info.get("deadline")) or lead.deadline
        lead.area = _crm_text(info.get("area")) or lead.area
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
                        "budget": budget_text,
                        "deadline": _crm_text(info.get("deadline")),
                        "area": _crm_text(info.get("area")),
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
    if _HAS_LLM:
        _set_properties_dynamo_client(dynamodb)

    # Token/valores injetados no deploy via Secrets Manager (placeholders de env); sem token, sem envio.
    telegram = None
    if token := os.environ.get("TELEGRAM_BOT_TOKEN"):
        telegram = TelegramApi(token)

    pii_store = None
    if key_id := os.environ.get("PII_KMS_KEY_ID"):
        pii_store = KmsPiiRegistry(
            dynamodb,
            os.environ.get("PII_TABLE", "sdr-pii"),
            boto3.client("kms"),
            key_id,
        )

    security = SecurityLayer(pii_store=pii_store)

    # LLM (OpenRouter) + RAG (catálogo sintético) — o coração do PRD (FR-02/FR-11).
    # Sem chave (env ou secret sdr/llm-api-key), cai no classificador por regex (comportamento POC igual ao anterior).
    llm_key = _llm_api_key() if _HAS_LLM else None
    llm_classifier = None
    if llm_key:

        def llm_classify(message: str) -> tuple[str, float]:
            try:
                return _llm_classify_intent(message, api_key=llm_key)
            except Exception:
                logger.warning(
                    "LLM classify falhou; usando regex como fallback", exc_info=True
                )
                return SalesFlow._default_classify(message)

        llm_classifier = llm_classify

    llm_reply = None
    llm_router = None
    if llm_key and _HAS_LLM:

        def llm_reply(
            message: str,
            canned: str,
            lead_info: dict[str, Any],
            properties: list[dict[str, Any]],
            **kwargs: Any,
        ) -> str:
            try:
                return _llm_generate_reply(
                    message,
                    canned,
                    lead_info,
                    properties,
                    api_key=llm_key,
                    favorite_property=kwargs.get("favorite_property"),
                    conversation_stage=kwargs.get("conversation_stage"),
                    shown_properties_count=kwargs.get("shown_properties_count"),
                    visit_interest=bool(kwargs.get("visit_interest")),
                    rejected_properties=kwargs.get("rejected_properties"),
                    last_tool=kwargs.get("last_tool"),
                    conversation_history=kwargs.get("conversation_history"),
                    photos_sending=int(kwargs.get("photos_sending") or 0),
                    contact_channels=kwargs.get("contact_channels"),
                    contact_request=bool(kwargs.get("contact_request")),
                )
            except Exception:
                logger.warning(
                    "LLM reply falhou; usando resposta oficial como fallback",
                    exc_info=True,
                )
                return canned

        def llm_route(
            message: str,
            lead_info: dict[str, Any],
            current_state: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            return _llm_extract_and_route(
                message,
                lead_info,
                current_state,
                api_key=llm_key,
                shown_properties=kwargs.get("shown_properties"),
                favorite_property=kwargs.get("favorite_property"),
                conversation_history=kwargs.get("conversation_history"),
                photos_sent=kwargs.get("photos_sent"),
                places=_known_places(),
            )

        llm_router = llm_route

    flow = SalesFlow(
        lead_qualifier=LeadQualifier(),
        llm_classify_intent=llm_classifier,
        reply_generator=llm_reply,
        llm_router=llm_router,
        properties_rag=(lambda info: _search_properties(info, top_k=9))
        if _HAS_LLM
        else None,
        specialist_rotation=[
            s.strip()
            for s in os.environ.get("SPECIALIST_ROTATION", "").split(",")
            if s.strip()
        ],
        specialist_fallback=os.environ.get("SPECIALIST_FALLBACK", "diretor"),
        restriction_check=DynamoRestrictionCheck(
            dynamodb, os.environ.get("ALERTS_TABLE", "sdr-alerts")
        ),
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
