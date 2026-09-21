from __future__ import annotations

import json
import logging
import os
from typing import Any

from service.flow.lead_qualifier import LeadQualifier
from service.flow.sales_flow import SalesFlow
from service.security_layer import FALLBACK_MESSAGE, SecurityLayer
from service.entities import utc_now_iso
from infra.session_store import SessionStore

logger = logging.getLogger()
logger.setLevel(logging.INFO)


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
    ) -> None:
        self.store = store
        self.security = security_layer
        self.flow = sales_flow
        self.sqs = sqs_client
        self.telegram = telegram_client
        self.voice_queue_url = voice_queue_url
        self.crm_queue_url = crm_queue_url
        self.secret_token = secret_token

    def validate_secret(self, headers: dict[str, str] | None, expected: str | None = None) -> bool:
        expected = expected or self.secret_token or os.environ.get("TELEGRAM_SECRET_TOKEN")
        if not expected:
            return False
        if not headers:
            return False
        normalized = {k.lower(): v for k, v in headers.items()}
        return normalized.get("x-telegram-bot-api-secret-token") == expected

    def handle(self, event: dict[str, Any]) -> dict[str, Any]:
        headers = event.get("headers") or {}
        if not self.validate_secret(headers):
            logger.warning("Webhook rejected: invalid secret")
            return {"statusCode": 401, "body": json.dumps({"error": "unauthorized"})}

        body = event.get("body")
        try:
            update = json.loads(body) if isinstance(body, str) else body
        except json.JSONDecodeError:
            return {"statusCode": 400, "body": json.dumps({"error": "invalid payload"})}
        message = (update or {}).get("message") or {}
        if not message:
            return {"statusCode": 400, "body": json.dumps({"error": "invalid payload"})}

        sender = message.get("from") or {}
        chat = message.get("chat") or {}
        telegram_user_id = sender.get("id")
        text = message.get("text", "")
        if telegram_user_id is None:
            return {"statusCode": 400, "body": json.dumps({"error": "invalid payload"})}

        lead, conversation, created = self.store.get_or_create(telegram_user_id)
        if created and text.strip().lower() != "/start":
            pass

        if text.strip().lower() == "/start" and not conversation.consent_recorded:
            conversation.consent_recorded = True

        violation, reason = self.security.guard(text)
        if violation:
            logger.warning("Guardrail violation: %s", reason)
            response = FALLBACK_MESSAGE
            state = conversation.current_state
        else:
            masked = self.security.mask(text, session_id=conversation.session_id)
            conversation.messages.append(
                {"role": "lead", "text": masked, "at": utc_now_iso()}
            )
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
            lead.intent = flow_state.get("intent", lead.intent)
            score = flow_state.get("score")
            if score is not None:
                lead.score = score
            if flow_state.get("lead_qualified"):
                lead.status = "qualified"
                self._enqueue_crm(lead, conversation)
            leak, _ = self.security.check_output_leak(response)
            if leak:
                logger.error("PII leakage in response; using fallback")
                response = FALLBACK_MESSAGE
            conversation.context = flow_state.get("context", conversation.context)

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

    def _enqueue_crm(self, lead: Any, conversation: Any) -> None:
        if not self.sqs or not self.crm_queue_url:
            return
        self.sqs.send_message(
            QueueUrl=self.crm_queue_url,
            MessageBody=json.dumps(
                {
                    "message_id": conversation.session_id,
                    "lead_id": lead.lead_id,
                    "lead_data": {
                        "name": lead.decision_maker,
                        "email": None,
                        "phone": None,
                        "score": lead.score,
                        "urgency": lead.urgency,
                        "intent": lead.intent,
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
    security = SecurityLayer()
    flow = SalesFlow(lead_qualifier=LeadQualifier())
    router = ConversationRouter(
        store=store,
        security_layer=security,
        sales_flow=flow,
        sqs_client=sqs,
        voice_queue_url=os.environ.get("VOICE_QUEUE_URL"),
        crm_queue_url=os.environ.get("CRM_QUEUE_URL"),
    )
    return router.handle(event)
