from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class SessionError(Exception):
    pass


def new_id() -> str:
    return str(uuid.uuid4())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def conversation_ttl_seconds() -> int:
    return 90 * 24 * 60 * 60


def channel_user_id(email: str, lead_id: str) -> int:
    """Pseudo-id determinístico do canal para leads de portal.

    O Contrato 5 exige `telegram_user_id` inteiro no Lead; leads de portal não
    têm conta Telegram, então o id é derivado por hash de (e-mail, lead_id) —
    sem colisão com ids reais do Telegram e estável para auditoria.
    """
    digest = hashlib.sha256(f"{email.strip().lower()}:{lead_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 1_000_000_000


class SessionWriter:
    """Escritor do Contrato 5 (sessões): abre Lead + Conversation do lead de portal.

    Espelha o padrão de escrita do SessionStore do u1 (`PK=LEAD#<id>` com
    `SK=PROFILE` e `SK=CONV#<session_id>`), mesmo schema compartilhado — a
    conversa nasce em `greeting` com o contato extraído no `context`.
    """

    def __init__(self, dynamodb_client: Any, table_name: str) -> None:
        self._client = dynamodb_client
        self._table = table_name

    def open_session(self, contact: dict[str, Any]) -> dict[str, Any]:
        email = str(contact.get("email") or "").strip()
        if not email:
            raise SessionError("contact email required to open session")
        lead_id = new_id()
        session_id = new_id()
        user_id = channel_user_id(email, lead_id)
        now = utc_now_iso()
        lead_item = self._marshal(
            {
                "PK": f"LEAD#{lead_id}",
                "SK": "PROFILE",
                "lead_id": lead_id,
                "telegram_user_id": user_id,
                "status": "new",
                "created_at": now,
                "updated_at": now,
            }
        )
        conversation = {
            "session_id": session_id,
            "lead_id": lead_id,
            "messages": [],
            "context": {"channel": "email", "contact": self._contact_context(contact, email)},
            "current_state": "greeting",
            "pii_masked": False,
            "consent_recorded": False,
            "created_at": now,
            "ttl": conversation_ttl_seconds(),
        }
        conversation_item = self._marshal(
            {"PK": f"LEAD#{lead_id}", "SK": f"CONV#{session_id}", **conversation}
        )
        try:
            self._client.put_item(TableName=self._table, Item=lead_item)
            self._client.put_item(TableName=self._table, Item=conversation_item)
        except Exception as exc:
            logger.error("session store unavailable: %s", exc)
            raise SessionError("session store unavailable") from exc
        return {"lead_id": lead_id, "session_id": session_id, "telegram_user_id": user_id}

    @staticmethod
    def _contact_context(contact: dict[str, Any], email: str) -> dict[str, Any]:
        context: dict[str, Any] = {"email": email}
        for key in ("name", "phone"):
            value = contact.get(key)
            if value:
                context[key] = value
        return context

    @staticmethod
    def _marshal(item: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in item.items():
            if isinstance(value, bool):
                out[key] = {"BOOL": value}
            elif isinstance(value, (int, float)):
                out[key] = {"N": str(value)}
            elif isinstance(value, (list, dict)):
                out[key] = {"S": json.dumps(value, default=str)}
            else:
                out[key] = {"S": str(value)}
        return out
