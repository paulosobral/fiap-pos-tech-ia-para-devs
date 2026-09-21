from __future__ import annotations

from typing import Any

from service.cadence import parse_iso8601


class DuplicateGuard:
    """Guarda anti-spam (FR8.3): um lead nunca recebe o mesmo passo duas vezes
    nem mensagem depois de ter respondido ao follow-up anterior.

    - passo já enviado (`last_step >= step`) → idempotência do tick EventBridge;
    - `last_lead_message_at` posterior ao `last_followup_at` registrado → a
      conversa foi retomada pelo lead (a cadência para de remarcar).
    """

    def __init__(self, state_store: Any) -> None:
        self._state_store = state_store

    def evaluate(self, lead_id: str, step: int, last_lead_message_at: Any = None) -> str | None:
        state = self._state_store.get_state(lead_id) or {}
        if int(state.get("last_step", 0)) >= step:
            return "already_followed_up"
        last_sent = state.get("last_followup_at")
        if last_sent and last_lead_message_at:
            sent_at = parse_iso8601(last_sent)
            replied_at = parse_iso8601(last_lead_message_at)
            if sent_at and replied_at and replied_at > sent_at:
                return "lead_replied"
        return None
