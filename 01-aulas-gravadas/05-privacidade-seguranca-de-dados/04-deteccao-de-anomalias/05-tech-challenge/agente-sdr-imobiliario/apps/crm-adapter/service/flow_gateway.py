from __future__ import annotations

import json
import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, json.dumps({"event": event, **fields}, default=str))


class FlowError(Exception):
    pass


class FlowGateway(Protocol):
    def notify_status(self, lead_id: str, session_id: str, stage: str) -> None: ...


class HttpFlowGateway:
    """Callback que devolve o status da esteira Kanban ao fluxo.

    Receptor REAL (u1, apps/conversation-router/handler.py `POST /internal/crm-status`):
    método POST estrito, header `X-Internal-Secret` obrigatório (env
    INTERNAL_SECRET_TOKEN do receptor) e body `{"lead_id": str, "session_id": str,
    "stage": str}` — todos obrigatórios, `stage` ∈ KANBAN_STAGE_ORDER da u1.
    Respostas do receptor: 200/400/401/404/405/503 — só 2xx é aceito aqui.
    Desacoplado: nenhum import de internals de `apps/conversation-router`.
    """

    def __init__(
        self,
        http: Any,
        base_url: str,
        secret_token: str,
        timeout: int = 10,
    ) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._secret = secret_token
        self._timeout = timeout

    def notify_status(self, lead_id: str, session_id: str, stage: str) -> None:
        headers = {"Content-Type": "application/json", "X-Internal-Secret": self._secret}
        try:
            response = self._http.post(
                f"{self._base_url}/internal/crm-status",
                json={"lead_id": lead_id, "session_id": session_id, "stage": stage},
                headers=headers,
                timeout=self._timeout,
            )
        except Exception as exc:
            log_event("flow_unreachable", level=logging.ERROR, error=str(exc))
            raise FlowError("flow unreachable") from exc
        if response.status_code >= 300:
            log_event("flow_rejected", level=logging.ERROR, status=response.status_code)
            raise FlowError(f"flow rejected with status {response.status_code}")
