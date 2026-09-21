from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class FlowError(Exception):
    pass


class FlowGateway(Protocol):
    def notify_status(self, lead_id: str, session_id: str, stage: str) -> None: ...


class HttpFlowGateway:
    """Callback que devolve o status da esteira Kanban ao fluxo (u1).

    Desacoplado: nenhum import de internals de `apps/conversation-router`.
    """

    def __init__(
        self,
        http: Any,
        base_url: str,
        secret_token: str | None = None,
        timeout: int = 10,
    ) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._secret = secret_token
        self._timeout = timeout

    def notify_status(self, lead_id: str, session_id: str, stage: str) -> None:
        headers = {"Content-Type": "application/json"}
        if self._secret:
            headers["X-Internal-Secret"] = self._secret
        try:
            response = self._http.post(
                f"{self._base_url}/internal/crm-status",
                json={"lead_id": lead_id, "session_id": session_id, "stage": stage},
                headers=headers,
                timeout=self._timeout,
            )
        except Exception as exc:
            logger.error("flow unreachable: %s", exc)
            raise FlowError("flow unreachable") from exc
        if response.status_code >= 300:
            logger.error("flow rejected status notification with status %s", response.status_code)
            raise FlowError(f"flow rejected with status {response.status_code}")