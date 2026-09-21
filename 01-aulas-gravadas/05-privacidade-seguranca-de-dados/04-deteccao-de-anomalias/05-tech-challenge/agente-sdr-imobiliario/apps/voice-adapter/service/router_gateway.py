from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class RouterError(Exception):
    pass


class RouterGateway(Protocol):
    def reinject(self, telegram_user_id: int, session_id: str, text: str) -> None: ...


class HttpRouterGateway:
    """Re-injeção no ConversationRouter (contrato interno real da U1):
    POST {base}/internal/inbound-text com header X-Internal-Secret == env
    INTERNAL_SECRET_TOKEN e body {"telegram_user_id": int, "session_id": str, "text": str}.
    """

    def __init__(
        self,
        http: Any,
        base_url: str,
        secret_token: str | None = None,
        timeout: int = 10,
    ) -> None:
        if not secret_token:
            raise RouterError("internal secret is required for re-injection")
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._secret = secret_token
        self._timeout = timeout

    def reinject(self, telegram_user_id: int, session_id: str, text: str) -> None:
        headers = {"Content-Type": "application/json", "X-Internal-Secret": self._secret}
        try:
            response = self._http.post(
                f"{self._base_url}/internal/inbound-text",
                json={
                    "telegram_user_id": telegram_user_id,
                    "session_id": session_id,
                    "text": text,
                },
                headers=headers,
                timeout=self._timeout,
            )
        except Exception as exc:
            logger.error("router unreachable: %s", exc)
            raise RouterError("router unreachable") from exc
        if response.status_code >= 300:
            logger.error("router rejected re-injection with status %s", response.status_code)
            raise RouterError(f"router rejected with status {response.status_code}")
