from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class RouterError(Exception):
    pass


class RouterGateway(Protocol):
    def reinject(self, telegram_user_id: int, session_id: str, text: str) -> None: ...


class HttpRouterGateway:
    """Re-injeta a 1ª mensagem do lead de portal no ConversationRouter (u1).

    Contrato interno real da u1 (`POST /internal/inbound-text`): autenticação
    obrigatória pelo header `X-Internal-Secret` == env `INTERNAL_SECRET_TOKEN`
    (sem header → 401; env ausente no router → 503), body JSON
    `{"telegram_user_id": int, "session_id": str, "text": str}` (todos
    obrigatórios; `telegram_user_id` inteiro não-booleano). Qualquer status
    >= 300 (400/401/405/503/5xx) vira `RouterError` (falha transitória →
    outcome `retry`). Sem secret configurado, falha alta e cedo (fail-fast)
    em vez de enviar requisição sem autenticação. Desacoplado: nenhum import
    de internals de `apps/conversation-router`.
    """

    def __init__(
        self,
        http: Any,
        base_url: str,
        secret_token: str,
        timeout: int = 10,
    ) -> None:
        if not secret_token:
            raise ValueError("secret_token obrigatório (X-Internal-Secret)")
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
            logger.error("router unreachable: %s", type(exc).__name__)
            raise RouterError("router unreachable") from exc
        if response.status_code >= 300:
            logger.error("router rejected re-injection with status %s", response.status_code)
            raise RouterError(f"router rejected with status {response.status_code}")
