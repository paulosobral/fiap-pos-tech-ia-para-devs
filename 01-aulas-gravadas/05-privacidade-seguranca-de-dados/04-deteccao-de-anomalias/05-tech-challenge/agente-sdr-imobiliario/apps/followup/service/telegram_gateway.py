from __future__ import annotations

from typing import Any


class GatewayError(Exception):
    pass


class TelegramGateway:
    """Envio do follow-up via Telegram Bot API `sendMessage` (mesmo mecanismo de
    envio da U1 — o Contrato 1 cobre apenas o webhook de entrada).

    HTTP injetado (padrão do voice-adapter). Diferente do voice-adapter, o
    erro é propagado (`GatewayError`): o orquestrador não grava o estado do
    passo, e o próximo tick repete o follow-up (retry assíncrono da Lambda).
    A mensagem do `GatewayError` nunca inclui a URL/token do bot — apenas o
    tipo da exceção ou o status HTTP (PII-safe nos logs).
    """

    def __init__(
        self,
        http: Any,
        bot_token: str,
        base_url: str = "https://api.telegram.org",
        timeout: int = 10,
    ) -> None:
        self._http = http
        self._token = bot_token
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def send_message(self, chat_id: Any, text: str) -> None:
        try:
            response = self._http.post(
                f"{self._base_url}/bot{self._token}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=self._timeout,
            )
        except Exception as exc:
            raise GatewayError(f"sendMessage failed: {type(exc).__name__}") from exc
        if not response.ok:
            raise GatewayError(f"sendMessage rejected with status {response.status_code}")
