from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = (
    "Desculpe, não consegui processar seu áudio. Tente novamente ou digite sua mensagem."
)


class GatewayError(Exception):
    pass


class TelegramGateway:
    def __init__(self, http: Any, bot_token: str, base_url: str = "https://api.telegram.org") -> None:
        self._http = http
        self._token = bot_token
        self._base_url = base_url.rstrip("/")

    def get_file_path(self, voice_file_id: str) -> str:
        try:
            response = self._http.get(
                f"{self._base_url}/bot{self._token}/getFile",
                params={"file_id": voice_file_id},
                timeout=10,
            )
            payload = response.json()
        except Exception as exc:
            raise GatewayError(f"getFile failed: {exc}") from exc
        if not response.ok or not payload.get("ok"):
            raise GatewayError(f"getFile rejected: {payload}")
        file_path = (payload.get("result") or {}).get("file_path")
        if not file_path:
            raise GatewayError("getFile returned no file_path")
        return file_path

    def download(self, file_path: str) -> bytes:
        try:
            response = self._http.get(
                f"{self._base_url}/file/bot{self._token}/{file_path}", timeout=30
            )
        except Exception as exc:
            raise GatewayError(f"download failed: {exc}") from exc
        if not response.ok:
            raise GatewayError(f"download rejected with status {response.status_code}")
        return response.content

    def send_message(self, chat_id: Any, text: str) -> None:
        try:
            response = self._http.post(
                f"{self._base_url}/bot{self._token}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=10,
            )
            if not response.ok:
                logger.error("sendMessage failed with status %s", response.status_code)
        except Exception as exc:
            logger.error("sendMessage error: %s", exc)
