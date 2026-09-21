from __future__ import annotations

import logging
import os
from typing import Any

from infra.session_store import SessionLookup
from service.router_gateway import HttpRouterGateway
from service.telegram_gateway import TelegramGateway
from service.transcriber import WhisperTranscriber
from service.voice_adapter import VoiceAdapter

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_transcriber: WhisperTranscriber | None = None


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required env var {name}")
    return value


def _http_client() -> Any:
    import requests

    return requests.Session()


def get_transcriber() -> WhisperTranscriber:
    """Singleton lazy em nível de módulo: a mesma instância sobrevive entre invocações
    warm do Lambda, então o modelo Whisper (carregado no primeiro uso) não é recarregado
    por mensagem."""
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperTranscriber()
    return _transcriber


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    import boto3

    http = _http_client()
    telegram = TelegramGateway(http, bot_token=_env("TELEGRAM_BOT_TOKEN"))
    transcriber = get_transcriber()
    router = HttpRouterGateway(
        http,
        base_url=_env("ROUTER_BASE_URL"),
        secret_token=_env("INTERNAL_SECRET_TOKEN"),
    )
    sessions = SessionLookup(
        boto3.client("dynamodb"), os.environ.get("SESSIONS_TABLE", "sdr-sessions")
    )
    adapter = VoiceAdapter(
        telegram=telegram, transcriber=transcriber, router=router, sessions=sessions
    )
    return adapter.handle_event(event)
