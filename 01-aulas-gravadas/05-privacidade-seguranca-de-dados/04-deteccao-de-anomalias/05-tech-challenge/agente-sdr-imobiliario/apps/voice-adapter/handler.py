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


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required env var {name}")
    return value


def _http_client() -> Any:
    import requests

    return requests.Session()


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    import boto3

    http = _http_client()
    telegram = TelegramGateway(http, bot_token=_env("TELEGRAM_BOT_TOKEN"))
    transcriber = WhisperTranscriber()
    router = HttpRouterGateway(
        http,
        base_url=_env("ROUTER_BASE_URL"),
        secret_token=os.environ.get("INTERNAL_SECRET_TOKEN"),
    )
    sessions = SessionLookup(
        boto3.client("dynamodb"), os.environ.get("SESSIONS_TABLE", "sdr-sessions")
    )
    adapter = VoiceAdapter(
        telegram=telegram, transcriber=transcriber, router=router, sessions=sessions
    )
    return adapter.handle_event(event)
