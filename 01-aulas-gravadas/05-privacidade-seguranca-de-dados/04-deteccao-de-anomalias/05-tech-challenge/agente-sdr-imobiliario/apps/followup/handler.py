from __future__ import annotations

import logging
import os
from typing import Any

from infra.conversation_store import ConversationStore
from infra.followup_state import FollowupStateStore
from infra.silence_window import SilenceWindow
from service.cadence import CadenceCalculator, parse_cadence_days, utc_now
from service.duplicate_guard import DuplicateGuard
from service.followup import FollowupService
from service.message_builder import FollowupMessageBuilder
from service.telegram_gateway import TelegramGateway

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

    if not isinstance(event, dict):
        logger.warning("unexpected EventBridge payload; running scheduled follow-up anyway")
    client = boto3.client("dynamodb")
    state_store = FollowupStateStore(
        client, os.environ.get("FOLLOWUP_TABLE", "sdr-followup-state")
    )
    service = FollowupService(
        conversations=ConversationStore(
            client, os.environ.get("SESSIONS_TABLE", "sdr-sessions")
        ),
        state_store=state_store,
        guard=DuplicateGuard(state_store),
        cadence=CadenceCalculator(
            days=parse_cadence_days(os.environ.get("FOLLOWUP_CADENCE_DAYS")), now_fn=utc_now
        ),
        silence_window=SilenceWindow.from_env(now_fn=utc_now),
        builder=FollowupMessageBuilder(),
        telegram=TelegramGateway(_http_client(), bot_token=_env("TELEGRAM_BOT_TOKEN")),
        now_fn=utc_now,
    )
    return service.run()
