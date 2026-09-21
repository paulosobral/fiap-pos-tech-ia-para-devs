from __future__ import annotations

import logging
import os
from typing import Any

from infra.dedupe_store import DedupeStore
from infra.session_store import SessionWriter
from service.contact_ingest import ContactIngest, PendingRetryError
from service.email_parser import HeuristicEmailParser
from service.router_gateway import HttpRouterGateway

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

    parser = HeuristicEmailParser()
    dedupe = DedupeStore(
        boto3.client("dynamodb"), os.environ.get("DEDUPE_TABLE", "sdr-ingest-dedupe")
    )
    sessions = SessionWriter(
        boto3.client("dynamodb"), os.environ.get("SESSIONS_TABLE", "sdr-sessions")
    )
    router = HttpRouterGateway(
        _http_client(),
        base_url=_env("ROUTER_BASE_URL"),
        secret_token=_env("INTERNAL_SECRET_TOKEN"),
    )
    ingest = ContactIngest(parser=parser, dedupe=dedupe, sessions=sessions, router=router)
    summary = ingest.handle_event(event)
    if summary["retry"]:
        raise PendingRetryError(f"{summary['retry']} record(s) pending retry")
    return summary
