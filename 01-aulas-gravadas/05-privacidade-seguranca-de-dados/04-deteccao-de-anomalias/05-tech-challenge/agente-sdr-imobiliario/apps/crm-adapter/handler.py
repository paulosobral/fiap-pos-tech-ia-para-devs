from __future__ import annotations

import logging
import os
from typing import Any

from infra.csv_store import CsvStore
from infra.session_store import SessionLookup
from service.crm_adapter import CrmAdapter
from service.crm_gateway import CsvCrmGateway
from service.flow_gateway import HttpFlowGateway
from service.status_sync import StatusSync

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

    crm = CsvCrmGateway(
        CsvStore(os.environ.get("CRM_CSV_PATH", "/tmp/crm-leads.csv"))
    )
    sessions = SessionLookup(
        boto3.client("dynamodb"), os.environ.get("SESSIONS_TABLE", "sdr-sessions")
    )
    flow = HttpFlowGateway(
        _http_client(),
        base_url=_env("FLOW_BASE_URL"),
        secret_token=os.environ.get("INTERNAL_SECRET_TOKEN"),
    )
    status = StatusSync(crm, flow)
    max_receives = int(os.environ.get("CRM_MAX_RECEIVES", "3"))
    adapter = CrmAdapter(
        crm=crm, status=status, flow=flow, sessions=sessions, max_receives=max_receives
    )
    return adapter.handle_event(event)