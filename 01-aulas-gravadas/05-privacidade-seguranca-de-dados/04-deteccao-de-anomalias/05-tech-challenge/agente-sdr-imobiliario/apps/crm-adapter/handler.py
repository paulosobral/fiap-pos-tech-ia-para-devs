from __future__ import annotations

import logging
import os
from typing import Any

import tracing
from infra.csv_store import CsvStore
from infra.session_store import SessionLookup
from service.crm_adapter import CrmAdapter
from service.crm_gateway import CsvCrmGateway
from service.flow_gateway import HttpFlowGateway
from service.status_sync import StatusSync

tracing.enable()

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


def _build_crm(boto3: Any) -> Any:
    """HubSpot via MCP quando o secret está configurado; senão o CRM simulado (CSV) da POC."""
    secret_id = os.environ.get("HUBSPOT_SECRET_ID")
    if secret_id:
        from service.hubspot_mcp import HubSpotCrmGateway, HubSpotMcpClient, SecretsManagerTokenStore

        store = SecretsManagerTokenStore(boto3.client("secretsmanager"), secret_id)
        return HubSpotCrmGateway(HubSpotMcpClient(store))
    return CsvCrmGateway(CsvStore(os.environ.get("CRM_CSV_PATH", "/tmp/crm-leads.csv")))


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    import boto3

    crm = _build_crm(boto3)
    sessions = SessionLookup(
        boto3.client("dynamodb"), os.environ.get("SESSIONS_TABLE", "sdr-sessions")
    )
    flow = HttpFlowGateway(
        _http_client(),
        base_url=_env("FLOW_BASE_URL"),
        # Obrigatória: o receptor real (/internal/crm-status da u1) rejeita 401 sem o header.
        secret_token=_env("INTERNAL_SECRET_TOKEN"),
    )
    status = StatusSync(crm, flow)
    max_receives = int(os.environ.get("CRM_MAX_RECEIVES", "3"))
    adapter = CrmAdapter(
        crm=crm, status=status, flow=flow, sessions=sessions, max_receives=max_receives
    )
    return adapter.handle_event(event)