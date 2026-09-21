from __future__ import annotations

import json
import logging
import os
from typing import Any

from infra.alert_store import AlertStoreReader
from infra.conversation_store import ConversationStore
from infra.metrics import CloudWatchMeter
from service.kpis import KpiService, utc_now

logger = logging.getLogger()
logger.setLevel(logging.INFO)

API_PATH = "/api/kpis"
_UNAUTHORIZED_BODY = {"message": "não autenticado — faça login via Cognito"}
_ERROR_BODY = {"message": "erro interno ao agregar KPIs"}
_NOT_FOUND_BODY = {"message": "recurso não encontrado"}
_METHOD_BODY = {"message": "método não suportado"}


def _response(status_code: int, body: dict[str, Any], origin: str) -> dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": origin,
        },
        "body": json.dumps(body, ensure_ascii=False, default=str),
    }


def _bearer_token(event: dict[str, Any]) -> str | None:
    headers = {
        str(key).lower(): value
        for key, value in (event.get("headers") or {}).items()
        if value is not None
    }
    authorization = str(headers.get("authorization") or "")
    if authorization.startswith("Bearer "):
        token = authorization[len("Bearer ") :].strip()
        return token or None
    return None


def build_service(client: Any, env: dict[str, str] | None = None) -> KpiService:
    """Wiring DI completo — testável com cliente fake (zero AWS)."""
    environment = env if env is not None else os.environ
    return KpiService(
        conversations=ConversationStore(client, environment.get("SESSIONS_TABLE", "sdr-sessions")),
        alerts=AlertStoreReader(client, environment.get("ALERTS_TABLE", "sdr-alerts")),
        meter=CloudWatchMeter(
            client,
            namespace=environment.get("CW_NAMESPACE", "SdrApp"),
            response_metric=environment.get("CW_RESPONSE_METRIC", "ResponseTimeP90"),
            cost_metric=environment.get("CW_COST_METRIC", "CostMonthly"),
            now_fn=utc_now,
        ),
        now_fn=utc_now,
    )


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    origin = os.environ.get("DASHBOARD_ALLOWED_ORIGIN", "*")
    if not isinstance(event, dict):
        logger.error("unexpected lambda payload type: %s", type(event).__name__)
        return _response(500, _ERROR_BODY, origin)
    method = str(event.get("httpMethod") or event.get("requestContext", {}).get("http", {}).get("method") or "").upper()
    path = str(event.get("path") or event.get("rawPath") or "")
    if not path.endswith(API_PATH):
        return _response(404, _NOT_FOUND_BODY, origin)
    if method != "GET":
        return _response(405, _METHOD_BODY, origin)
    if _bearer_token(event) is None:
        logger.warning("kpis request without bearer token")
        return _response(401, _UNAUTHORIZED_BODY, origin)
    try:
        import boto3

        client = boto3.client("dynamodb")
        service = build_service(client)
        snapshot = service.snapshot()
    except Exception as exc:
        logger.error("kpi aggregation failed: %s", exc)
        return _response(500, _ERROR_BODY, origin)
    return _response(200, snapshot, origin)
