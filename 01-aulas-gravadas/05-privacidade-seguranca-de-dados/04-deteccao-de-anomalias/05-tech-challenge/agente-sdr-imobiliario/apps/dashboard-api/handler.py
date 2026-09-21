from __future__ import annotations

import json
import logging
import os
from typing import Any

from infra.alert_store import AlertStoreReader
from infra.conversation_store import ConversationStore
from infra.metrics import CloudWatchMeter
from logs import log_event
from service.kpis import KpiService, utc_now

logging.getLogger().setLevel(logging.INFO)

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


def build_service(
    dynamodb_client: Any, cloudwatch_client: Any, env: dict[str, str] | None = None
) -> KpiService:
    """Wiring DI completo — um cliente boto3 por serviço AWS (testável, zero AWS).

    DynamoDB (Contratos 5/7) e CloudWatch (Contrato 2) são serviços distintos:
    cada cliente expõe apenas a API do próprio serviço. O mesmo objeto NÃO
    pode servir aos dois componentes — um cliente dynamodb não possui
    `get_metric_data` (e o erro seria engolido pela degradação graciosa do
    meter, deixando `response_time_p90`/`cost_monthly` em 0.0 para sempre)."""
    environment = env if env is not None else os.environ
    return KpiService(
        conversations=ConversationStore(dynamodb_client, environment.get("SESSIONS_TABLE", "sdr-sessions")),
        alerts=AlertStoreReader(dynamodb_client, environment.get("ALERTS_TABLE", "sdr-alerts")),
        meter=CloudWatchMeter(
            cloudwatch_client,
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
        log_event(
            "unexpected_lambda_payload", level=logging.ERROR, payload_type=type(event).__name__
        )
        return _response(500, _ERROR_BODY, origin)
    method = str(event.get("httpMethod") or event.get("requestContext", {}).get("http", {}).get("method") or "").upper()
    path = str(event.get("path") or event.get("rawPath") or "")
    if not path.endswith(API_PATH):
        return _response(404, _NOT_FOUND_BODY, origin)
    if method != "GET":
        return _response(405, _METHOD_BODY, origin)
    if _bearer_token(event) is None:
        log_event("kpis_request_unauthorized", level=logging.WARNING)
        return _response(401, _UNAUTHORIZED_BODY, origin)
    try:
        import boto3

        service = build_service(
            boto3.client("dynamodb"),
            boto3.client("cloudwatch"),
        )
        snapshot = service.snapshot()
    except Exception as exc:
        log_event("kpi_aggregation_failed", level=logging.ERROR, error=str(exc))
        return _response(500, _ERROR_BODY, origin)
    return _response(200, snapshot, origin)
