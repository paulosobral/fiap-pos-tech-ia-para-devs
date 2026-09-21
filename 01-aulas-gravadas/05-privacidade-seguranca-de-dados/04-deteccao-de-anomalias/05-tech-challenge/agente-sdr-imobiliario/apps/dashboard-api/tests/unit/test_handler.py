import json
import sys
from datetime import datetime, timezone

import handler as handler_module
from handler import handler


class FakeAwsClient:
    def __init__(self, pages_by_table=None, cw_results=None, scan_error=None):
        self._pages = pages_by_table or {}
        self._cw = cw_results or {}
        self._scan_error = scan_error
        self.calls: list[dict] = []

    def scan(self, **kwargs):
        self.calls.append(kwargs)
        if self._scan_error:
            raise self._scan_error
        raw = self._pages.get(kwargs["TableName"], [])
        if raw and isinstance(raw[0], dict) and "Items" in raw[0]:
            items = [item for page in raw for item in page.get("Items", [])]
        else:
            items = list(raw)
        prefix = (kwargs.get("ExpressionAttributeValues") or {}).get(":prefix", {}).get("S", "")
        if prefix:
            items = [
                item
                for item in items
                if str((item.get("SK") or {}).get("S", "")).startswith(prefix)
            ]
        return {"Items": items}

    def get_metric_data(self, **kwargs):
        query_id = kwargs["MetricDataQueries"][0]["Id"]
        return {"MetricDataResults": self._cw.get(query_id, [{"Values": []}])}


class FakeBoto3Module:
    def __init__(self, client):
        self._client = client

    def client(self, name):
        return self._client


def s(value):
    return {"S": str(value)}


def n(value):
    return {"N": str(value)}


def conv(session_id, lead_id, state, created_at):
    return {
        "PK": s(f"LEAD#{lead_id}"),
        "SK": s(f"CONV#{session_id}"),
        "session_id": s(session_id),
        "lead_id": s(lead_id),
        "current_state": s(state),
        "created_at": s(created_at),
        "context": {"S": json.dumps({})},
        "messages": {"S": json.dumps([{"role": "lead", "text": "quero visitar"}])},
    }


def lead(lead_id, created_at, **extra):
    item = {
        "PK": s(f"LEAD#{lead_id}"),
        "SK": s("PROFILE"),
        "lead_id": s(lead_id),
        "telegram_user_id": n(42),
        "status": s("new"),
        "created_at": s(created_at),
        "updated_at": s(created_at),
    }
    for key, value in extra.items():
        item[key] = s(value)
    return item


def alert(anomaly_id, detected_at):
    return {
        "PK": s(anomaly_id),
        "anomaly_id": s(anomaly_id),
        "lead_id": s("l1"),
        "type": s("volume_spike"),
        "confidence": n(0.9),
        "detected_at": s(detected_at),
        "status": s("open"),
        "action_taken": s("alert_issued"),
        "features": {"S": json.dumps({"message_count": 9})},
    }


def get_event(path="/api/kpis", method="GET", token="jwt-poc"):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return {"httpMethod": method, "path": path, "headers": headers}


FROZEN_NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)


def seed_env(monkeypatch):
    monkeypatch.setenv("SESSIONS_TABLE", "sdr-sessions")
    monkeypatch.setenv("ALERTS_TABLE", "sdr-alerts")
    monkeypatch.setenv("CW_NAMESPACE", "SdrApp")
    monkeypatch.setattr(handler_module, "utc_now", lambda: FROZEN_NOW)


def install_fake_boto3(monkeypatch, client):
    monkeypatch.setitem(sys.modules, "boto3", FakeBoto3Module(client))


def make_client():
    return FakeAwsClient(
        pages_by_table={
            "sdr-sessions": [
                {
                    "Items": [
                        conv("s1", "l1", "scheduling", "2026-09-20T10:00:00Z"),
                        conv("s2", "l2", "greeting", "2026-09-19T10:00:00Z"),
                        lead("l1", "2026-09-20T09:00:00Z", intent="compra", area="800 m²"),
                        lead("l2", "2026-09-19T09:00:00Z", intent="locação", area="300 m²"),
                    ]
                }
            ],
            "sdr-alerts": [
                {"Items": [alert("a1", "2026-09-20T11:00:00Z"), alert("a2", "2026-09-15T11:00:00Z")]}
            ],
        },
        cw_results={
            "response_p90": [{"Values": [0.8]}],
            "cost_monthly": [{"Values": [15.0]}],
        },
    )


class TestHandler:
    def test_get_kpis_returns_200_contract_payload(self, monkeypatch):
        seed_env(monkeypatch)
        install_fake_boto3(monkeypatch, make_client())
        response = handler(get_event())
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        for key in (
            "leads_today", "leads_week", "response_time_p90", "qualification_rate",
            "scheduled_visits", "anomalies_count", "cost_monthly",
        ):
            assert key in body
        assert body["leads_today"] == 1
        assert body["leads_week"] == 2
        assert body["anomalies_count"] == 1
        assert body["funnel"]["scheduling"] == 1
        assert body["alerts"][0]["anomaly_id"] == "a1"

    def test_cors_header_default_and_override(self, monkeypatch):
        seed_env(monkeypatch)
        install_fake_boto3(monkeypatch, make_client())
        assert handler(get_event())["headers"]["Access-Control-Allow-Origin"] == "*"
        monkeypatch.setenv("DASHBOARD_ALLOWED_ORIGIN", "https://dash.streamlit.app")
        assert (
            handler(get_event())["headers"]["Access-Control-Allow-Origin"]
            == "https://dash.streamlit.app"
        )

    def test_missing_authorization_returns_401(self, monkeypatch):
        seed_env(monkeypatch)
        install_fake_boto3(monkeypatch, make_client())
        response = handler(get_event(token=None))
        assert response["statusCode"] == 401
        assert "Cognito" in json.loads(response["body"])["message"]

    def test_malformed_authorization_returns_401(self, monkeypatch):
        seed_env(monkeypatch)
        install_fake_boto3(monkeypatch, make_client())
        event = get_event()
        event["headers"] = {"Authorization": "Token abc"}
        assert handler(event)["statusCode"] == 401
        event["headers"] = {"authorization": "Bearer "}
        assert handler(event)["statusCode"] == 401

    def test_non_get_method_returns_405(self, monkeypatch):
        seed_env(monkeypatch)
        install_fake_boto3(monkeypatch, make_client())
        assert handler(get_event(method="POST"))["statusCode"] == 405

    def test_unknown_path_returns_404(self, monkeypatch):
        seed_env(monkeypatch)
        install_fake_boto3(monkeypatch, make_client())
        assert handler(get_event(path="/api/outra"))["statusCode"] == 404

    def test_raw_path_fallback_api_gateway_v2(self, monkeypatch):
        seed_env(monkeypatch)
        install_fake_boto3(monkeypatch, make_client())
        event = get_event()
        event.pop("path")
        event["rawPath"] = "/api/kpis"
        assert handler(event)["statusCode"] == 200

    def test_store_failure_returns_500(self, monkeypatch, caplog):
        seed_env(monkeypatch)
        install_fake_boto3(
            monkeypatch, FakeAwsClient(scan_error=RuntimeError("dynamodb down"))
        )
        response = handler(get_event())
        assert response["statusCode"] == 500
        assert "erro interno" in json.loads(response["body"])["message"]
        assert "kpi aggregation failed" in caplog.text

    def test_non_dict_event_returns_500(self, monkeypatch):
        seed_env(monkeypatch)
        install_fake_boto3(monkeypatch, make_client())
        assert handler(None)["statusCode"] == 500
        assert handler("event")["statusCode"] == 500


class TestBuildService:
    def test_build_service_wires_di_with_env(self, monkeypatch):
        seed_env(monkeypatch)
        service = handler_module.build_service(make_client(), env={"SESSIONS_TABLE": "t1"})
        assert service.conversations._table == "t1"
        assert service.now_fn() == FROZEN_NOW
