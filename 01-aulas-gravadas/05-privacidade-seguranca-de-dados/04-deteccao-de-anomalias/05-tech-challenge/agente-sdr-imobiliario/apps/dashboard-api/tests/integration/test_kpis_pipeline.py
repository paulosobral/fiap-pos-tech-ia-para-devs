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


def seed(monkeypatch, client):
    monkeypatch.setenv("SESSIONS_TABLE", "sdr-sessions")
    monkeypatch.setenv("ALERTS_TABLE", "sdr-alerts")
    monkeypatch.setenv("CW_NAMESPACE", "SdrApp")
    monkeypatch.setattr(
        handler_module, "utc_now", lambda: datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)
    )
    monkeypatch.setitem(sys.modules, "boto3", FakeBoto3Module(client))


def full_client():
    return FakeAwsClient(
        pages_by_table={
            "sdr-sessions": [
                {
                    "Items": [
                        conv("s1", "l1", "scheduling", "2026-09-20T10:00:00Z"),
                        conv("s2", "l1", "greeting", "2026-09-14T10:00:00Z"),
                        conv("s3", "l2", "handoff", "2026-09-19T10:00:00Z"),
                        lead("l1", "2026-09-20T09:00:00Z", intent="compra", area="800 m²"),
                        lead("l2", "2026-09-18T09:00:00Z", intent="locação", area="300 m²"),
                    ]
                }
            ],
            "sdr-alerts": [
                {"Items": [alert("a1", "2026-09-20T11:00:00Z"), alert("a2", "2026-09-16T11:00:00Z")]}
            ],
        },
        cw_results={"response_p90": [{"Values": [0.8]}], "cost_monthly": [{"Values": [15.0]}]},
    )


class TestKpisPipeline:
    def test_end_to_end_handler_aggregates_all_sources(self, monkeypatch):
        seed(monkeypatch, full_client())
        response = handler(get_event())
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["leads_today"] == 1
        assert body["leads_week"] == 2
        assert body["qualification_rate"] == 1.0
        assert body["scheduled_visits"] == 2
        assert body["anomalies_count"] == 1
        assert body["response_time_p90"] == 0.8
        assert body["cost_monthly"] == 15.0
        assert body["intents"] == {"compra": 1, "locação": 1}
        assert body["route_distribution"] == {"diretor": 1, "consultores": 1}
        assert body["funnel"]["scheduling"] == 1
        assert body["funnel"]["handoff"] == 1

    def test_empty_tables_yield_graceful_zeros(self, monkeypatch):
        seed(monkeypatch, FakeAwsClient())
        response = handler(get_event())
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["leads_today"] == 0
        assert body["anomalies_count"] == 0
        assert body["alerts"] == []
        assert body["response_time_p90"] == 0.0
        assert body["cost_monthly"] == 0.0

    def test_malformed_items_skipped_never_crash(self, monkeypatch):
        broken_conv = conv("s9", "l9", "greeting", "2026-09-20T10:00:00Z")
        broken_conv.pop("session_id")
        broken_lead = lead("l9", "2026-09-20T09:00:00Z")
        broken_lead.pop("lead_id")
        broken_alert = alert("a9", "2026-09-20T11:00:00Z")
        broken_alert.pop("anomaly_id")
        seed(
            monkeypatch,
            FakeAwsClient(
                pages_by_table={
                    "sdr-sessions": [
                        {
                            "Items": [
                                broken_conv,
                                broken_lead,
                                conv("s1", "l1", "greeting", "2026-09-20T10:00:00Z"),
                                lead("l1", "2026-09-20T09:00:00Z"),
                            ]
                        }
                    ],
                    "sdr-alerts": [{"Items": [broken_alert]}],
                }
            ),
        )
        response = handler(get_event())
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["leads_week"] == 1
        assert body["anomalies_count"] == 0

    def test_pii_never_leaks_into_body_or_logs(self, monkeypatch, caplog):
        client = full_client()
        client._pages["sdr-sessions"][0]["Items"].append(
            {
                **conv("s4", "l3", "greeting", "2026-09-20T10:30:00Z"),
                "messages": {
                    "S": json.dumps(
                        [{"role": "lead", "text": "Sou Ana, meu e-mail é ana@exemplo.com, tel +55 11 98888-7777"}]
                    )
                },
            }
        )
        seed(monkeypatch, client)
        response = handler(get_event())
        body = response["body"]
        assert "ana@exemplo.com" not in body
        assert "+55 11 98888-7777" not in body
        assert "Sou Ana" not in body
        assert "ana@exemplo.com" not in caplog.text

    def test_store_down_returns_500(self, monkeypatch, caplog):
        seed(monkeypatch, FakeAwsClient(scan_error=RuntimeError("dynamodb down")))
        assert handler(get_event())["statusCode"] == 500
        assert "kpi aggregation failed" in caplog.text

    def test_cloudwatch_down_still_returns_200_with_zeros(self, monkeypatch):
        client = FakeAwsClient(
            pages_by_table=full_client()._pages,
            cw_results={},
        )

        def broken_cw(**kwargs):
            raise RuntimeError("cw down")

        client.get_metric_data = broken_cw
        seed(monkeypatch, client)
        response = handler(get_event())
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["response_time_p90"] == 0.0
        assert body["cost_monthly"] == 0.0
        assert body["leads_week"] == 2

    def test_malformed_metric_values_become_zeros(self, monkeypatch):
        seed(
            monkeypatch,
            FakeAwsClient(
                pages_by_table=full_client()._pages,
                cw_results={
                    "response_p90": [{"Values": ["alto"]}],
                    "cost_monthly": [{"Values": [None]}],
                },
            ),
        )
        body = json.loads(handler(get_event())["body"])
        assert body["response_time_p90"] == 0.0
        assert body["cost_monthly"] == 0.0
