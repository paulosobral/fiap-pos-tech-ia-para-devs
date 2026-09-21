import json

import pytest

from infra.alert_store import AlertStoreError, AlertStoreReader, project_alert


def s(value):
    return {"S": str(value)}


def n(value):
    return {"N": str(value)}


def alert_item(anomaly_id="a1", lead_id="l1", detected_at="2026-09-20T11:00:00Z", confidence=0.87):
    return {
        "PK": s(anomaly_id),
        "anomaly_id": s(anomaly_id),
        "lead_id": s(lead_id),
        "type": s("volume_spike"),
        "confidence": n(confidence),
        "detected_at": s(detected_at),
        "status": s("open"),
        "action_taken": s("alert_issued"),
        "features": {"S": json.dumps({"message_count": 9, "sentiment": -0.4})},
    }


class FakeScanClient:
    def __init__(self, pages):
        self._pages = pages
        self._cursor = 0

    def scan(self, **kwargs):
        index = self._cursor
        self._cursor += 1
        if index >= len(self._pages):
            return {"Items": []}
        payload = self._pages[index]
        response = {"Items": payload.get("Items", [])}
        if payload.get("LastEvaluatedKey"):
            response["LastEvaluatedKey"] = payload["LastEvaluatedKey"]
        return response


class FailingClient:
    def scan(self, **kwargs):
        raise RuntimeError("dynamodb down")


class TestAlertStoreReader:
    def test_lists_sorted_desc_and_unmarshals(self):
        reader = AlertStoreReader(
            FakeScanClient([{"Items": [alert_item(anomaly_id="old", detected_at="2026-09-19T00:00:00Z"), alert_item()]}])
        )
        alerts = reader.list_alerts()
        assert [alert["anomaly_id"] for alert in alerts] == ["a1", "old"]
        assert alerts[0]["confidence"] == 0.87
        assert alerts[0]["features"] == {"message_count": 9, "sentiment": -0.4}

    def test_pagination(self):
        reader = AlertStoreReader(
            FakeScanClient(
                [
                    {"Items": [alert_item(anomaly_id="a1", detected_at="2026-09-20T10:00:00Z")], "LastEvaluatedKey": {"PK": {"S": "a1"}}},
                    {"Items": [alert_item(anomaly_id="a2", detected_at="2026-09-20T12:00:00Z")]},
                ]
            )
        )
        assert [alert["anomaly_id"] for alert in reader.list_alerts()] == ["a2", "a1"]

    def test_missing_anomaly_id_skipped_with_log(self, caplog):
        broken = alert_item(anomaly_id="a9")
        broken.pop("anomaly_id")
        reader = AlertStoreReader(FakeScanClient([{"Items": [broken, alert_item()]}]))
        alerts = reader.list_alerts()
        assert [alert["anomaly_id"] for alert in alerts] == ["a1"]
        assert "missing anomaly_id" in caplog.text

    def test_client_error_raises_alert_store_error(self):
        with pytest.raises(AlertStoreError):
            AlertStoreReader(FailingClient()).list_alerts()

    def test_project_alert_strips_features(self):
        projection = project_alert(
            {"anomaly_id": "a1", "lead_id": "l1", "type": "volume_spike", "confidence": 0.9,
             "detected_at": "2026-09-20T11:00:00Z", "status": "open", "action_taken": "alert_issued",
             "features": {"secret": "nada"}}
        )
        assert set(projection) == {
            "anomaly_id", "lead_id", "type", "confidence", "detected_at", "status", "action_taken"
        }
        assert "features" not in projection

    def test_default_table_name_matches_u5_owner(self):
        reader = AlertStoreReader(FakeScanClient([{"Items": []}]))
        reader.list_alerts()
