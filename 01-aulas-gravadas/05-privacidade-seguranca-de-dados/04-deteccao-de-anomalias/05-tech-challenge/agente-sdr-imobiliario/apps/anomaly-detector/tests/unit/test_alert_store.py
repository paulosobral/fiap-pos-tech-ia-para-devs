import json
from unittest.mock import MagicMock

import pytest

from infra.alert_store import (
    ACTION_ALERT_ISSUED,
    ACTION_SCHEDULE_RESTRICTED,
    ALERT_STATUS_OPEN,
    AlertStore,
    AlertStoreError,
)


def anomaly_item(anomaly_id="a-1", lead_id="l-1", **overrides):
    item = {
        "anomaly_id": anomaly_id,
        "lead_id": lead_id,
        "features": {"message_volume": 30, "negative_sentiment_ratio": 1.0},
        "confidence": 0.775,
        "type": "negative_sentiment",
        "detected_at": "2026-09-20T03:30:00+00:00",
        "status": ALERT_STATUS_OPEN,
        "action_taken": ACTION_ALERT_ISSUED,
    }
    item.update(overrides)
    return item


def raw_item(item):
    from infra.alert_store import marshal_item

    return marshal_item(item)


def make_store(query_items=None, fail=False):
    client = MagicMock()
    if fail:
        client.put_item.side_effect = RuntimeError("dynamo down")
        client.update_item.side_effect = RuntimeError("dynamo down")
        client.query.side_effect = RuntimeError("dynamo down")
    else:
        client.query.return_value = {"Items": [raw_item(i) for i in (query_items or [])]}
    return AlertStore(client, "sdr-alerts"), client


class TestAlertStore:
    def test_save_anomaly_marshals_contract_7_fields(self):
        store, client = make_store()
        item = anomaly_item()
        result = store.save_anomaly(item)
        assert result == item
        saved = client.put_item.call_args.kwargs
        assert saved["TableName"] == "sdr-alerts"
        assert saved["Item"]["PK"] == {"S": "a-1"}
        assert saved["Item"]["lead_id"] == {"S": "l-1"}
        assert saved["Item"]["confidence"] == {"N": "0.775"}
        assert json.loads(saved["Item"]["features"]["S"]) == item["features"]
        assert saved["Item"]["status"] == {"S": "open"}
        assert saved["Item"]["action_taken"] == {"S": ACTION_ALERT_ISSUED}

    def test_save_requires_anomaly_id_and_lead_id(self):
        store, client = make_store()
        with pytest.raises(AlertStoreError):
            store.save_anomaly({"lead_id": "l-1"})
        with pytest.raises(AlertStoreError):
            store.save_anomaly({"anomaly_id": "a-1"})
        client.put_item.assert_not_called()

    def test_save_failure_raises_alert_store_error(self):
        store, _ = make_store(fail=True)
        with pytest.raises(AlertStoreError, match="alert store unavailable"):
            store.save_anomaly(anomaly_item())

    def test_restrict_scheduling_updates_anomaly_item(self):
        store, client = make_store()
        store.restrict_scheduling("a-1")
        kwargs = client.update_item.call_args.kwargs
        assert kwargs["Key"] == {"PK": {"S": "a-1"}}
        assert kwargs["ExpressionAttributeValues"][":t"] == {"BOOL": True}
        assert kwargs["ExpressionAttributeValues"][":a"] == {"S": ACTION_SCHEDULE_RESTRICTED}

    def test_find_open_restriction_returns_latest_restricted_open_item(self):
        older = anomaly_item("a-1", detected_at="2026-09-19T03:30:00+00:00")
        newer = anomaly_item("a-2", detected_at="2026-09-20T03:30:00+00:00", scheduling_restricted=True)
        store, client = make_store([older, newer])
        found = store.find_open_restriction("l-1")
        assert found["anomaly_id"] == "a-2"
        assert found["scheduling_restricted"] is True
        assert client.query.call_args.kwargs["IndexName"] == "lead-index"

    def test_find_open_restriction_ignores_unrestricted_items(self):
        store, _ = make_store([anomaly_item()])
        assert store.find_open_restriction("l-1") is None
        assert store.is_scheduling_restricted("l-1") is False

    def test_query_failure_raises_alert_store_error(self):
        store, _ = make_store(fail=True)
        with pytest.raises(AlertStoreError):
            store.find_open_restriction("l-1")
