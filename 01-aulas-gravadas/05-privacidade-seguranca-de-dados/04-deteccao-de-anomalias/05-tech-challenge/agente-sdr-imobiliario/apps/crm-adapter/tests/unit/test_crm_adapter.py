import json
from unittest.mock import MagicMock

from service.crm_adapter import CrmAdapter
from service.crm_gateway import CrmError
from service.flow_gateway import FlowError


def crm_message(**overrides):
    base = {
        "message_id": "m1",
        "lead_id": "lead-1",
        "lead_data": {
            "name": "Ana Ribeiro",
            "email": "ana@empresa.com",
            "phone": "+5511999990000",
            "score": 85,
            "urgency": "alta",
            "intent": "compra",
        },
        "session_id": "s1",
        "timestamp": "2026-09-20T00:00:00+00:00",
    }
    base.update(overrides)
    return base


def make_adapter(crm_ok=True, sessions=True):
    crm = MagicMock()
    crm.upsert_lead.return_value = {"crm_id": "crm-1", "lead_id": "lead-1"}
    if not crm_ok:
        crm.upsert_lead.side_effect = CrmError("crm down")
    flow = MagicMock()
    store = MagicMock()
    store.get_session.return_value = {"session_id": "s1"} if sessions else None
    adapter = CrmAdapter(crm=crm, status=MagicMock(), flow=flow, sessions=store)
    return adapter, crm, flow, store


class TestCrmAdapter:
    def test_happy_path_syncs_lead_and_stage(self):
        adapter, crm, flow, store = make_adapter()
        adapter.status.sync.return_value = {"lead_id": "lead-1", "session_id": "s1", "stage": "qualificado", "synced_at": "t"}
        outcome = adapter.process_message(crm_message())
        assert outcome == "ok"
        crm.upsert_lead.assert_called_once()
        payload = crm.upsert_lead.call_args[0][0]
        assert payload["lead_id"] == "lead-1"
        assert payload["email"] == "ana@empresa.com"
        adapter.status.sync.assert_called_once_with("lead-1", "s1", crm_message()["lead_data"])
        flow.notify_status.assert_not_called()

    def test_non_dict_lead_data_is_dropped(self):
        adapter, crm, flow, _ = make_adapter()
        message = crm_message()
        message["lead_data"] = "not a dict"
        assert adapter.process_message(message) == "drop"
        crm.upsert_lead.assert_not_called()

    def test_missing_lead_data_is_dropped(self):
        adapter, crm, flow, _ = make_adapter()
        message = crm_message()
        message["lead_data"] = None
        assert adapter.process_message(message) == "drop"
        crm.upsert_lead.assert_not_called()

    def test_missing_lead_field_is_dropped(self):
        adapter, crm, _, _ = make_adapter()
        message = crm_message()
        message["lead_data"]["phone"] = ""
        assert adapter.process_message(message) == "drop"
        crm.upsert_lead.assert_not_called()

    def test_non_numeric_score_is_dropped(self):
        adapter, crm, _, _ = make_adapter()
        message = crm_message()
        message["lead_data"]["score"] = "alto"
        assert adapter.process_message(message) == "drop"
        crm.upsert_lead.assert_not_called()

    def test_bool_score_is_dropped(self):
        adapter, crm, _, _ = make_adapter()
        message = crm_message()
        message["lead_data"]["score"] = True
        assert adapter.process_message(message) == "drop"

    def test_non_dict_message_is_dropped(self):
        adapter, crm, _, _ = make_adapter()
        assert adapter.process_message("not a dict") == "drop"
        crm.upsert_lead.assert_not_called()

    def test_unknown_lead_is_dropped(self):
        adapter, crm, _, store = make_adapter(sessions=False)
        assert adapter.process_message(crm_message()) == "drop"
        store.get_session.assert_called_once_with("s1", "lead-1")
        crm.upsert_lead.assert_not_called()

    def test_crm_unreachable_is_retryable(self):
        adapter, crm, flow, _ = make_adapter(crm_ok=False)
        assert adapter.process_message(crm_message()) == "retry"
        adapter.status.sync.assert_not_called()

    def test_flow_error_is_retryable(self):
        adapter, crm, flow, _ = make_adapter()
        adapter.status.sync.side_effect = FlowError("flow down")
        assert adapter.process_message(crm_message()) == "retry"

    def test_stage_error_is_retryable(self):
        adapter, crm, flow, _ = make_adapter()
        adapter.status.sync.side_effect = CrmError("stage write failed")
        assert adapter.process_message(crm_message()) == "retry"

    def test_unexpected_exception_is_retryable(self):
        adapter, crm, flow, store = make_adapter()
        store.get_session.side_effect = RuntimeError("dynamo down")
        event = {"Records": [{"messageId": "r1", "body": json.dumps(crm_message())}]}
        assert adapter.handle_event(event) == {"batchItemFailures": [{"itemIdentifier": "r1"}]}

    def test_batch_retry_lists_item_failure(self):
        adapter, *_ = make_adapter(crm_ok=False)
        event = {"Records": [{"messageId": "r1", "body": json.dumps(crm_message())}]}
        assert adapter.handle_event(event) == {"batchItemFailures": [{"itemIdentifier": "r1"}]}

    def test_batch_mixed_outcomes(self):
        adapter, crm, flow, store = make_adapter()
        records = [
            {"messageId": "r1", "body": json.dumps({"message_id": "m2"})},
            {"messageId": "r2", "body": json.dumps(crm_message())},
        ]
        result = adapter.handle_event({"Records": records})
        assert result == {"batchItemFailures": []}

    def test_retry_policy_exhausted_becomes_drop(self):
        adapter, crm, flow, store = make_adapter(crm_ok=False)
        adapter.max_receives = 3
        record = {
            "messageId": "r1",
            "body": json.dumps(crm_message()),
            "attributes": {"ApproximateReceiveCount": "3"},
        }
        assert adapter.handle_event({"Records": [record]}) == {"batchItemFailures": []}

    def test_retry_within_policy_stays_retry(self):
        adapter, crm, flow, store = make_adapter(crm_ok=False)
        record = {
            "messageId": "r1",
            "body": json.dumps(crm_message()),
            "attributes": {"ApproximateReceiveCount": "2"},
        }
        assert adapter.handle_event({"Records": [record]}) == {
            "batchItemFailures": [{"itemIdentifier": "r1"}]
        }

    def test_missing_receive_count_counts_as_first_attempt(self):
        adapter, crm, flow, store = make_adapter(crm_ok=False)
        record = {"messageId": "r1", "body": json.dumps(crm_message())}
        record["attributes"] = {"ApproximateReceiveCount": "not-a-number"}
        assert adapter.handle_event({"Records": [record]}) == {
            "batchItemFailures": [{"itemIdentifier": "r1"}]
        }

    def test_malformed_body_is_dropped_explicitly(self):
        adapter, crm, flow, _ = make_adapter()
        event = {"Records": [{"messageId": "r1", "body": "not-json"}]}
        assert adapter.handle_event(event) == {"batchItemFailures": []}
        crm.upsert_lead.assert_not_called()

    def test_empty_records_returns_no_failures(self):
        adapter, *_ = make_adapter()
        assert adapter.handle_event({}) == {"batchItemFailures": []}

    def test_logs_do_not_leak_pii(self, caplog):
        adapter, crm, flow, _ = make_adapter()
        adapter.status.sync.return_value = {"stage": "qualificado"}
        with caplog.at_level("INFO"):
            adapter.process_message(crm_message())
        text = caplog.text
        assert "ana@empresa.com" not in text
        assert "Ana Ribeiro" not in text
        assert "+5511999990000" not in text