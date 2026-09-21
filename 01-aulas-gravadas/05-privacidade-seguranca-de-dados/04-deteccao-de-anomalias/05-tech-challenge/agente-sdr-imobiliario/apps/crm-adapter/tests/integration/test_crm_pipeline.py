import json
import logging
import sys
from unittest.mock import MagicMock

import pytest

from handler import handler
from service.crm_adapter import CrmAdapter
from service.crm_gateway import CrmError
from service.flow_gateway import FlowError
from service.status_sync import StatusSync


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


def sqs_record(message_id, body):
    return {"messageId": message_id, "body": body}


def make_adapter(stage="qualificado"):
    crm = MagicMock()
    crm.upsert_lead.return_value = {"crm_id": "crm-1", "lead_id": "lead-1"}
    flow = MagicMock()
    status = MagicMock()
    status.sync.return_value = {"lead_id": "lead-1", "session_id": "s1", "stage": stage}
    store = MagicMock()
    store.get_session.return_value = {"session_id": "s1"}
    adapter = CrmAdapter(crm=crm, status=status, flow=flow, sessions=store)
    return adapter, crm, flow, store


class TestCrmPipeline:
    def test_full_pipeline_no_failures_and_csv_written(self, tmp_path):
        from infra.csv_store import CsvStore
        from service.crm_gateway import CsvCrmGateway

        path = tmp_path / "crm-leads.csv"
        gateway = CsvCrmGateway(CsvStore(str(path)))
        flow = MagicMock()
        sessions = MagicMock()
        sessions.get_session.return_value = {"session_id": "s1"}
        adapter = CrmAdapter(crm=gateway, status=StatusSync(gateway, flow), flow=flow, sessions=sessions)
        result = adapter.handle_event({"Records": [sqs_record("r1", json.dumps(crm_message()))]})
        assert result == {"batchItemFailures": []}
        rows = path.open(encoding="utf-8").read()
        assert "lead-1" in rows
        assert "qualificado" in rows
        flow.notify_status.assert_called_once_with("lead-1", "s1", "qualificado")

    def test_low_score_lead_lands_in_novo(self, tmp_path):
        from infra.csv_store import CsvStore
        from service.crm_gateway import CsvCrmGateway

        gateway = CsvCrmGateway(CsvStore(str(tmp_path / "crm-leads.csv")))
        flow = MagicMock()
        sessions = MagicMock()
        sessions.get_session.return_value = {"session_id": "s1"}
        adapter = CrmAdapter(
            crm=gateway,
            status=StatusSync(gateway, flow),
            flow=flow,
            sessions=sessions,
        )
        message = crm_message(lead_data={**crm_message()["lead_data"], "score": 30, "urgency": "baixa"})
        adapter.handle_event({"Records": [sqs_record("r1", json.dumps(message))]})
        flow.notify_status.assert_called_once_with("lead-1", "s1", "novo")

    def test_mixed_batch_drops_invalid_without_failure(self):
        adapter, *_ = make_adapter()
        records = [
            sqs_record("r1", json.dumps(crm_message())),
            sqs_record("r2", json.dumps({"message_id": "m2"})),
        ]
        assert adapter.handle_event({"Records": records}) == {"batchItemFailures": []}

    def test_retry_record_listed_in_failures(self):
        adapter, crm, flow, _ = make_adapter()
        crm.upsert_lead.side_effect = CrmError("crm down")
        result = adapter.handle_event({"Records": [sqs_record("r1", json.dumps(crm_message()))]})
        assert result == {"batchItemFailures": [{"itemIdentifier": "r1"}]}

    def test_malformed_body_dropped_explicitly(self):
        adapter, crm, flow, _ = make_adapter()
        assert adapter.handle_event({"Records": [sqs_record("r1", "not-json")]}) == {
            "batchItemFailures": []
        }
        crm.upsert_lead.assert_not_called()

    def test_unknown_lead_dropped_in_batch(self):
        adapter, crm, flow, _ = make_adapter()
        adapter.sessions.get_session.return_value = None
        result = adapter.handle_event({"Records": [sqs_record("r1", json.dumps(crm_message()))]})
        assert result == {"batchItemFailures": []}
        crm.upsert_lead.assert_not_called()

    def test_unexpected_exception_becomes_retry(self):
        adapter, *_ = make_adapter()
        adapter.sessions.get_session.side_effect = RuntimeError("dynamo down")
        result = adapter.handle_event({"Records": [sqs_record("r1", json.dumps(crm_message()))]})
        assert result == {"batchItemFailures": [{"itemIdentifier": "r1"}]}

    def test_retry_policy_exhausted_drops_after_policy(self):
        adapter, crm, flow, _ = make_adapter()
        crm.upsert_lead.side_effect = CrmError("crm down")
        record = sqs_record("r1", json.dumps(crm_message()))
        record["attributes"] = {"ApproximateReceiveCount": "3"}
        assert adapter.handle_event({"Records": [record]}) == {"batchItemFailures": []}

    def test_empty_records_returns_no_failures(self):
        adapter, *_ = make_adapter()
        assert adapter.handle_event({}) == {"batchItemFailures": []}

    def test_pipeline_logs_no_pii(self, caplog):
        adapter, *_ = make_adapter()
        with caplog.at_level(logging.INFO):
            adapter.handle_event({"Records": [sqs_record("r1", json.dumps(crm_message()))]})
        assert "ana@empresa.com" not in caplog.text
        assert "Ana Ribeiro" not in caplog.text

    def test_repeated_upsert_updates_same_lead(self, tmp_path):
        from infra.csv_store import CsvStore
        from service.crm_gateway import CsvCrmGateway

        gateway = CsvCrmGateway(CsvStore(str(tmp_path / "crm-leads.csv")))
        flow = MagicMock()
        sessions = MagicMock()
        sessions.get_session.return_value = {"session_id": "s1"}
        adapter = CrmAdapter(crm=gateway, status=StatusSync(gateway, flow), flow=flow, sessions=sessions)
        for score in (85, 95):
            message = crm_message(lead_data={**crm_message()["lead_data"], "score": score})
            adapter.handle_event({"Records": [sqs_record("r1", json.dumps(message))]})
        lines = (tmp_path / "crm-leads.csv").open(encoding="utf-8").read().strip().split("\n")
        assert len(lines) == 2  # header + 1 lead


class TestHandlerWiring:
    def test_handler_builds_adapter_and_processes_event(self, monkeypatch, tmp_path):
        boto3 = MagicMock()
        requests = MagicMock()
        monkeypatch.setitem(sys.modules, "boto3", boto3)
        monkeypatch.setitem(sys.modules, "requests", requests)
        monkeypatch.setenv("FLOW_BASE_URL", "http://flow.local")
        monkeypatch.setenv("CRM_CSV_PATH", str(tmp_path / "crm-leads.csv"))
        event = {"Records": [sqs_record("r1", json.dumps({"message_id": "m1"}))]}
        result = handler(event)
        assert result == {"batchItemFailures": []}
        boto3.client.assert_called_once_with("dynamodb")

    def test_handler_missing_env_raises(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "boto3", MagicMock())
        monkeypatch.setitem(sys.modules, "requests", MagicMock())
        monkeypatch.delenv("FLOW_BASE_URL", raising=False)
        monkeypatch.setenv("CRM_CSV_PATH", str(tmp_path / "crm-leads.csv"))
        with pytest.raises(RuntimeError):
            handler({"Records": []})