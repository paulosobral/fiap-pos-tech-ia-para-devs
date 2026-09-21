import json
import logging
from unittest.mock import MagicMock

from service.crm_adapter import CrmAdapter
from service.crm_gateway import CrmError
from service.flow_gateway import FlowError


def crm_message(**overrides):
    """Payload do Contract 4 na forma atualizada (superfície real da u1)."""
    base = {
        "message_id": "m1",
        "lead_id": "lead-1",
        "lead_data": {
            "name": "Ana Ribeiro",
            "email": "ana@empresa.com",
            "phone": "+5511999990000",
            "score": 85,
            "urgency": "high",
            "intent": "compra",
            "budget": "R$ 800.000",
            "deadline": "1 mês",
            "area": "150 m2",
        },
        "session_id": "s1",
        "timestamp": "2026-09-20T00:00:00+00:00",
    }
    base.update(overrides)
    return base


def real_producer_message(contact=None, lead=None, info=None):
    """Snapshot REAL do `_enqueue_crm` da u1 (apps/conversation-router/handler.py).

    `name` vem de contact["NOME"] com fallback seguro (nunca a flag yes/no de
    decision_maker); `email`/`phone` vêm do registro de PII e podem faltar;
    `urgency` ∈ low/medium/high; `budget`/`deadline`/`area` vêm de
    context["lead_info"] e podem faltar; `message_id` == `session_id`.
    """
    contact = dict(contact) if contact is not None else {}
    lead = {"decision_maker": "no", **(lead or {})}
    info = dict(info) if info is not None else {}
    session_id = "s1"
    names = contact.get("NOME") or []
    if names:
        name = names[0]
    elif lead.get("decision_maker") == "yes":
        name = "Decisor (nome não informado)"
    else:
        name = "Lead (nome não informado)"
    return {
        "message_id": session_id,
        "lead_id": lead.get("lead_id", "lead-1"),
        "lead_data": {
            "name": name,
            "email": (contact.get("EMAIL") or [None])[0],
            "phone": (contact.get("TELEFONE") or [None])[0],
            "score": lead.get("score"),
            "urgency": lead.get("urgency", "high"),
            "intent": lead.get("intent"),
            "budget": info.get("budget"),
            "deadline": info.get("deadline"),
            "area": info.get("area"),
        },
        "session_id": session_id,
        "timestamp": "2026-09-20T00:00:00+00:00",
    }


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


class TestContract4RealProducer:
    def test_real_payload_without_contact_registers_lead(self, caplog):
        """Sem registro de PII (sem NOME/EMAIL/TELEFONE): registra, não descarta."""
        adapter, crm, _, _ = make_adapter()
        adapter.status.sync.return_value = {"stage": "qualificado"}
        message = real_producer_message(contact={}, lead={"score": 85, "urgency": "high"})
        with caplog.at_level(logging.INFO):
            assert adapter.process_message(message) == "ok"
        payload = crm.upsert_lead.call_args[0][0]
        assert payload["name"] == "Lead (nome não informado)"
        assert payload["email"] is None
        assert payload["phone"] is None
        record = next(r for r in caplog.records if "lead_synced" in r.message)
        assert json.loads(record.message)["contact_info_partial"] is True

    def test_real_payload_decision_maker_without_name(self):
        """Decisor sem NOME no registro de PII: fallback seguro, nunca a flag yes/no."""
        adapter, crm, _, _ = make_adapter()
        adapter.status.sync.return_value = {"stage": "qualificado"}
        message = real_producer_message(contact={}, lead={"decision_maker": "yes"})
        assert adapter.process_message(message) == "ok"
        assert crm.upsert_lead.call_args[0][0]["name"] == "Decisor (nome não informado)"

    def test_real_payload_name_from_pii_record(self, caplog):
        adapter, crm, _, _ = make_adapter()
        adapter.status.sync.return_value = {"stage": "qualificado"}
        contact = {"NOME": ["Ana Ribeiro"], "EMAIL": ["ana@empresa.com"], "TELEFONE": ["+5511999990000"]}
        message = real_producer_message(contact=contact, lead={"score": 85, "urgency": "high"})
        with caplog.at_level(logging.INFO):
            assert adapter.process_message(message) == "ok"
        payload = crm.upsert_lead.call_args[0][0]
        assert payload["name"] == "Ana Ribeiro"
        assert payload["email"] == "ana@empresa.com"
        record = next(r for r in caplog.records if "lead_synced" in r.message)
        assert json.loads(record.message)["contact_info_partial"] is False


class TestCrmAdapter:
    def test_happy_path_syncs_lead_and_stage(self):
        adapter, crm, flow, _ = make_adapter()
        adapter.status.sync.return_value = {"lead_id": "lead-1", "session_id": "s1", "stage": "qualificado", "synced_at": "t"}
        outcome = adapter.process_message(crm_message())
        assert outcome == "ok"
        crm.upsert_lead.assert_called_once()
        payload = crm.upsert_lead.call_args[0][0]
        assert payload["lead_id"] == "lead-1"
        assert payload["email"] == "ana@empresa.com"
        assert payload["budget"] == "R$ 800.000"
        assert payload["deadline"] == "1 mês"
        assert payload["area"] == "150 m2"
        adapter.status.sync.assert_called_once_with("lead-1", "s1", crm_message()["lead_data"])
        flow.notify_status.assert_not_called()

    def test_missing_phone_is_registered_not_dropped(self):
        """email/phone ausentes do Contract 4 real → campos vazios, mensagem não descartada."""
        adapter, crm, _, _ = make_adapter()
        adapter.status.sync.return_value = {"stage": "qualificado"}
        message = crm_message()
        message["lead_data"]["phone"] = None
        assert adapter.process_message(message) == "ok"
        assert crm.upsert_lead.call_args[0][0]["phone"] is None

    def test_missing_email_is_registered_not_dropped(self):
        adapter, crm, _, _ = make_adapter()
        adapter.status.sync.return_value = {"stage": "qualificado"}
        message = crm_message()
        message["lead_data"]["email"] = None
        assert adapter.process_message(message) == "ok"
        assert crm.upsert_lead.call_args[0][0]["email"] is None

    def test_missing_score_is_registered_not_dropped(self):
        adapter, crm, _, _ = make_adapter()
        adapter.status.sync.return_value = {"stage": "novo"}
        message = crm_message()
        message["lead_data"]["score"] = None
        assert adapter.process_message(message) == "ok"
        assert crm.upsert_lead.call_args[0][0]["score"] is None

    def test_missing_name_is_dropped(self):
        adapter, crm, _, _ = make_adapter()
        message = crm_message()
        message["lead_data"]["name"] = ""
        assert adapter.process_message(message) == "drop"
        crm.upsert_lead.assert_not_called()

    def test_missing_urgency_is_dropped(self):
        adapter, crm, _, _ = make_adapter()
        message = crm_message()
        message["lead_data"]["urgency"] = None
        assert adapter.process_message(message) == "drop"
        crm.upsert_lead.assert_not_called()

    def test_non_string_email_is_dropped(self):
        adapter, crm, _, _ = make_adapter()
        message = crm_message()
        message["lead_data"]["email"] = 123
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

    def test_non_dict_lead_data_is_dropped(self):
        adapter, crm, _, _ = make_adapter()
        message = crm_message()
        message["lead_data"] = "not a dict"
        assert adapter.process_message(message) == "drop"
        crm.upsert_lead.assert_not_called()

    def test_missing_lead_data_is_dropped(self):
        adapter, crm, _, _ = make_adapter()
        message = crm_message()
        message["lead_data"] = None
        assert adapter.process_message(message) == "drop"
        crm.upsert_lead.assert_not_called()

    def test_non_dict_message_is_dropped(self):
        adapter, crm, _, _ = make_adapter()
        assert adapter.process_message("not a dict") == "drop"
        crm.upsert_lead.assert_not_called()

    def test_unknown_lead_is_dropped(self):
        adapter, crm, _, store = make_adapter(sessions=False)
        assert adapter.process_message(crm_message()) == "drop"
        store.get_session.assert_called_once_with("lead-1", "s1")
        crm.upsert_lead.assert_not_called()

    def test_crm_unreachable_is_retryable(self):
        adapter, crm, _, _ = make_adapter(crm_ok=False)
        assert adapter.process_message(crm_message()) == "retry"
        adapter.status.sync.assert_not_called()

    def test_flow_error_is_retryable(self):
        adapter, crm, _, _ = make_adapter()
        adapter.status.sync.side_effect = FlowError("flow down")
        assert adapter.process_message(crm_message()) == "retry"

    def test_stage_error_is_retryable(self):
        adapter, crm, _, _ = make_adapter()
        adapter.status.sync.side_effect = CrmError("stage write failed")
        assert adapter.process_message(crm_message()) == "retry"

    def test_unexpected_exception_is_retryable(self):
        adapter, _, _, store = make_adapter()
        store.get_session.side_effect = RuntimeError("dynamo down")
        event = {"Records": [{"messageId": "r1", "body": json.dumps(crm_message())}]}
        assert adapter.handle_event(event) == {"batchItemFailures": [{"itemIdentifier": "r1"}]}

    def test_batch_retry_lists_item_failure(self):
        adapter, *_ = make_adapter(crm_ok=False)
        event = {"Records": [{"messageId": "r1", "body": json.dumps(crm_message())}]}
        assert adapter.handle_event(event) == {"batchItemFailures": [{"itemIdentifier": "r1"}]}

    def test_batch_mixed_outcomes(self):
        adapter, _, _, _ = make_adapter()
        records = [
            {"messageId": "r1", "body": json.dumps({"message_id": "m2"})},
            {"messageId": "r2", "body": json.dumps(crm_message())},
        ]
        result = adapter.handle_event({"Records": records})
        assert result == {"batchItemFailures": []}

    def test_retry_policy_exhausted_becomes_drop(self):
        adapter, *_ = make_adapter(crm_ok=False)
        adapter.max_receives = 3
        record = {
            "messageId": "r1",
            "body": json.dumps(crm_message()),
            "attributes": {"ApproximateReceiveCount": "3"},
        }
        assert adapter.handle_event({"Records": [record]}) == {"batchItemFailures": []}

    def test_retry_within_policy_stays_retry(self):
        adapter, *_ = make_adapter(crm_ok=False)
        record = {
            "messageId": "r1",
            "body": json.dumps(crm_message()),
            "attributes": {"ApproximateReceiveCount": "2"},
        }
        assert adapter.handle_event({"Records": [record]}) == {
            "batchItemFailures": [{"itemIdentifier": "r1"}]
        }

    def test_missing_receive_count_counts_as_first_attempt(self):
        adapter, *_ = make_adapter(crm_ok=False)
        record = {"messageId": "r1", "body": json.dumps(crm_message())}
        record["attributes"] = {"ApproximateReceiveCount": "not-a-number"}
        assert adapter.handle_event({"Records": [record]}) == {
            "batchItemFailures": [{"itemIdentifier": "r1"}]
        }

    def test_malformed_body_is_dropped_explicitly(self):
        adapter, crm, _, _ = make_adapter()
        event = {"Records": [{"messageId": "r1", "body": "not-json"}]}
        assert adapter.handle_event(event) == {"batchItemFailures": []}
        crm.upsert_lead.assert_not_called()

    def test_empty_records_returns_no_failures(self):
        adapter, *_ = make_adapter()
        assert adapter.handle_event({}) == {"batchItemFailures": []}


class TestLogsPiiSafe:
    def test_logs_do_not_leak_pii(self, caplog):
        adapter, _, _, _ = make_adapter()
        adapter.status.sync.return_value = {"stage": "qualificado"}
        with caplog.at_level(logging.INFO):
            adapter.process_message(crm_message())
        text = caplog.text
        assert "ana@empresa.com" not in text
        assert "Ana Ribeiro" not in text
        assert "+5511999990000" not in text

    def test_invalid_body_log_masks_pii(self, caplog):
        """Body SQS cru malformado com PII → preview mascarado (padrão oficial da u1)."""
        adapter, *_ = make_adapter()
        body = (
            '{"message_id": "m1", "lead_data": {"name": "Ana Ribeiro",'
            ' "email": "ana@empresa.com", "phone": "+5511999990000"'
        )
        with caplog.at_level(logging.INFO):
            adapter.handle_event({"Records": [{"messageId": "r1", "body": body}]})
        text = caplog.text
        assert "ana@empresa.com" not in text
        assert "Ana Ribeiro" not in text
        assert "+5511999990000" not in text
        assert "[EMAIL]" in text
        assert "[TELEFONE]" in text
        assert "[NOME]" in text
        assert '"event": "invalid_body"' in text
        assert '"body_length"' in text

    def test_unexpected_failure_logs_structured_json(self, caplog):
        adapter, *_ = make_adapter()
        adapter.sessions.get_session.side_effect = RuntimeError("dynamo down")
        with caplog.at_level(logging.INFO):
            adapter.handle_event({"Records": [{"messageId": "r1", "body": json.dumps(crm_message())}]})
        record = next(r for r in caplog.records if "unexpected_failure" in r.message)
        payload = json.loads(record.message)
        assert payload["event"] == "unexpected_failure"
        assert payload["error"] == "dynamo down"

    def test_crm_error_logs_structured_json(self, caplog):
        adapter, *_ = make_adapter(crm_ok=False)
        with caplog.at_level(logging.INFO):
            adapter.process_message(crm_message())
        record = next(r for r in caplog.records if "crm_unreachable" in r.message)
        payload = json.loads(record.message)
        assert payload["event"] == "crm_unreachable"
        assert payload["error"] == "crm down"
