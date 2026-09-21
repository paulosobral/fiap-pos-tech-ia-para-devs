from unittest.mock import MagicMock

import pytest

from service.flow.lead_qualifier import LeadQualifier
from service.flow.sales_flow import SalesFlow
from service.restriction import DynamoRestrictionCheck


def restricted_item(lead_id="l1", status="open"):
    return {
        "PK": {"S": "anomaly-1"},
        "lead_id": {"S": lead_id},
        "scheduling_restricted": {"BOOL": True},
        "status": {"S": status},
        "action_taken": {"S": "schedule_restricted"},
    }


class FakeAlertClient:
    def __init__(self, items=None, error=None):
        self._items = items or []
        self._error = error
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return {"Items": self._items}


class TestDynamoRestrictionCheck:
    def test_open_restriction_for_lead(self):
        client = FakeAlertClient([restricted_item()])
        check = DynamoRestrictionCheck(client, "t-alerts")
        assert check("l1") is True

    def test_open_restriction_for_other_lead(self):
        client = FakeAlertClient([restricted_item(lead_id="l2")])
        check = DynamoRestrictionCheck(client, "t-alerts")
        assert check("l1") is False

    def test_closed_restriction_is_not_restricting(self):
        client = FakeAlertClient([restricted_item(status="resolved")])
        check = DynamoRestrictionCheck(client, "t-alerts")
        assert check("l1") is False

    def test_query_error_fail_open(self):
        client = FakeAlertClient(error=RuntimeError("table down"))
        check = DynamoRestrictionCheck(client, "t-alerts")
        assert check("l1") is False

    def test_empty_lead_id_skips_query(self):
        client = FakeAlertClient([restricted_item()])
        check = DynamoRestrictionCheck(client, "t-alerts")
        assert check("") is False
        assert client.calls == []

    def test_queries_alert_table_via_lead_index_gsi(self):
        client = FakeAlertClient([])
        check = DynamoRestrictionCheck(client, "t-alerts")
        check("l1")
        kwargs = client.calls[0]
        assert kwargs["TableName"] == "t-alerts"
        assert kwargs["IndexName"] == "lead-index"

    def test_table_defaults_to_alerts_table_env(self, monkeypatch):
        monkeypatch.setenv("ALERTS_TABLE", "t-env")
        client = FakeAlertClient([])
        check = DynamoRestrictionCheck(client)
        check("l1")
        assert client.calls[0]["TableName"] == "t-env"

    def test_table_default_name_matches_u5(self, monkeypatch):
        monkeypatch.delenv("ALERTS_TABLE", raising=False)
        client = FakeAlertClient([])
        check = DynamoRestrictionCheck(client)
        check("l1")
        assert client.calls[0]["TableName"] == "sdr-alerts"

    def test_flow_uses_checker_result(self):
        scheduler = MagicMock(return_value={"confirmed": True})
        check = DynamoRestrictionCheck(FakeAlertClient([restricted_item()]), "t-alerts")
        flow = SalesFlow(lead_qualifier=LeadQualifier(), scheduler=scheduler, restriction_check=check)
        state = flow.invoke({"current_state": "scheduling", "message": "amanhã", "lead_id": "l1"})
        scheduler.assert_not_called()
        assert state["scheduling_restricted"] is True
