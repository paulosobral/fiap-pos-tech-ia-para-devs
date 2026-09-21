from unittest.mock import MagicMock

from service.crm_gateway import CrmError
from service.flow_gateway import FlowError
from service.status_sync import KANBAN_STAGES, StatusSync, stage_for_lead


def lead_data(**overrides):
    base = {
        "name": "Ana",
        "email": "ana@empresa.com",
        "phone": "+5511",
        "score": 85,
        "urgency": "alta",
        "intent": "compra",
    }
    base.update(overrides)
    return base


class TestStageForLead:
    def test_high_score_advances_to_qualificado(self):
        assert stage_for_lead(lead_data(score=85)) == "qualificado"

    def test_urgent_lead_advances_to_qualificado(self):
        assert stage_for_lead(lead_data(score=10, urgency="alta")) == "qualificado"

    def test_low_score_low_urgency_stays_novo(self):
        assert stage_for_lead(lead_data(score=30, urgency="baixa")) == "novo"

    def test_missing_score_stays_novo(self):
        assert stage_for_lead(lead_data(score=None, urgency="baixa")) == "novo"

    def test_invalid_score_falls_back_to_novo(self):
        assert stage_for_lead(lead_data(score="alto", urgency="baixa")) == "novo"

    def test_every_stage_is_a_valid_kanban_column(self):
        assert set(stage_for_lead(lead_data(score=s)) for s in (0, 30, 70, 100)) <= set(KANBAN_STAGES)


class TestStatusSync:
    def make_sync(self, flow=True):
        crm = MagicMock()
        flow_gw = MagicMock() if flow else None
        clock = MagicMock()
        clock.return_value.isoformat.return_value = "2026-09-20T00:00:00+00:00"
        sync = StatusSync(crm=crm, flow=flow_gw, clock=clock)
        return sync, crm, flow_gw

    def test_sync_writes_stage_and_returns_status(self):
        sync, crm, flow_gw = self.make_sync()
        status = sync.sync("lead-1", "s1", lead_data())
        crm.update_stage.assert_called_once_with("lead-1", "qualificado")
        assert status == {
            "lead_id": "lead-1",
            "session_id": "s1",
            "stage": "qualificado",
            "synced_at": "2026-09-20T00:00:00+00:00",
        }

    def test_sync_notifies_flow(self):
        sync, crm, flow_gw = self.make_sync()
        sync.sync("lead-1", "s1", lead_data())
        flow_gw.notify_status.assert_called_once_with("lead-1", "s1", "qualificado")

    def test_sync_without_flow_skips_notification(self):
        sync, crm, flow_gw = self.make_sync(flow=False)
        status = sync.sync("lead-1", "s1", lead_data())
        assert status["stage"] == "qualificado"

    def test_low_score_notifies_novo(self):
        sync, crm, flow_gw = self.make_sync()
        sync.sync("lead-2", "s2", lead_data(score=10, urgency="baixa"))
        crm.update_stage.assert_called_once_with("lead-2", "novo")
        flow_gw.notify_status.assert_called_once_with("lead-2", "s2", "novo")

    def test_crm_failure_propagates(self):
        sync, crm, _ = self.make_sync()
        crm.update_stage.side_effect = CrmError("crm down")
        try:
            sync.sync("lead-1", "s1", lead_data())
        except CrmError:
            pass
        else:
            raise AssertionError("expected CrmError")

    def test_flow_failure_propagates(self):
        sync, crm, flow_gw = self.make_sync()
        flow_gw.notify_status.side_effect = FlowError("flow down")
        try:
            sync.sync("lead-1", "s1", lead_data())
        except FlowError:
            pass
        else:
            raise AssertionError("expected FlowError")

    def test_unexpected_flow_exception_propagates(self):
        sync, crm, flow_gw = self.make_sync()
        flow_gw.notify_status.side_effect = RuntimeError("boom")
        try:
            sync.sync("lead-1", "s1", lead_data())
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError")