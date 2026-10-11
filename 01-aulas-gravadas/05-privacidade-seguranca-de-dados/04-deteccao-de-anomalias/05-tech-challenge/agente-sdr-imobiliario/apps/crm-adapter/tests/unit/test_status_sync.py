from unittest.mock import MagicMock

from service.crm_gateway import CrmError
from service.flow_gateway import FlowError
from service.status_sync import KANBAN_STAGES, StatusSync, stage_for_lead


def lead_data(**overrides):
    """lead_data do Contract 4 na forma real da u1 (urgency low/medium/high)."""
    base = {
        "name": "Ana",
        "email": "ana@empresa.com",
        "phone": "+5511",
        "score": 85,
        "urgency": "high",
        "intent": "compra",
        "budget": "R$ 800.000",
        "deadline": "1 mês",
        "area": "150 m2",
    }
    base.update(overrides)
    return base


class TestStageForLead:
    def test_high_score_advances_to_qualificado(self):
        assert stage_for_lead(lead_data(score=85)) == "qualificado"

    def test_urgent_lead_advances_to_qualificado(self):
        assert stage_for_lead(lead_data(score=10, urgency="high")) == "qualificado"

    def test_medium_urgency_stays_novo(self):
        assert stage_for_lead(lead_data(score=30, urgency="medium")) == "novo"

    def test_low_score_low_urgency_stays_novo(self):
        assert stage_for_lead(lead_data(score=30, urgency="low")) == "novo"

    def test_missing_score_stays_novo(self):
        assert stage_for_lead(lead_data(score=None, urgency="low")) == "novo"

    def test_invalid_score_falls_back_to_novo(self):
        assert stage_for_lead(lead_data(score="alto", urgency="low")) == "novo"

    def test_produced_stages_are_members_of_official_order(self):
        """Todo estágio produzido é membro da ordem monotônica oficial (u1)."""
        produced = set()
        for score in (0, 30, 70, 100):
            for urgency in ("low", "medium", "high"):
                produced.add(stage_for_lead(lead_data(score=score, urgency=urgency)))
        assert produced <= set(KANBAN_STAGES)


class TestStatusSync:
    def make_sync(self, flow=True, current_stage=None):
        crm = MagicMock()
        if current_stage is None:
            crm.get_lead.return_value = None
        else:
            crm.get_lead.return_value = {"lead_id": "lead-1", "stage": current_stage}
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
        sync.sync("lead-2", "s2", lead_data(score=10, urgency="low"))
        crm.update_stage.assert_called_once_with("lead-2", "novo")
        flow_gw.notify_status.assert_called_once_with("lead-2", "s2", "novo")

    def test_fresh_lead_empty_stage_applies_target(self):
        """Lead novo no CRM (stage ""): estágio alvo é aplicado."""
        sync, crm, _ = self.make_sync(current_stage="")
        sync.sync("lead-1", "s1", lead_data())
        crm.update_stage.assert_called_once_with("lead-1", "qualificado")

    def test_forward_progression_updates_stage(self):
        sync, crm, _ = self.make_sync(current_stage="novo")
        sync.sync("lead-1", "s1", lead_data())
        crm.update_stage.assert_called_once_with("lead-1", "qualificado")

    def test_equal_stage_rewrites_same_stage(self):
        sync, crm, _ = self.make_sync(current_stage="qualificado")
        status = sync.sync("lead-1", "s1", lead_data())
        crm.update_stage.assert_called_once_with("lead-1", "qualificado")
        assert status["stage"] == "qualificado"

    def test_regression_is_blocked_keeps_advanced_stage(self):
        """Redelivery/re-enfileiramento não regride estágio avançado pelo corretor."""
        sync, crm, flow_gw = self.make_sync(current_stage="contato-feito")
        status = sync.sync("lead-1", "s1", lead_data())
        crm.update_stage.assert_not_called()
        assert status["stage"] == "contato-feito"
        flow_gw.notify_status.assert_called_once_with("lead-1", "s1", "contato-feito")

    def test_regression_from_externally_moved_stage(self):
        sync, crm, flow_gw = self.make_sync(current_stage="visita-agendada")
        status = sync.sync("lead-1", "s1", lead_data(score=10, urgency="low"))
        crm.update_stage.assert_not_called()
        assert status["stage"] == "visita-agendada"
        flow_gw.notify_status.assert_called_once_with("lead-1", "s1", "visita-agendada")

    def test_unknown_stage_in_crm_is_overwritten(self):
        sync, crm, _ = self.make_sync(current_stage="estágio-fora-da-esteira")
        status = sync.sync("lead-1", "s1", lead_data())
        crm.update_stage.assert_called_once_with("lead-1", "qualificado")
        assert status["stage"] == "qualificado"

    def test_get_lead_failure_propagates(self):
        sync, crm, _ = self.make_sync()
        crm.get_lead.side_effect = CrmError("crm down")
        try:
            sync.sync("lead-1", "s1", lead_data())
        except CrmError:
            pass
        else:
            raise AssertionError("expected CrmError")

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
