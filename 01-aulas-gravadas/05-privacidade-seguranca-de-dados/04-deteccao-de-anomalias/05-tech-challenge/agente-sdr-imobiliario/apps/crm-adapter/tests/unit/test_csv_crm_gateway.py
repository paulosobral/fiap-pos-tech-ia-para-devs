from datetime import datetime, timezone

import pytest

from infra.csv_store import CsvStore, CsvStoreError
from service.crm_gateway import CrmError, CsvCrmGateway, LeadNotFoundError


def lead(**overrides):
    base = {
        "lead_id": "lead-1",
        "name": "Ana Ribeiro",
        "email": "ana@empresa.com",
        "phone": "+5511999990000",
        "score": 85,
        "urgency": "alta",
        "intent": "compra",
        "session_id": "s1",
    }
    base.update(overrides)
    return base


def fixed_clock():
    return datetime(2026, 9, 20, tzinfo=timezone.utc)


@pytest.fixture()
def store(tmp_path):
    return CsvStore(str(tmp_path / "crm-leads.csv"))


class TestCsvCrmGateway:
    def test_upsert_creates_lead_with_crm_id(self, store):
        gateway = CsvCrmGateway(store, clock=fixed_clock)
        record = gateway.upsert_lead(lead())
        assert record["crm_id"]
        assert record["stage"] == ""
        assert record["created_at"] == "2026-09-20T00:00:00+00:00"
        reloaded = gateway.get_lead("lead-1")
        assert reloaded["crm_id"] == record["crm_id"]
        assert reloaded["email"] == "ana@empresa.com"

    def test_upsert_existing_updates_same_row(self, store):
        gateway = CsvCrmGateway(store, clock=fixed_clock)
        first = gateway.upsert_lead(lead())
        second = gateway.upsert_lead(lead(score=95))
        assert first["crm_id"] == second["crm_id"]
        assert second["score"] == "95"
        rows = store.load()
        assert len(rows) == 1
        assert rows[0]["created_at"] == "2026-09-20T00:00:00+00:00"

    def test_update_stage_persists(self, store):
        gateway = CsvCrmGateway(store, clock=fixed_clock)
        gateway.upsert_lead(lead())
        gateway.update_stage("lead-1", "qualificado")
        record = gateway.get_lead("lead-1")
        assert record["stage"] == "qualificado"
        assert record["updated_at"] == "2026-09-20T00:00:00+00:00"

    def test_update_stage_unknown_lead_raises(self, store):
        gateway = CsvCrmGateway(store, clock=fixed_clock)
        with pytest.raises(LeadNotFoundError):
            gateway.update_stage("ghost", "novo")

    def test_get_lead_missing_returns_none(self, store):
        gateway = CsvCrmGateway(store)
        assert gateway.get_lead("ghost") is None

    def test_unreadable_store_raises_crm_error(self, tmp_path):
        path = tmp_path / "crm-leads.csv"
        path.write_text("coluna-errada\nvalor\n", encoding="utf-8")
        gateway = CsvCrmGateway(CsvStore(str(path)))
        with pytest.raises(CrmError):
            gateway.upsert_lead(lead())

    def test_unwritable_store_raises_crm_error(self, store, monkeypatch):
        gateway = CsvCrmGateway(store)

        def broken_save(rows):
            raise CsvStoreError("disk full")

        monkeypatch.setattr(store, "save", broken_save)
        with pytest.raises(CrmError):
            gateway.upsert_lead(lead())

    def test_get_lead_unreadable_store_raises_crm_error(self, tmp_path):
        path = tmp_path / "crm-leads.csv"
        path.write_text("coluna-errada\nvalor\n", encoding="utf-8")
        gateway = CsvCrmGateway(CsvStore(str(path)))
        with pytest.raises(CrmError):
            gateway.get_lead("lead-1")

    def test_update_stage_unwritable_store_raises_crm_error(self, store, monkeypatch):
        gateway = CsvCrmGateway(store, clock=fixed_clock)
        gateway.upsert_lead(lead())

        def broken_save(rows):
            raise CsvStoreError("disk full")

        monkeypatch.setattr(store, "save", broken_save)
        with pytest.raises(CrmError):
            gateway.update_stage("lead-1", "qualificado")

    def test_update_stage_unreadable_store_raises_crm_error(self, tmp_path):
        path = tmp_path / "crm-leads.csv"
        path.write_text("coluna-errada\nvalor\n", encoding="utf-8")
        gateway = CsvCrmGateway(CsvStore(str(path)))
        with pytest.raises(CrmError):
            gateway.update_stage("lead-1", "qualificado")

    def test_roundtrip_multiple_leads(self, store):
        gateway = CsvCrmGateway(store, clock=fixed_clock)
        gateway.upsert_lead(lead())
        gateway.upsert_lead(lead(lead_id="lead-2", email="bob@empresa.com", score=10))
        rows = store.load()
        assert len(rows) == 2
        assert {row["lead_id"] for row in rows} == {"lead-1", "lead-2"}