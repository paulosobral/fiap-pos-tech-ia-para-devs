import pytest

from infra.csv_store import COLUMNS, CsvStore, CsvStoreError


def row(**overrides):
    base = {column: "" for column in COLUMNS}
    base.update(
        {
            "crm_id": "crm-1",
            "lead_id": "lead-1",
            "name": "Ana",
            "stage": "novo",
        }
    )
    base.update(overrides)
    return base


class TestCsvStore:
    def test_save_and_load_roundtrip(self, tmp_path):
        store = CsvStore(str(tmp_path / "leads.csv"))
        store.save([row(email="ana@empresa.com", score="85")])
        rows = store.load()
        assert rows == [row(email="ana@empresa.com", score="85")]

    def test_load_missing_file_returns_empty(self, tmp_path):
        assert CsvStore(str(tmp_path / "missing.csv")).load() == []

    def test_save_creates_parent_directory(self, tmp_path):
        store = CsvStore(str(tmp_path / "crm" / "leads.csv"))
        store.save([row()])
        assert store.load() == [row()]

    def test_load_corrupt_header_raises(self, tmp_path):
        path = tmp_path / "crm-leads.csv"
        path.write_text("lead_id,extra\n1,2\n", encoding="utf-8")
        with pytest.raises(CsvStoreError):
            CsvStore(str(path)).load()

    def test_load_row_missing_required_column_raises(self, tmp_path):
        path = tmp_path / "crm-leads.csv"
        path.write_text(",".join(COLUMNS) + "\n" + "crm-1,,Ana\n", encoding="utf-8")
        with pytest.raises(CsvStoreError):
            CsvStore(str(path)).load()

    def test_save_overwrites_previous_content(self, tmp_path):
        store = CsvStore(str(tmp_path / "crm-leads.csv"))
        store.save([row()])
        store.save([row(lead_id="lead-2")])
        rows = store.load()
        assert len(rows) == 1
        assert rows[0]["lead_id"] == "lead-2"

    def test_save_handles_special_characters(self, tmp_path):
        store = CsvStore(str(tmp_path / "crm-leads.csv"))
        store.save([row(name="Ana, Ribeiro", email="a@b.com; c")])
        assert store.load()[0]["name"] == "Ana, Ribeiro"

    def test_path_property(self, tmp_path):
        store = CsvStore(str(tmp_path / "crm-leads.csv"))
        assert store.path.endswith("crm-leads.csv")

    def test_save_io_error_raises(self, tmp_path, monkeypatch):
        import os

        store = CsvStore(str(tmp_path / "crm-leads.csv"))
        store.save([row()])

        def broken_replace(src, dst):
            raise OSError("no space left on device")

        monkeypatch.setattr(os, "replace", broken_replace)
        with pytest.raises(CsvStoreError):
            store.save([row()])

    def test_load_io_error_raises(self, tmp_path, monkeypatch):
        import builtins

        path = tmp_path / "crm-leads.csv"
        path.write_text(",".join(COLUMNS) + "\n", encoding="utf-8")
        real_open = builtins.open

        def broken_open(file, *args, **kwargs):
            if str(file) == str(path):
                raise OSError("permission denied")
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", broken_open)
        with pytest.raises(CsvStoreError):
            CsvStore(str(path)).load()