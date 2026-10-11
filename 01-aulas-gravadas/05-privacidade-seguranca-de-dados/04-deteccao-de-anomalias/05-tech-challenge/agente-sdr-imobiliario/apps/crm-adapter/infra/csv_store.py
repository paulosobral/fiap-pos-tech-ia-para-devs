from __future__ import annotations

import csv
import os
import tempfile
from typing import Any


class CsvStoreError(Exception):
    pass


COLUMNS = (
    "crm_id",
    "lead_id",
    "name",
    "email",
    "phone",
    "score",
    "urgency",
    "intent",
    "budget",
    "deadline",
    "area",
    "stage",
    "session_id",
    "created_at",
    "updated_at",
)


class CsvStore:
    """CSV file workspace que simula a base do CRM na POC."""

    def __init__(self, path: str) -> None:
        self._path = path

    @property
    def path(self) -> str:
        return self._path

    def load(self) -> list[dict[str, str]]:
        if not os.path.exists(self._path):
            return []
        try:
            with open(self._path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is not None and tuple(reader.fieldnames) != COLUMNS:
                    raise CsvStoreError(f"csv header mismatch in {self._path}")
                rows = []
                for row in reader:
                    rows.append(self._clean(row))
        except (OSError, csv.Error) as exc:
            raise CsvStoreError(f"csv store unreadable: {exc}") from exc
        return rows

    def save(self, rows: list[dict[str, str]]) -> None:
        directory = os.path.dirname(self._path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory or ".", suffix=".csv")
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(COLUMNS))
                writer.writeheader()
                for row in rows:
                    writer.writerow({column: str(row.get(column) or "") for column in COLUMNS})
            os.replace(tmp_path, self._path)
        except OSError as exc:
            raise CsvStoreError(f"csv store unwritable: {exc}") from exc

    @staticmethod
    def _clean(row: dict[str, Any]) -> dict[str, str]:
        cleaned = {column: str(row.get(column) or "") for column in COLUMNS}
        for column in ("crm_id", "lead_id"):
            if not cleaned[column]:
                raise CsvStoreError(f"csv row missing required column {column}")
        return cleaned