from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from infra.csv_store import CsvStore, CsvStoreError

logger = logging.getLogger(__name__)


class CrmError(Exception):
    """Falha transitória/permanente ao falar com o CRM (→ retry/DLQ)."""


class LeadNotFoundError(CrmError):
    pass


class CrmGateway(Protocol):
    def upsert_lead(self, lead: dict[str, Any]) -> dict[str, Any]: ...

    def get_lead(self, lead_id: str) -> dict[str, Any] | None: ...

    def update_stage(self, lead_id: str, stage: str) -> None: ...


class CsvCrmGateway:
    """CRM simulado default da POC: workspace CSV (FR11.2)."""

    def __init__(self, store: CsvStore, clock: Callable[[], datetime] | None = None) -> None:
        self._store = store
        self._clock = clock or datetime.now

    def upsert_lead(self, lead: dict[str, Any]) -> dict[str, Any]:
        try:
            rows = self._store.load()
        except CsvStoreError as exc:
            logger.error("crm store unreadable: %s", exc)
            raise CrmError("crm store unreadable") from exc
        lead_id = str(lead.get("lead_id") or "")
        existing = next((row for row in rows if row["lead_id"] == lead_id), None)
        timestamp = self._clock().isoformat()
        if existing is None:
            record = {
                "crm_id": str(uuid.uuid4()),
                "lead_id": lead_id,
                "name": str(lead.get("name") or ""),
                "email": str(lead.get("email") or ""),
                "phone": str(lead.get("phone") or ""),
                "score": str(lead.get("score") or ""),
                "urgency": str(lead.get("urgency") or ""),
                "intent": str(lead.get("intent") or ""),
                "stage": str(lead.get("stage") or ""),
                "session_id": str(lead.get("session_id") or ""),
                "created_at": timestamp,
                "updated_at": timestamp,
            }
            rows.append(record)
        else:
            record = dict(existing)
            for column in ("name", "email", "phone", "score", "urgency", "intent", "session_id"):
                value = lead.get(column)
                if value is not None:
                    record[column] = str(value)
            record["updated_at"] = timestamp
        try:
            self._store.save(rows)
        except CsvStoreError as exc:
            logger.error("crm store unwritable: %s", exc)
            raise CrmError("crm store unwritable") from exc
        return record

    def get_lead(self, lead_id: str) -> dict[str, Any] | None:
        try:
            rows = self._store.load()
        except CsvStoreError as exc:
            logger.error("crm store unreadable: %s", exc)
            raise CrmError("crm store unreadable") from exc
        return next((row for row in rows if row["lead_id"] == lead_id), None)

    def update_stage(self, lead_id: str, stage: str) -> None:
        try:
            rows = self._store.load()
        except CsvStoreError as exc:
            logger.error("crm store unreadable: %s", exc)
            raise CrmError("crm store unreadable") from exc
        existing = next((row for row in rows if row["lead_id"] == lead_id), None)
        if existing is None:
            raise LeadNotFoundError(f"lead {lead_id} not found in crm")
        existing["stage"] = stage
        existing["updated_at"] = self._clock().isoformat()
        try:
            self._store.save(rows)
        except CsvStoreError as exc:
            logger.error("crm store unwritable: %s", exc)
            raise CrmError("crm store unwritable") from exc


class McpUnavailableError(CrmError):
    pass


try:  # pragma: no cover - dependência pesada opcional (POC não exige)
    from mcp import ClientSession

    _HAS_MCP = True
except ImportError:  # pragma: no cover
    ClientSession = None  # type: ignore[assignment,misc]
    _HAS_MCP = False


def build_hubspot_client() -> Any:
    """Fábrica do cliente MCP HubSpot — só usada na demo ao vivo (FR11.1)."""
    if not _HAS_MCP:
        raise McpUnavailableError("mcp sdk not installed")
    return ClientSession  # placeholder de wiring real fora da POC


class McpClient(Protocol):
    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


class McpCrmGateway:
    """CRM real (HubSpot/Kenlo) atrás da interface CrmGateway, via MCP.

    O cliente MCP é injetado; a suíte nunca exige o SDK real.
    """

    def __init__(self, mcp_client: McpClient) -> None:
        self._client = mcp_client

    def upsert_lead(self, lead: dict[str, Any]) -> dict[str, Any]:
        properties = {
            "lead_id": str(lead.get("lead_id") or ""),
            "name": str(lead.get("name") or ""),
            "email": str(lead.get("email") or ""),
            "phone": str(lead.get("phone") or ""),
            "score": str(lead.get("score") or ""),
            "urgency": str(lead.get("urgency") or ""),
            "intent": str(lead.get("intent") or ""),
            "session_id": str(lead.get("session_id") or ""),
        }
        result = self._call("crm_upsert_lead", {"properties": properties})
        return result.get("record", result)

    def get_lead(self, lead_id: str) -> dict[str, Any] | None:
        result = self._call("crm_get_lead", {"lead_id": lead_id})
        return result.get("record")

    def update_stage(self, lead_id: str, stage: str) -> None:
        self._call("crm_update_stage", {"lead_id": lead_id, "stage": stage})

    def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self._client.call_tool(name, arguments)
        except Exception as exc:
            logger.error("mcp tool %s failed: %s", name, exc)
            raise CrmError(f"mcp tool {name} failed") from exc
        if not isinstance(result, dict):
            logger.error("mcp tool %s returned non-object result", name)
            raise CrmError(f"mcp tool {name} returned invalid payload")
        return result