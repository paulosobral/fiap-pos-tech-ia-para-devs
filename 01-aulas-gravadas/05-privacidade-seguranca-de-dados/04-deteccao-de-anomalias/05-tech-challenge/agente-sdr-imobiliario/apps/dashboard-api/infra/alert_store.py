from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

ALERT_STATUS_OPEN = "open"
_JSON_FIELDS = ("features",)


class AlertStoreError(Exception):
    pass


def unmarshal_item(item: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, raw in item.items():
        if isinstance(raw, dict) and len(raw) == 1:
            kind, value = next(iter(raw.items()))
            if kind == "BOOL":
                out[key] = value
            elif kind == "N":
                number = float(value)
                out[key] = int(number) if number.is_integer() else number
            else:
                out[key] = _decode_json_field(key, str(value))
        else:
            out[key] = raw
    return out


def _decode_json_field(key: str, value: str) -> Any:
    if key in _JSON_FIELDS and value[:1] in ("[", "{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def project_alert(item: dict[str, Any]) -> dict[str, Any]:
    """Projeção PII-safe do alerta (Contrato 7) para o payload do dashboard —
    `features` fica fora (payload mínimo, FR7.4)."""
    return {
        "anomaly_id": item.get("anomaly_id"),
        "lead_id": item.get("lead_id"),
        "type": item.get("type"),
        "confidence": item.get("confidence"),
        "detected_at": item.get("detected_at"),
        "status": item.get("status"),
        "action_taken": item.get("action_taken"),
    }


class AlertStoreReader:
    """Leitor do Contrato 7 (alertas) — U5 é owner, U7 é reader.

    Scan paginado da tabela `sdr-alerts` (schema do dono U5: `anomaly_id`,
    `lead_id`, `features`, `confidence`, `type`, `detected_at`, `status`,
    `action_taken`). Itens sem `anomaly_id` são ignorados com log; ordena por
    `detected_at` desc. Erros de cliente viram `AlertStoreError` (Lambda → 500).
    """

    def __init__(self, dynamodb_client: Any, table_name: str = "sdr-alerts") -> None:
        self._client = dynamodb_client
        self._table = table_name

    def list_alerts(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {"TableName": self._table}
        try:
            while True:
                response = self._client.scan(**kwargs)
                for raw in response.get("Items", []):
                    item = unmarshal_item(raw)
                    if not item.get("anomaly_id"):
                        logger.warning("alert item missing anomaly_id; skipped")
                        continue
                    items.append(item)
                last_key = response.get("LastEvaluatedKey")
                if not last_key:
                    break
                kwargs["ExclusiveStartKey"] = last_key
        except AlertStoreError:
            raise
        except Exception as exc:
            logger.error("alert store unavailable: %s", exc)
            raise AlertStoreError("alert store unavailable") from exc
        items.sort(key=lambda item: str(item.get("detected_at") or ""), reverse=True)
        return items
