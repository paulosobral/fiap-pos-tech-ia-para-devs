"""Popula a tabela DynamoDB `sdr-properties` a partir de data/properties.json.

Invocado pelo start.sh após o deploy (fase 5), depois do scale-out do ECS e
antes do tráfego real ser roteado. Idempotente: upsert por `id` via
`batch_write_item` em lotes de 25 (limite do BatchWriteItem) — seguro para
rodar em todo start.sh.

Uso: PROPERTIES_TABLE=sdr-properties python scripts/load_properties_dynamodb.py
(região/credenciais ambiente, mesma convenção do resto do repo).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_DATA_PATH = Path(__file__).resolve().parent.parent / "apps" / "conversation-router" / "data" / "properties.json"
_BATCH_SIZE = 25


def _marshal(value: Any) -> dict[str, Any]:
    if value is None:
        return {"NULL": True}
    if isinstance(value, bool):
        return {"BOOL": value}
    if isinstance(value, (int, float)):
        return {"N": str(value)}
    if isinstance(value, list):
        return {"L": [_marshal(v) for v in value]}
    return {"S": str(value)}


def _marshal_item(item: dict[str, Any]) -> dict[str, Any]:
    return {key: _marshal(value) for key, value in item.items() if value is not None}


def load_properties(data_path: Path = _DATA_PATH) -> list[dict[str, Any]]:
    with open(data_path, encoding="utf-8") as fh:
        data = json.load(fh)
    items = data.get("properties") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"catálogo sem lista 'properties': {data_path}")
    return items


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def seed_table(client: Any, table_name: str, properties: list[dict[str, Any]]) -> int:
    written = 0
    for batch in _chunks(properties, _BATCH_SIZE):
        request_items = {
            table_name: [{"PutRequest": {"Item": _marshal_item(prop)}} for prop in batch]
        }
        while request_items:
            response = client.batch_write_item(RequestItems=request_items)
            written += len(request_items[table_name])
            request_items = response.get("UnprocessedItems") or {}
    return written


def main() -> None:
    table_name = os.environ.get("PROPERTIES_TABLE", "sdr-properties")
    properties = load_properties()
    if not properties:
        print(f"AVISO: nenhum imóvel encontrado em {_DATA_PATH}; nada a popular.")
        return

    import boto3

    client = boto3.client("dynamodb")
    written = seed_table(client, table_name, properties)
    print(f"{written} imóveis carregados na tabela {table_name}.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # falha aqui não deve derrubar o deploy (fail-open)
        print(f"AVISO: falha ao popular catálogo no DynamoDB: {exc}", file=sys.stderr)
        sys.exit(1)
