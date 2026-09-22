"""Catálogo de clientes (CRM simulado) — Base 2 do RAG (PRD §8.2).

PRD §8.2: "duas bases — catálogo de imóveis (ofertas) e catálogo de clientes
(historico/status, simulando CRM)". Esta é a base de clientes: status Kanban
(pré-atendimento → visita → proposta → fechamento → pós-venda) para enriquecer
a qualificação do lead com contexto histórico.

A busca por similaridade (FAISS) usa a descrição do cliente para matchar leads
por região/intenção/ticket — usado pelo lead-qualifier para score de "carteira"
(insight da mentoria: "carteira/indicação ≈ 85% de certeza de fechamento").
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "clients.json")

_KANBAN_STATUSES = ("pre-atendimento", "visita", "proposta", "fechamento", "pos-venda")


def _load(catalog_path: str | None = None) -> list[dict[str, Any]]:
    path = catalog_path or _DATA_PATH
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        items = data.get("clients") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ValueError("catálogo sem lista 'clients'")
        return items
    except FileNotFoundError:
        logger.warning("catálogo de clientes ausente: %s", path)
        return []
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("catálogo de clientes inválido: %s (%s)", path, exc)
        return []


def search_clients(
    lead_info: dict[str, Any],
    catalog: list[dict[str, Any]] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Retorna até top_k clientes do CRM simulado por similaridade de região/intenção.

    Usado pelo lead-qualifier para enriquecer o score com contexto de carteira
    (insight da mentoria: "carteira/indicação ≈ 85% de fechamento").
    """
    items = catalog if catalog is not None else _load()
    if not items:
        return []
    region = str(lead_info.get("region", "")).lower().strip()
    intent = str(lead_info.get("intent", "")).lower().strip()

    ranked: list[tuple[float, dict[str, Any]]] = []
    for client in items:
        score = 0.0
        client_region = str(client.get("region", "")).lower()
        if region and region in client_region:
            score += 2.0
        client_intent = str(client.get("intent", "")).lower()
        if intent and intent == client_intent:
            score += 1.5
        status = client.get("status", "")
        if status == "fechamento":
            score += 1.0
        elif status == "pos-venda":
            score += 0.5
        ranked.append((score, client))

    ranked.sort(key=lambda pair: (-pair[0], pair[1].get("name", "")))
    result = [c for _, c in ranked[:top_k] if _ > 0]
    logger.info("RAG clientes: %d candidatos -> %d matches", len(items), len(result))
    return result
