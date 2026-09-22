"""Catálogo sintético de imóveis + busca por filtro (RAG da POC).

O PRD (FR-11, §8.2) manda "base simulada de imóveis corporativos sintéticos +
RAG": NENHUM dado real é necessário. A calibração (metragem/região/ticket) segue
Mercado Livre/Viva Real/consultorias, e o lookup determinístico (escore por
intent + região + área ± 25% + orçamento) substitui o FAISS da arquitetura-alvo
sem adicionar peso de embeddings no zip da Lambda (limite 50MB).

Fonte: data/properties.json (gerado por scripts/seed_properties.py). A resposta
do agente SEMPRE vem daqui — constraint do PRD: "nunca inventar imóveis".
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "properties.json")

_AREA_RE = re.compile(r"(\d+(?:[.,]\d+)?)")
_AMOUNT_UNIT_RE = r"(?:mil(?:h[õo]es|h[ãa]o)?|k)"
_AMOUNT_RE = re.compile(rf"\b(\d+(?:[.,]\d+)?)\s*({_AMOUNT_UNIT_RE})\b", re.IGNORECASE)

# Peso por overfit: intenção > região > orçamento > metragem.
_REGION_BONUS = 3
_INTENT_BONUS = 4
_PRICE_WITHIN_BUDGET = 2


def _load(catalog_path: str | None = None) -> list[dict[str, Any]]:
    """Carrega o catálogo sintético; loga e devolve [] se indisponível (fail-open)."""
    path = catalog_path or _DATA_PATH
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        items = data.get("properties") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ValueError("catálogo sem lista 'properties'")
        return items
    except FileNotFoundError:
        logger.warning("catálogo de imóveis ausente: %s (RAG indisponível)", path)
        return []
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("catálogo de imóveis inválido: %s (%s)", path, exc)
        return []


def _area_m2(area: Any) -> float | None:
    if not area:
        return None
    match = _AREA_RE.search(str(area))
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def parse_budget(raw: Any) -> float | None:
    """R$ 1,5 milhão | 50 mil | 250k | R$ 1.200.000 | R$ 8.500 -> float BRL."""
    if not raw:
        return None
    text = str(raw).strip()
    unit_match = _AMOUNT_RE.search(text)
    if unit_match:
        base = float(unit_match.group(1).replace(",", "."))
        mult = 1_000_000 if unit_match.group(2).lower().startswith("milh") else 1_000
        return base * mult
    plain = text.replace("R$", "").replace("$", "").strip()
    thr = re.search(r"(\d{1,3}(?:\.\d{3})+(?:,\d+)?)", plain)
    if thr:
        return float(thr.group(1).replace(".", "").replace(",", "."))
    dec = re.search(r"(\d+(?:,\d+)?)", plain)
    if dec:
        return float(dec.group(1).replace(",", "."))
    return None


def _norm_region(region: Any) -> str | None:
    if not region:
        return None
    value = str(region).lower().strip()
    return value or None


def search_properties(
    lead_info: dict[str, Any],
    catalog: list[dict[str, Any]] | None = None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Retorna até top_k imóveis ranqueados pelos filtros do lead (RAG)."""
    items = catalog if catalog is not None else _load()
    if not items:
        return []
    intent = (lead_info.get("intent") or "").lower()
    region = _norm_region(lead_info.get("region"))
    area_m2 = _area_m2(lead_info.get("area"))
    budget = parse_budget(lead_info.get("budget"))

    ranked: list[tuple[float, dict[str, Any]]] = []
    for prop in items:
        score = 0.0
        pintent = str(prop.get("mode") or "").lower()
        if intent and pintent == intent:
            score += _INTENT_BONUS
        if intent in ("purchase", "investment") and pintent == "rent":
            score -= _INTENT_BONUS
        if region:
            prop_regions = " ".join(
                {str(prop.get("region", "")), *map(str, prop.get("regions") or [])}
            ).lower()
            if region in prop_regions or any(tok in prop_regions for tok in region.split()):
                score += _REGION_BONUS
        price = _fprice(prop.get("price"))
        if budget:
            if price is None or price > budget * 1.25:
                continue
            score += _PRICE_WITHIN_BUDGET
        parea = _farea(prop.get("area_util"))
        if area_m2 and parea is not None:
            if parea <= area_m2 * 1.25 and parea >= area_m2 * 0.75:
                score += 1
            else:
                score -= abs(parea - area_m2) / max(area_m2, 1) * 0.5
        ranked.append((score, prop))

    ranked.sort(key=lambda pair: (-pair[0], _fprice(pair[1].get("price")) or 0.0))
    result = [prop for _, prop in ranked[:top_k]]
    logger.info("RAG: %d candidatos -> %d recomendações", len(items), len(result))
    return result


def _fprice(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _farea(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)