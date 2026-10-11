"""Catálogo sintético de imóveis + RAG via FAISS (PRD §8.2: FAISS local + embeddings).

O índice FAISS é construído em memória a partir dos embeddings TF-IDF das
descrições dos imóveis. A busca combina similaridade vetorial (FAISS) com
filtros determinísticos (intent/região/orçamento/área) — top-k por score híbrido.

Fonte: data/properties.json (gerado por scripts/seed_properties.py, 120 imóveis
sintéticos com campos de negócio do PRD §8.2). O catálogo padrão é carregado da
tabela DynamoDB `sdr-properties` (populada por scripts/load_properties_dynamodb.py
após o deploy) em um único Scan no cold start; sem `PROPERTIES_TABLE` configurado
(dev local/testes) ou em caso de falha, cai de volta para o JSON local
(data/ é gitignored e regenerado no build via start.sh).

Embeddings: TF-IDF esparso sobre n-gramas de caracteres (PT-BR robusto, sem
depender de modelo de embedding externo). FAISS IndexFlatIP (inner product)
para cosine similarity. ~120 docs carrega em <1ms.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "properties.json")

_AREA_RE = re.compile(r"(\d+(?:[.,]\d+)?)")
_AMOUNT_UNIT_RE = r"(?:mil(?:h[õo]es|h[ãa]o)?|k)"
_AMOUNT_RE = re.compile(rf"\b(\d+(?:[.,]\d+)?)\s*({_AMOUNT_UNIT_RE})\b", re.IGNORECASE)

_REGION_BONUS = 3
_INTENT_BONUS = 4
_PRICE_WITHIN_BUDGET = 2


# --- TF-IDF embedding (leve, sem dependência externa) -----------------------

_VOCAB_CACHE: dict[str, int] = {}
_IDF_CACHE: np.ndarray | None = None
_DOCS_CACHE: list[str] = []
_INDEX_CACHE: faiss.IndexFlatIP | None = None
_CATALOG_CACHE: list[dict[str, Any]] = []

_DYNAMO_CLIENT: Any | None = None


def set_dynamodb_client(client: Any | None) -> None:
    """Injeta o cliente boto3 DynamoDB usado pelo carregamento padrão do catálogo.

    Chamado uma vez no cold start (handler.py), no mesmo padrão de injeção de
    SessionStore/KmsPiiRegistry/DynamoRestrictionCheck.
    """
    global _DYNAMO_CLIENT
    _DYNAMO_CLIENT = client


def _tokenize(text: str) -> list[str]:
    """N-gramas de caracteres (3-5) — robusto para PT-BR sem modelo externo."""
    text = text.lower().strip()
    grams: list[str] = []
    for word in re.findall(r"[a-zà-ÿ0-9]+", text):
        padded = f"#{word}#"
        for n in (3, 4, 5):
            for i in range(len(padded) - n + 1):
                grams.append(padded[i : i + n])
    return grams


def _build_tfidf(documents: list[str]) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """Constrói matriz TF-IDF esparsa (densa para ~120 docs)."""
    # Vocabulário
    vocab: dict[str, int] = {}
    tokenized: list[list[str]] = []
    for doc in documents:
        tokens = _tokenize(doc)
        tokenized.append(tokens)
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    n_docs = len(documents)
    n_vocab = len(vocab)
    # TF
    tf = np.zeros((n_docs, n_vocab), dtype=np.float32)
    for i, tokens in enumerate(tokenized):
        for tok in tokens:
            tf[i, vocab[tok]] += 1
    # IDF
    df = (tf > 0).sum(axis=0).astype(np.float32)
    idf = np.log((1 + n_docs) / (1 + df)) + 1.0
    tfidf = tf * idf
    # L2 normalize
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tfidf = tfidf / norms
    return tfidf, vocab, idf


def _vectorize_query(text: str, vocab: dict[str, int], idf: np.ndarray) -> np.ndarray:
    n_vocab = len(vocab)
    vec = np.zeros(n_vocab, dtype=np.float32)
    for tok in _tokenize(text):
        if tok in vocab:
            vec[vocab[tok]] += 1
    vec *= idf
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _ensure_index(catalog: list[dict[str, Any]] | None = None) -> None:
    """Constrói o índice FAISS se ainda não construído (lazy, uma vez por processo)."""
    global _VOCAB_CACHE, _IDF_CACHE, _DOCS_CACHE, _INDEX_CACHE, _CATALOG_CACHE

    if _INDEX_CACHE is not None and catalog is None:
        return

    items = catalog if catalog is not None else _load_default()
    if not items:
        return

    docs = [
        f"{p.get('title', '')} {p.get('description', '')} {p.get('region', '')} "
        f"{p.get('type', '')} {p.get('class', '')} {p.get('disponibilidade', '')} "
        f"{p.get('mode', '')}"
        for p in items
    ]
    tfidf, vocab, idf = _build_tfidf(docs)

    dim = tfidf.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(tfidf)

    _VOCAB_CACHE = vocab
    _IDF_CACHE = idf
    _DOCS_CACHE = docs
    _INDEX_CACHE = index
    _CATALOG_CACHE = items
    logger.info("FAISS index built: %d docs, dim=%d", index.ntotal, dim)


def _load(catalog_path: str | None = None) -> list[dict[str, Any]]:
    """Carrega o catálogo sintético; fail-open → [] se indisponível."""
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


def unmarshal_property_item(item: dict[str, Any]) -> dict[str, Any]:
    """Desserializa um item DynamoDB (low-level, tipos {"S": ...}/{"N": ...}/{"L": ...})."""
    out: dict[str, Any] = {}
    for key, raw in item.items():
        if isinstance(raw, dict) and len(raw) == 1:
            kind, value = next(iter(raw.items()))
            if kind == "BOOL":
                out[key] = value
            elif kind == "N":
                number = float(value)
                out[key] = int(number) if number.is_integer() else number
            elif kind == "NULL":
                out[key] = None
            elif kind == "L":
                out[key] = [
                    next(iter(v.values())) if isinstance(v, dict) and len(v) == 1 else v
                    for v in value
                ]
            else:
                out[key] = value
        else:
            out[key] = raw
    return out


def _load_from_dynamodb() -> list[dict[str, Any]]:
    """Scan completo da tabela `PROPERTIES_TABLE`; fail-open → [] se indisponível."""
    table = os.environ.get("PROPERTIES_TABLE")
    if _DYNAMO_CLIENT is None or not table:
        return []
    items: list[dict[str, Any]] = []
    kwargs: dict[str, Any] = {"TableName": table}
    try:
        while True:
            response = _DYNAMO_CLIENT.scan(**kwargs)
            items.extend(unmarshal_property_item(raw) for raw in response.get("Items", []))
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            kwargs["ExclusiveStartKey"] = last_key
    except Exception as exc:
        logger.warning("catálogo de imóveis no DynamoDB indisponível (%s); usando JSON local", exc)
        return []
    return items


def _load_default() -> list[dict[str, Any]]:
    """Catálogo padrão: DynamoDB (se configurado) com fallback para o JSON local."""
    items = _load_from_dynamodb()
    return items if items else _load()


# --- Parsers pt-BR -----------------------------------------------------------

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


def _fold(value: Any) -> str:
    """Minúsculas, sem acento e sem pontuação — só normalização de texto para comparar."""
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _norm_region(region: Any) -> str | None:
    return _fold(region) or None


def known_places(catalog: list[dict[str, Any]] | None = None) -> dict[str, list[str]]:
    """Vocabulário do catálogo: cidade (corredor) -> bairros. Vai pro roteador LLM, que
    grava no lead_info o nome EXATO do lugar ("scs", "sao caetano", "são caetano do sul"
    -> "São Caetano do Sul") em vez de o código tentar adivinhar por palavra."""
    items = catalog if catalog is not None else _load_default()
    places: dict[str, set[str]] = {}
    for prop in items:
        city = str(prop.get("corredor") or "").strip()
        if not city:
            continue
        bucket = places.setdefault(city, set())
        for name in (prop.get("region"), *(prop.get("regions") or [])):
            if name and str(name).strip():
                bucket.add(str(name).strip())
    return {city: sorted(names) for city, names in sorted(places.items())}


def _region_matches(region: str, prop: dict[str, Any]) -> bool:
    """Compara o lugar que a LLM gravou (nome do catálogo) com a cidade (corredor) ou o
    bairro do imóvel — igualdade/contenção de texto normalizado, sem lista de palavras.
    Entender "scs", "sao caetano" ou "perto da paulista" é trabalho da LLM."""
    wanted = _fold(region)
    if not wanted:
        return False
    for name in (prop.get("corredor"), prop.get("region"), *(prop.get("regions") or [])):
        place = _fold(name)
        if place and (wanted == place or wanted in place or place in wanted):
            return True
    return False


# --- Busca RAG (FAISS + filtros) ---------------------------------------------

def search_properties(
    lead_info: dict[str, Any],
    catalog: list[dict[str, Any]] | None = None,
    top_k: int = 3,
    list_scope: str = "filtered",
) -> list[dict[str, Any]]:
    """Retorna até top_k imóveis ranqueados por RAG (FAISS) + filtros do lead.

    list_scope="all" eleva top_k para >= 50 (pedido "mostra tudo").
    """
    if list_scope == "all":
        top_k = max(top_k, 50)
    items = catalog if catalog is not None else _load_default()
    if not items:
        return []

    intent = (lead_info.get("intent") or "").lower()
    requested_mode = "purchase" if intent == "investment" else intent
    region = _norm_region(lead_info.get("region"))
    area_m2 = _area_m2(lead_info.get("area"))
    budget = parse_budget(lead_info.get("budget"))  # teto
    budget_min = parse_budget(lead_info.get("budget_min"))  # piso ("a partir de")

    # Do not substitute rent listings when the lead asked to buy (or vice versa).
    mode_candidates = (
        [p for p in items if str(p.get("mode") or "").lower() == requested_mode]
        if requested_mode in ("purchase", "rent")
        else items
    )
    if not mode_candidates:
        return []

    # 1. Filtros determinísticos (hard filters)
    candidates: list[dict[str, Any]] = []
    for prop in mode_candidates:
        price = _fprice(prop.get("price"))
        if budget and (price is None or price > budget * 1.25):
            continue
        if budget_min and price is not None and price < budget_min * 0.8:
            continue
        candidates.append(prop)

    # fail-open: sem candidatos OU poucos no orçamento (soft budget).
    # Sem soft, budget R$5k deixa 1 de 120 → bot repete o mesmo imóvel.
    min_for_strict = max(top_k, 3)
    if len(candidates) < min_for_strict:
        candidates = mode_candidates

    # 2. FAISS: similaridade vetorial da query contra descrições
    query_text = " ".join(str(v) for v in lead_info.values() if v)
    _ensure_index(candidates if catalog is None else items)

    faiss_scores: dict[int, float] = {}
    if _INDEX_CACHE is not None and _VOCAB_CACHE:
        qvec = _vectorize_query(query_text, _VOCAB_CACHE, _IDF_CACHE).reshape(1, -1)
        D, I = _INDEX_CACHE.search(qvec, min(_INDEX_CACHE.ntotal, top_k * 5))
        for idx, score in zip(I[0], D[0]):
            if idx >= 0:
                faiss_scores[int(idx)] = float(score)

    # 3. Score híbrido: FAISS + filtros determinísticos
    ranked: list[tuple[float, dict[str, Any]]] = []
    for prop in candidates:
        score = 0.0
        pintent = str(prop.get("mode") or "").lower()
        if requested_mode and pintent == requested_mode:
            score += _INTENT_BONUS
        if requested_mode == "purchase" and pintent == "rent":
            score -= _INTENT_BONUS
        if region and _region_matches(region, prop):
            score += _REGION_BONUS
        price = _fprice(prop.get("price"))
        if (
            (budget or budget_min)
            and price is not None
            and (not budget or price <= budget)
            and (not budget_min or price >= budget_min)
        ):
            score += _PRICE_WITHIN_BUDGET
        parea = _farea(prop.get("area_util"))
        if area_m2 and parea is not None:
            if parea <= area_m2 * 1.25 and parea >= area_m2 * 0.75:
                score += 1
            else:
                score -= abs(parea - area_m2) / max(area_m2, 1) * 0.5
        # FAISS semantic similarity bonus
        prop_idx = None
        if catalog is None and _CATALOG_CACHE:
            try:
                prop_idx = _CATALOG_CACHE.index(prop)
            except ValueError:
                pass
        if prop_idx is not None and prop_idx in faiss_scores:
            score += faiss_scores[prop_idx] * 2.0
        ranked.append((score, prop))

    ranked.sort(key=lambda pair: (-pair[0], _fprice(pair[1].get("price")) or 0.0))
    result = [prop for _, prop in ranked[:top_k]]
    logger.info("RAG: %d candidatos -> %d recomendações (FAISS)", len(candidates), len(result))
    return result


def _fprice(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _farea(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
