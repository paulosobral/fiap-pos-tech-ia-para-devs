from __future__ import annotations

import json

import pytest

from service import properties_catalog as pc

PROPERTIES = [
    {
        "id": "p1",
        "title": "Laje na Faria Lima",
        "region": "Faria Lima",
        "regions": ["Itaim Bibi"],
        "area_util": 200,
        "mode": "rent",
        "price": 15_000,
    },
    {
        "id": "p2",
        "title": "Escritório no Centrão",
        "region": "Centro",
        "regions": ["República"],
        "area_util": 100,
        "mode": "rent",
        "price": 6_000,
    },
    {
        "id": "p3",
        "title": "Conjunto na Vila Olímpia",
        "region": "Vila Olímpia",
        "regions": ["Itaim Bibi"],
        "area_util": 300,
        "mode": "purchase",
        "price": 3_000_000,
    },
    {
        "id": "p4",
        "title": "Sala no Itaim",
        "region": "Itaim Bibi",
        "regions": ["Faria Lima"],
        "area_util": 180,
        "mode": "rent",
        "price": 20_000,
    },
]


def test_search_matches_intent_region_and_budget():
    result = pc.search_properties(
        {
            "intent": "rent",
            "region": "Itaim Bibi",
            "area": "200 m²",
            "budget": "R$ 18 mil",
        },
        catalog=PROPERTIES,
    )
    ids = [p["id"] for p in result]
    assert ids[0] == "p1"
    assert "p3" not in ids


def test_search_prefers_purchase_when_intent_purchase():
    result = pc.search_properties({"intent": "purchase"}, catalog=PROPERTIES)
    assert result[0]["id"] == "p3"


def test_search_strict_budget_when_enough_in_budget_candidates():
    catalog = [
        {"id": "a", "title": "A", "region": "Moema", "mode": "rent", "price": 4_000},
        {"id": "b", "title": "B", "region": "Moema", "mode": "rent", "price": 5_000},
        {"id": "c", "title": "C", "region": "Moema", "mode": "rent", "price": 6_000},
        {"id": "d", "title": "D", "region": "Moema", "mode": "rent", "price": 15_000},
        {"id": "e", "title": "E", "region": "Moema", "mode": "rent", "price": 20_000},
    ]
    result = pc.search_properties(
        {"intent": "rent", "budget": "R$ 5 mil"}, catalog=catalog, top_k=3
    )
    ids = [p["id"] for p in result]
    assert "d" not in ids
    assert "e" not in ids
    assert len(result) == 3


def test_search_soft_budget_when_few_in_budget_candidates():
    catalog = [
        {"id": "cheap", "title": "Barato", "region": "Moema", "mode": "rent", "price": 4_000},
        {"id": "mid", "title": "Médio", "region": "Moema", "mode": "rent", "price": 9_000},
        {"id": "high", "title": "Alto", "region": "Moema", "mode": "rent", "price": 20_000},
        {"id": "lux", "title": "Luxo", "region": "Moema", "mode": "rent", "price": 40_000},
    ]
    result = pc.search_properties(
        {"intent": "rent", "region": "Moema", "budget": "R$ 5 mil"},
        catalog=catalog,
        top_k=3,
    )
    ids = [p["id"] for p in result]
    assert len(result) >= 3
    assert ids[0] == "cheap"


def test_search_empty_catalog_returns_empty():
    assert pc.search_properties({}, catalog=[]) == []


def test_search_top_k_limit():
    result = pc.search_properties({"intent": "rent", "region": "Itaim Bibi"}, catalog=PROPERTIES, top_k=1)
    assert len(result) == 1


def test_search_missing_file_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(pc, "_DATA_PATH", str(tmp_path / "nope.json"))
    assert pc._load() == []


def test_load_invalid_json_returns_empty(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert pc._load(str(bad)) == []


def test_load_non_list_root_returns_empty(tmp_path):
    f = tmp_path / "f.json"
    f.write_text(json.dumps({"properties": "nope"}), encoding="utf-8")
    assert pc._load(str(f)) == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("R$ 50 mil", 50_000),
        ("1,5 milhão", 1_500_000),
        ("250k", 250_000),
        ("R$ 1.200.000", 1_200_000),
        ("150 mil", 150_000),
        ("R$ 8.500", 8_500),
        ("sem valor", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_budget(raw: object, expected: object):
    assert pc.parse_budget(raw) == expected


def test_area_parser():
    assert pc._area_m2("200 m²") == 200
    assert pc._area_m2("180,5 m²") == 180.5
    assert pc._area_m2(None) is None
    assert pc._area_m2("nada") is None


def test_norm_region():
    assert pc._norm_region("  Faria Lima ") == "faria lima"
    assert pc._norm_region(None) is None
    assert pc._norm_region("") is None