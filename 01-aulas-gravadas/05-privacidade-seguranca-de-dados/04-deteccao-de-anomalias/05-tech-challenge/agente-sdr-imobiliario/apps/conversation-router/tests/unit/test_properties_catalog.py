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


def test_purchase_intent_never_falls_back_to_rent_listings():
    catalog = [
        {"id": "r1", "title": "Rent 1", "mode": "rent", "price": 1000},
        {"id": "r2", "title": "Rent 2", "mode": "rent", "price": 2000},
        {"id": "r3", "title": "Rent 3", "mode": "rent", "price": 3000},
        {"id": "p1", "title": "Purchase 1", "mode": "purchase", "price": 3_000_000},
    ]

    result = pc.search_properties({"intent": "purchase"}, catalog=catalog, top_k=3)

    assert [prop["id"] for prop in result] == ["p1"]


def test_purchase_intent_returns_empty_when_catalog_has_no_purchase_inventory():
    result = pc.search_properties(
        {"intent": "purchase"}, catalog=PROPERTIES[:2], top_k=3
    )

    assert result == []


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


class FakePropertiesClient:
    def __init__(self, pages=None, error=None):
        self._pages = pages or [{"Items": []}]
        self._error = error
        self.calls = []

    def scan(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._pages[len(self.calls) - 1]


class TestUnmarshalPropertyItem:
    def test_unmarshals_scalars(self):
        item = {
            "id": {"S": "p1"},
            "title": {"S": "Laje na Faria Lima"},
            "area_util": {"N": "200"},
            "price": {"N": "15000.5"},
            "laje": {"BOOL": True},
            "cep": {"NULL": True},
        }
        out = pc.unmarshal_property_item(item)
        assert out == {
            "id": "p1",
            "title": "Laje na Faria Lima",
            "area_util": 200,
            "price": 15000.5,
            "laje": True,
            "cep": None,
        }

    def test_unmarshals_string_list(self):
        item = {"regions": {"L": [{"S": "Itaim Bibi"}, {"S": "Faria Lima"}]}}
        assert pc.unmarshal_property_item(item) == {
            "regions": ["Itaim Bibi", "Faria Lima"]
        }


class TestLoadFromDynamoDB:
    def teardown_method(self):
        pc.set_dynamodb_client(None)

    def test_no_client_returns_empty(self, monkeypatch):
        monkeypatch.setenv("PROPERTIES_TABLE", "sdr-properties")
        pc.set_dynamodb_client(None)
        assert pc._load_from_dynamodb() == []

    def test_no_table_env_returns_empty(self, monkeypatch):
        monkeypatch.delenv("PROPERTIES_TABLE", raising=False)
        pc.set_dynamodb_client(FakePropertiesClient())
        assert pc._load_from_dynamodb() == []

    def test_scans_and_unmarshals_items(self, monkeypatch):
        monkeypatch.setenv("PROPERTIES_TABLE", "sdr-properties")
        client = FakePropertiesClient(
            pages=[{"Items": [{"id": {"S": "p1"}, "price": {"N": "1000"}}]}]
        )
        pc.set_dynamodb_client(client)
        assert pc._load_from_dynamodb() == [{"id": "p1", "price": 1000}]
        assert client.calls[0]["TableName"] == "sdr-properties"

    def test_paginates_via_last_evaluated_key(self, monkeypatch):
        monkeypatch.setenv("PROPERTIES_TABLE", "sdr-properties")
        client = FakePropertiesClient(
            pages=[
                {
                    "Items": [{"id": {"S": "p1"}}],
                    "LastEvaluatedKey": {"id": {"S": "p1"}},
                },
                {"Items": [{"id": {"S": "p2"}}]},
            ]
        )
        pc.set_dynamodb_client(client)
        assert pc._load_from_dynamodb() == [{"id": "p1"}, {"id": "p2"}]
        assert "ExclusiveStartKey" in client.calls[1]

    def test_scan_error_fails_open_to_empty(self, monkeypatch):
        monkeypatch.setenv("PROPERTIES_TABLE", "sdr-properties")
        pc.set_dynamodb_client(FakePropertiesClient(error=RuntimeError("down")))
        assert pc._load_from_dynamodb() == []


class TestLoadDefault:
    def teardown_method(self):
        pc.set_dynamodb_client(None)

    def test_falls_back_to_local_json_without_dynamodb(self, monkeypatch):
        monkeypatch.delenv("PROPERTIES_TABLE", raising=False)
        pc.set_dynamodb_client(None)
        monkeypatch.setattr(pc, "_load", lambda catalog_path=None: PROPERTIES)
        assert pc._load_default() == PROPERTIES

    def test_prefers_dynamodb_when_available(self, monkeypatch):
        monkeypatch.setenv("PROPERTIES_TABLE", "sdr-properties")
        client = FakePropertiesClient(pages=[{"Items": [{"id": {"S": "p1"}}]}])
        pc.set_dynamodb_client(client)
        assert pc._load_default() == [{"id": "p1"}]

    def test_falls_back_when_dynamodb_scan_is_empty(self, monkeypatch):
        monkeypatch.setenv("PROPERTIES_TABLE", "sdr-properties")
        pc.set_dynamodb_client(FakePropertiesClient(pages=[{"Items": []}]))
        monkeypatch.setattr(pc, "_load", lambda catalog_path=None: PROPERTIES)
        assert pc._load_default() == PROPERTIES


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


def test_search_list_scope_all_returns_more():
    from service.properties_catalog import search_properties

    props = search_properties({"region": "Moema"}, top_k=3, list_scope="all")
    assert len(props) >= 3


def test_search_list_scope_filtered_default():
    from service.properties_catalog import search_properties

    props = search_properties({"region": "Moema"}, top_k=3)
    assert len(props) <= 3


def test_search_list_scope_all_returns_more():
    from service.properties_catalog import search_properties

    props = search_properties({"region": "Moema"}, top_k=3, list_scope="all")
    assert len(props) >= 3


def test_search_list_scope_filtered_default():
    from service.properties_catalog import search_properties

    props = search_properties({"region": "Moema"}, top_k=3)
    assert len(props) <= 3