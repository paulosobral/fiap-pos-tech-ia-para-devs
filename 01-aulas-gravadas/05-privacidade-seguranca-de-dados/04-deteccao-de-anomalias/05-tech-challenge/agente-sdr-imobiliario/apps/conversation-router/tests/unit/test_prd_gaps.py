from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from service.clients_catalog import search_clients
from service.flow.lead_qualifier import LeadQualifier
from service.flow.sales_flow import SalesFlow, _build_ics
from service.properties_catalog import search_properties


def test_sales_flow_uses_langgraph_when_available():
    flow = SalesFlow(LeadQualifier())
    assert flow._graph is not None, "LangGraph graph must be compiled"
    res = flow.invoke({"current_state": "greeting", "message": "olá"})
    assert res["current_state"] == "elicitation"
    assert "LGPD" in res["response"] or "dados" in res["response"]


def test_sales_flow_ics_generated_on_scheduling():
    flow = SalesFlow(
        LeadQualifier(),
        scheduler=lambda req: {
            "confirmed": True,
            "when": "2026-09-15T14:00:00Z",
            "summary": "Visita Torre Faria Lima",
            "location": "Av. Faria Lima, 3477",
        },
        properties_rag=lambda info: [{"title": f"Imóvel {i}"} for i in range(5)],
    )
    res = flow.invoke({
        "current_state": "scheduling",
        "message": "terça às 14h",
        "lead_info": {"intent": "rent", "region": "Faria Lima"},
        "shown_properties_count": 3,
    })
    assert res["current_state"] == "handoff"
    assert "ics_invite" in res
    ics = res["ics_invite"]
    assert "BEGIN:VCALENDAR" in ics
    assert "SUMMARY:Visita Torre Faria Lima" in ics
    assert "DTSTART:2026-09-15T14:00:00Z" in ics
    assert "END:VCALENDAR" in ics


def test_properties_catalog_faiss_search():
    lead = {"intent": "rent", "region": "Faria Lima", "budget": "R$ 50 mil", "area": "300 m²"}
    props = search_properties(lead, top_k=3)
    assert len(props) > 0
    assert len(props) <= 3
    # Confirma que campos do PRD §8.2 estão presentes
    p = props[0]
    for key in ("id", "title", "region", "area_util", "price", "mode"):
        assert key in p, f"campo {key} ausente no imóvel"


def test_clients_catalog_search():
    lead = {"region": "Faria Lima", "intent": "rent"}
    clients = search_clients(lead, top_k=3)
    assert len(clients) > 0
    assert len(clients) <= 3
    c = clients[0]
    for key in ("id", "name", "region", "intent", "status", "ticket"):
        assert key in c, f"campo {key} ausente no cliente"
