"""Gate de qualidade (LLM real) da detecção de anomalias: chat → detector → dashboard (FR9 → FR7.4).

Roda o chat de verdade (`ConversationRouter` + `SalesFlow` com a LLM real, a mesma fiação de produção),
grava as conversas no DynamoDB falso e entrega o que foi gravado ao código REAL do `anomaly-detector`
(Lambda handler) e, depois, ao `dashboard-api` (GET /api/kpis). Cada app roda em subprocesso próprio,
porque os três têm módulos de mesmo nome (`handler`, `service`, `infra`).

Só roda com chave de LLM no ambiente (como o resto desta pasta). Limite de teste = 0,4 (ver ADR-019):
com o 0,7 de produção seriam ~20 mensagens de milhares de caracteres, inviável de digitar na demo.
Números medidos: a nota soma volume (0,3), tamanho médio (0,2), negatividade (0,3) e horário (0,2); mensagens do bot
entram na média.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import handler as handler_module
from service import llm
from tests.integration.fixtures import FakeDynamo, telegram_update as update

API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY") or ""

pytestmark = pytest.mark.skipif(
    not API_KEY,
    reason="Teste de qualidade com LLM real — requer OPENROUTER_API_KEY/LLM_API_KEY no ambiente.",
)

APPS = Path(__file__).resolve().parents[3]  # .../apps
NIGHT_UTC = "2026-10-09T01:30:00+00:00"  # 22:30 em Brasília (fora do horário comercial 08–19h)
DAY_UTC = "2026-10-09T17:00:00+00:00"  # 14:00 em Brasília
NEGATIVE = "isso é um golpe absurdo, péssimo atendimento, que mentira e que engano, nunca mais volto aqui. "
ALERT_TYPES = {"high_volume", "long_messages", "negative_sentiment", "atypical_hours", "composite", "ml_ensemble"}


def _build_router(dynamo: FakeDynamo):
    from handler import ConversationRouter
    from infra.session_store import SessionStore
    from service.flow.lead_qualifier import LeadQualifier
    from service.flow.sales_flow import SalesFlow
    from service.properties_catalog import known_places, search_properties
    from service.security_layer import SecurityLayer

    def route(message, lead_info, current_state, **kw):
        return llm.extract_and_route(
            message, lead_info, current_state, api_key=API_KEY,
            shown_properties=kw.get("shown_properties"), favorite_property=kw.get("favorite_property"),
            conversation_history=kw.get("conversation_history"), photos_sent=kw.get("photos_sent"),
            places=known_places(),
        )

    def reply(message, canned, lead_info, properties, **kw):
        return llm.generate_reply(message, canned, lead_info, properties, api_key=API_KEY, **kw)

    flow = SalesFlow(lead_qualifier=LeadQualifier(), llm_router=route, reply_generator=reply,
                     properties_rag=lambda info: search_properties(info, top_k=9))
    return ConversationRouter(
        store=SessionStore(dynamo, "t"), security_layer=SecurityLayer(pii_store=None), sales_flow=flow,
        sqs_client=MagicMock(), telegram_client=MagicMock(), voice_queue_url="https://sqs/voice",
        crm_queue_url="https://sqs/crm", secret_token="tok", pii_store=None, internal_secret_token="internal-tok",
    )


def _chat(monkeypatch, at_utc: str, texts: list[str]) -> list[dict]:
    """Conversa de verdade (LLM real) com relógio fixo; devolve os itens crus gravados no DynamoDB."""
    monkeypatch.setattr(handler_module, "utc_now_iso", lambda: at_utc)
    dynamo = FakeDynamo()
    router = _build_router(dynamo)
    for index, text in enumerate(["oi", "sim", *texts], start=1):
        assert router.handle(update(text, update_id=index))["statusCode"] == 200
    return list(dynamo.items.values())


def _run_in(app: str, code: str, payload: dict) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=APPS / app, input=json.dumps(payload),
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, f"{app}: {result.stderr[-1500:]}"
    return json.loads(result.stdout.strip().splitlines()[-1])


_DETECTOR = """
import json, os, sys, types
payload = json.load(sys.stdin)
os.environ.update(SESSIONS_TABLE="t", ALERTS_TABLE="a", ANOMALY_SCORER="heuristic", ANOMALY_THRESHOLD=str(payload["threshold"]))
from tests.integration.test_anomaly_pipeline import FakeDynamo
import handler as h
fake = FakeDynamo(payload["items"])
boto = types.ModuleType("boto3"); boto.client = lambda name: fake; sys.modules["boto3"] = boto
summary = h.handler({"source": "aws.events", "detail-type": "Scheduled Event"})
alerts = [v for v in fake._items.values() if "anomaly_id" in v]
print(json.dumps({"summary": summary, "alerts": alerts}))
"""

_DASHBOARD = """
import json, os, sys
payload = json.load(sys.stdin)
os.environ.update(SESSIONS_TABLE="sdr-sessions", ALERTS_TABLE="sdr-alerts", CW_NAMESPACE="SdrApp")
from tests.unit.test_handler import FakeAwsClient, FakeBoto3Module, get_event
client = FakeAwsClient(pages_by_table={
    "sdr-sessions": [{"Items": payload["items"]}],
    "sdr-alerts": [{"Items": payload["alerts"]}],
})
sys.modules["boto3"] = FakeBoto3Module({"dynamodb": client, "cloudwatch": client})
import handler as h
response = h.handler(get_event())
print(json.dumps({"status": response["statusCode"], "body": json.loads(response["body"])}))
"""


def _is_true(value) -> bool:
    return value is True or (isinstance(value, dict) and value.get("BOOL") is True)


def test_negative_long_night_chat_raises_alert_that_reaches_the_dashboard(monkeypatch):
    # 4 mensagens de 3500 caracteres (limite do Telegram: 4096). As duas do início ("oi", "sim") e as
    # respostas curtas do bot diluem a média, então menos que isso não passa de ~0,3.
    items = _chat(monkeypatch, NIGHT_UTC, [(NEGATIVE * 60)[:3500] for _ in range(4)])

    detected = _run_in("anomaly-detector", _DETECTOR, {"items": items, "threshold": 0.4})
    summary = detected["summary"]
    assert summary["scored"] >= 1 and summary["errors"] == 0, summary
    assert summary["anomalies"] >= 1, detected
    assert summary["restricted"] >= 1, summary  # FR9.4: agendamento do lead suspeito é bloqueado
    alert = detected["alerts"][0]
    assert alert["type"]["S"] in ALERT_TYPES, alert
    assert alert["status"]["S"] == "open"

    dash = _run_in("dashboard-api", _DASHBOARD, {"items": items, "alerts": detected["alerts"]})
    assert dash["status"] == 200, dash
    body = dash["body"]
    assert body["anomalies_count"] >= 1, body
    shown = body["alerts"][0]
    assert shown["lead_id"] == alert["lead_id"]["S"] and shown["type"] == alert["type"]["S"], shown
    assert "features" not in shown  # projeção sem PII/payload bruto (FR7.4)


def test_normal_daytime_chat_is_not_flagged(monkeypatch):
    items = _chat(monkeypatch, DAY_UTC, ["quero comprar uma sala comercial em santo andré", "tem foto?"])

    detected = _run_in("anomaly-detector", _DETECTOR, {"items": items, "threshold": 0.4})
    assert detected["summary"]["scored"] >= 1, detected
    assert detected["summary"]["anomalies"] == 0, detected  # sem falso positivo em conversa normal
    assert detected["alerts"] == []
