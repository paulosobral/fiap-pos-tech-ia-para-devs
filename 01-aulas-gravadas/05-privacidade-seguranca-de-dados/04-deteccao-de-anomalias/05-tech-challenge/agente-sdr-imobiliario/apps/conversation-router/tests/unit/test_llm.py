from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from service import llm as lm


class FakeResp:
    def __init__(self, body: str) -> None:
        self._body = body

    def __enter__(self) -> "FakeResp":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body.encode()


def _completion(content: str) -> str:
    return json.dumps({"choices": [{"message": {"content": content}}]})


def test_classify_intent_success(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_urlopen(req: object, timeout: float) -> FakeResp:
        captured["url"] = req.full_url  # type: ignore[attr-defined]
        captured["auth"] = req.get_header("Authorization")  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        return FakeResp(_completion('{"intent": "rent", "confidence": 0.92}'))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    intent, confidence = lm.classify_intent("você tem algo para alugar?", "key-123", model="m/1")
    assert intent == "rent"
    assert confidence == 0.92
    assert captured["url"] == lm.OPENROUTER_URL
    assert captured["auth"] == "Bearer key-123"
    assert captured["timeout"] == 8


def test_classify_intent_env_model_and_timeout(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLM_MODEL", "env/model")
    monkeypatch.setenv("LLM_TIMEOUT", "3")

    def fake_urlopen(req: object, timeout: float) -> FakeResp:
        body = json.loads(req.data.decode())  # type: ignore[attr-defined]
        assert body["model"] == "env/model"
        return FakeResp(_completion('{"intent": "purchase", "confidence": 0.8}'))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    intent, confidence = lm.classify_intent("quero comprar", "k")
    assert intent == "purchase"
    assert confidence == 0.8


def test_classify_intent_invalid_content_raises(monkeypatch: pytest.MonkeyPatch):
    def fake_urlopen(req: object, timeout: float) -> FakeResp:
        return FakeResp(_completion("não é json"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ValueError):
        lm.classify_intent("oi", "k")


def test_classify_intent_unknown_intent_raises(monkeypatch: pytest.MonkeyPatch):
    def fake_urlopen(req: object, timeout: float) -> FakeResp:
        return FakeResp(_completion('{"intent": "signup", "confidence": 0.9}'))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ValueError):
        lm.classify_intent("cadastro", "k")


def test_classify_intent_http_error_propagates(monkeypatch: pytest.MonkeyPatch):
    def boom(req: object, timeout: float) -> FakeResp:
        raise urllib.error.HTTPError("u", 401, "unauthorized", [], None)

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(urllib.error.HTTPError):
        lm.classify_intent("oi", "k")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"intent": "investment", "confidence": 0.5}', ("investment", 0.5)),
        ('{"intent": "unknown", "confidence": 0.2}', ("unknown", 0.2)),
        ('{"intent": "rent", "confidence": 1.5}', ("rent", 1.0)),
        ('{"intent": "rent", "confidence": -0.5}', ("rent", 0.0)),
        ('{"intent": "rent", "confidence": "x"}', (None, None)),
        ("", (None, None)),
        ("{invalid", (None, None)),
        ('{"intent": "purchase"}', ("purchase", 0.0)),
    ],
)
def test_parse(raw: str, expected: tuple[object, object]):
    assert lm._parse(raw) == expected


def test_generate_reply_success(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_urlopen(req: object, timeout: float) -> FakeResp:
        captured["body"] = json.loads(req.data.decode())  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        return FakeResp(_completion("Ótima escolha! Podemos agendar uma visita?"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    out = lm.generate_reply(
        message="quero ver opções na Faria Lima",
        canned_response="Encontramos estas opções:\n1. Torre Office One — Faria Lima, 500 m²\nGostaria de agendar?",
        lead_info={"intent": "rent", "region": "Faria Lima", "area": "100 m²"},
        properties=[{"title": "Torre Office One", "area_util": 500, "price": 10000}],
        api_key="key",
        model="m/1",
    )
    assert out == "Ótima escolha! Podemos agendar uma visita?"
    assert captured["timeout"] == 8
    body = captured["body"]
    assert body["model"] == "m/1"
    assert body["messages"][0]["role"] == "system"
    sys_prompt = body["messages"][0]["content"]
    assert "nunca invente" in sys_prompt
    assert "verificar com a equipe" in sys_prompt
    user_prompt = body["messages"][1]["content"]
    assert "RESPOSTA OFICIAL" in user_prompt
    assert "Torre Office One" in user_prompt


def test_generate_reply_empty_raises(monkeypatch: pytest.MonkeyPatch):
    def fake_urlopen(req: object, timeout: float) -> FakeResp:
        return FakeResp(_completion("   "))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ValueError):
        lm.generate_reply("oi", "resposta", {}, [], api_key="k")


def test_generate_reply_http_error_propagates(monkeypatch: pytest.MonkeyPatch):
    def boom(req: object, timeout: float) -> FakeResp:
        raise urllib.error.HTTPError("u", 500, "erro", [], None)

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(urllib.error.HTTPError):
        lm.generate_reply("oi", "resposta", {}, [], api_key="k")