import sys
import types

import pytest

import handler


def test_llm_key_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_API_SECRET_ID", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    assert handler._llm_api_key() == "env-key"


def test_llm_key_from_secret_manager(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_SECRET_ID", "arn:aws:secretsmanager:us-east-1::secret/sdr/llm-api-key")
    sm = types.SimpleNamespace(get_secret_value=lambda SecretId: {"SecretString": "sk-or-123"})
    fake = types.ModuleType("boto3")
    fake.client = lambda service: sm
    monkeypatch.setitem(sys.modules, "boto3", fake)
    assert handler._llm_api_key() == "sk-or-123"


def test_llm_key_absent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_SECRET_ID", raising=False)
    assert handler._llm_api_key() is None


def test_llm_key_secret_failure_returns_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_SECRET_ID", "arn:aws:secretsmanager:us-east-1::secret/sdr/llm-api-key")
    fake = types.ModuleType("boto3")

    def boom(service):
        raise RuntimeError("GetSecretValue indisponível")

    fake.client = boom
    monkeypatch.setitem(sys.modules, "boto3", fake)
    assert handler._llm_api_key() is None
