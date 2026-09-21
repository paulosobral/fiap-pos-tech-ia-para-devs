import json
import sys
from unittest.mock import MagicMock

import pytest

from handler import handler
from service.router_gateway import RouterError
from service.telegram_gateway import FALLBACK_MESSAGE, GatewayError
from service.voice_adapter import VoiceAdapter


def sqs_record(message_id, body):
    return {"messageId": message_id, "body": body}


def voice_message(**overrides):
    base = {
        "message_id": "m1",
        "telegram_user_id": 42,
        "voice_file_id": "f1",
        "session_id": "s1",
        "timestamp": "2026-09-20T00:00:00+00:00",
    }
    base.update(overrides)
    return base


def make_adapter(transcript="Quero um espaço para 20 pessoas"):
    telegram = MagicMock()
    telegram.get_file_path.return_value = "voice/f1.ogg"
    telegram.download.return_value = b"ogg-bytes"
    transcriber = MagicMock()
    transcriber.transcribe.return_value = transcript
    router = MagicMock()
    store = MagicMock()
    store.get_session.return_value = {"session_id": "s1"}
    adapter = VoiceAdapter(telegram=telegram, transcriber=transcriber, router=router, sessions=store)
    return adapter, telegram, transcriber, router, store


class TestVoiceAdapterPipeline:
    def test_full_pipeline_no_failures(self):
        adapter, telegram, transcriber, router, store = make_adapter()
        event = {"Records": [sqs_record("r1", json.dumps(voice_message()))]}
        result = adapter.handle_event(event)
        assert result == {"batchItemFailures": []}
        transcriber.transcribe.assert_called_once_with(b"ogg-bytes")
        router.reinject.assert_called_once_with(42, "s1", "Quero um espaço para 20 pessoas")

    def test_transcript_masked_end_to_end(self):
        adapter, *_ = make_adapter(transcript="Meu e-mail é joao@empresa.com")
        adapter.handle_event({"Records": [sqs_record("r1", json.dumps(voice_message()))]})
        sent = adapter.router.reinject.call_args[0][2]
        assert "[EMAIL]" in sent
        assert "joao@empresa.com" not in sent

    def test_mixed_batch_drops_invalid_without_failure(self):
        adapter, *_ = make_adapter()
        records = [
            sqs_record("r1", json.dumps(voice_message())),
            sqs_record("r2", json.dumps({"message_id": "m2"})),
        ]
        result = adapter.handle_event({"Records": records})
        assert result == {"batchItemFailures": []}

    def test_retry_record_listed_in_failures(self):
        adapter, *_ = make_adapter()
        adapter.router.reinject.side_effect = RouterError("router down")
        result = adapter.handle_event({"Records": [sqs_record("r1", json.dumps(voice_message()))]})
        assert result == {"batchItemFailures": [{"itemIdentifier": "r1"}]}

    def test_malformed_body_dropped_explicitly(self):
        adapter, telegram, _, router, _ = make_adapter()
        result = adapter.handle_event({"Records": [sqs_record("r1", "not-json")]})
        assert result == {"batchItemFailures": []}
        telegram.send_message.assert_not_called()
        router.reinject.assert_not_called()

    def test_unexpected_exception_becomes_retry(self):
        adapter, *_ = make_adapter()
        adapter.sessions.get_session.side_effect = RuntimeError("dynamo down")
        result = adapter.handle_event({"Records": [sqs_record("r1", json.dumps(voice_message()))]})
        assert result == {"batchItemFailures": [{"itemIdentifier": "r1"}]}

    def test_unreachable_audio_sends_fallback_in_batch(self):
        adapter, telegram, transcriber, router, _ = make_adapter()
        adapter.telegram.get_file_path.side_effect = GatewayError("file not found")
        result = adapter.handle_event({"Records": [sqs_record("r1", json.dumps(voice_message()))]})
        assert result == {"batchItemFailures": []}
        telegram.send_message.assert_called_once_with(42, FALLBACK_MESSAGE)
        transcriber.transcribe.assert_not_called()

    def test_empty_records_returns_no_failures(self):
        adapter, *_ = make_adapter()
        assert adapter.handle_event({}) == {"batchItemFailures": []}


class TestHandlerWiring:
    def test_handler_builds_adapter_and_processes_event(self, monkeypatch):
        boto3 = MagicMock()
        requests = MagicMock()
        monkeypatch.setitem(sys.modules, "boto3", boto3)
        monkeypatch.setitem(sys.modules, "requests", requests)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("ROUTER_BASE_URL", "http://router.local")
        monkeypatch.setenv("INTERNAL_SECRET_TOKEN", "sec")
        event = {"Records": [sqs_record("r1", json.dumps({"message_id": "m1"}))]}
        result = handler(event)
        assert result == {"batchItemFailures": []}
        boto3.client.assert_called_once_with("dynamodb")

    def test_handler_missing_env_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "boto3", MagicMock())
        monkeypatch.setitem(sys.modules, "requests", MagicMock())
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.setenv("ROUTER_BASE_URL", "http://router.local")
        monkeypatch.setenv("INTERNAL_SECRET_TOKEN", "sec")
        with pytest.raises(RuntimeError):
            handler({"Records": []})

    def test_handler_missing_internal_secret_env_raises(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "boto3", MagicMock())
        monkeypatch.setitem(sys.modules, "requests", MagicMock())
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("ROUTER_BASE_URL", "http://router.local")
        monkeypatch.delenv("INTERNAL_SECRET_TOKEN", raising=False)
        with pytest.raises(RuntimeError):
            handler({"Records": []})


class TestHandlerTranscriberSingleton:
    def test_transcriber_created_once_across_warm_invocations(self, monkeypatch):
        import handler as handler_module

        created = []

        class CountingTranscriber:
            def __init__(self) -> None:
                created.append(self)

        monkeypatch.setattr(handler_module, "WhisperTranscriber", CountingTranscriber)
        monkeypatch.setattr(handler_module, "_transcriber", None)
        monkeypatch.setitem(sys.modules, "boto3", MagicMock())
        monkeypatch.setitem(sys.modules, "requests", MagicMock())
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("ROUTER_BASE_URL", "http://router.local")
        monkeypatch.setenv("INTERNAL_SECRET_TOKEN", "sec")
        handler_module.handler({"Records": []})
        handler_module.handler({"Records": []})
        assert len(created) == 1

    def test_get_transcriber_returns_same_instance(self, monkeypatch):
        import handler as handler_module

        monkeypatch.setattr(handler_module, "_transcriber", None)
        first = handler_module.get_transcriber()
        second = handler_module.get_transcriber()
        assert first is second
