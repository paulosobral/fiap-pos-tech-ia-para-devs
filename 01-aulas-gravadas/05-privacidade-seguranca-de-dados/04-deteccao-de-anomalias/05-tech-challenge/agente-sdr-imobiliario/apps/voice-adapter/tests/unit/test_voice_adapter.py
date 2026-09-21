from unittest.mock import MagicMock

from service.router_gateway import RouterError
from service.telegram_gateway import FALLBACK_MESSAGE, GatewayError
from service.transcriber import AudioConversionError, TranscriptionError
from service.voice_adapter import VoiceAdapter


def message(**overrides):
    base = {
        "message_id": "m1",
        "telegram_user_id": 42,
        "voice_file_id": "f1",
        "session_id": "s1",
        "timestamp": "2026-09-20T00:00:00+00:00",
    }
    base.update(overrides)
    return base


def make_adapter(transcript="Quero um espaço para 20 pessoas", sessions=True):
    telegram = MagicMock()
    telegram.get_file_path.return_value = "voice/f1.ogg"
    telegram.download.return_value = b"ogg-bytes"
    transcriber = MagicMock()
    transcriber.transcribe.return_value = transcript
    router = MagicMock()
    store = MagicMock()
    store.get_session.return_value = {"session_id": "s1"} if sessions else None
    adapter = VoiceAdapter(telegram=telegram, transcriber=transcriber, router=router, sessions=store)
    return adapter, telegram, transcriber, router, store


class TestVoiceAdapter:
    def test_happy_path_reinjects_transcript(self):
        adapter, telegram, transcriber, router, store = make_adapter()
        outcome = adapter.process_message(message())
        assert outcome == "ok"
        transcriber.transcribe.assert_called_once_with(b"ogg-bytes")
        router.reinject.assert_called_once_with(42, "s1", "Quero um espaço para 20 pessoas")
        telegram.send_message.assert_not_called()

    def test_invalid_message_is_dropped(self):
        adapter, telegram, _, router, _ = make_adapter()
        assert adapter.process_message(message(voice_file_id="")) == "drop"
        router.reinject.assert_not_called()
        telegram.get_file_path.assert_not_called()

    def test_non_integer_user_id_is_dropped(self):
        adapter, _, _, router, _ = make_adapter()
        assert adapter.process_message(message(telegram_user_id="42")) == "drop"
        router.reinject.assert_not_called()

    def test_non_dict_message_is_dropped(self):
        adapter, _, _, router, _ = make_adapter()
        assert adapter.process_message("not a dict") == "drop"
        router.reinject.assert_not_called()

    def test_empty_transcript_sends_fallback(self):
        adapter, telegram, _, router, _ = make_adapter(transcript="   ")
        assert adapter.process_message(message()) == "drop"
        telegram.send_message.assert_called_once_with(42, FALLBACK_MESSAGE)
        router.reinject.assert_not_called()

    def test_fallback_send_failure_does_not_crash(self):
        adapter, telegram, _, router, _ = make_adapter(sessions=False)
        telegram.send_message.side_effect = RuntimeError("telegram down")
        assert adapter.process_message(message()) == "drop"
        router.reinject.assert_not_called()

    def test_missing_session_sends_fallback(self):
        adapter, telegram, _, router, _ = make_adapter(sessions=False)
        assert adapter.process_message(message()) == "drop"
        telegram.send_message.assert_called_once_with(42, FALLBACK_MESSAGE)
        router.reinject.assert_not_called()

    def test_unreachable_audio_sends_fallback(self):
        adapter, telegram, transcriber, router, _ = make_adapter()
        adapter.telegram.get_file_path.side_effect = GatewayError("file not found")
        assert adapter.process_message(message()) == "drop"
        telegram.send_message.assert_called_once_with(42, FALLBACK_MESSAGE)
        transcriber.transcribe.assert_not_called()

    def test_conversion_failure_sends_fallback(self):
        adapter, telegram, transcriber, router, _ = make_adapter()
        transcriber.transcribe.side_effect = AudioConversionError("ffmpeg failed")
        assert adapter.process_message(message()) == "drop"
        telegram.send_message.assert_called_once_with(42, FALLBACK_MESSAGE)
        router.reinject.assert_not_called()

    def test_model_failure_is_retryable_without_fallback(self):
        adapter, telegram, transcriber, router, _ = make_adapter()
        transcriber.transcribe.side_effect = TranscriptionError("model oom")
        assert adapter.process_message(message()) == "retry"
        telegram.send_message.assert_not_called()

    def test_reinject_failure_is_retryable(self):
        adapter, telegram, _, router, _ = make_adapter()
        router.reinject.side_effect = RouterError("router down")
        assert adapter.process_message(message()) == "retry"
        telegram.send_message.assert_not_called()

    def test_transcript_masked_before_reinject(self):
        adapter, telegram, transcriber, router, _ = make_adapter(
            transcript="Meu e-mail é joao@empresa.com"
        )
        adapter.process_message(message())
        sent = router.reinject.call_args[0][2]
        assert "joao@empresa.com" not in sent
        assert "[EMAIL]" in sent
