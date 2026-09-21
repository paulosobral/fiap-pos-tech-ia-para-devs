from unittest.mock import MagicMock

import pytest

import subprocess

from service import transcriber as transcriber_module
from service.transcriber import AudioConversionError, TranscriptionError, WhisperTranscriber


class Segment:
    def __init__(self, text: str) -> None:
        self.text = text


def make_model(segments=("olá tudo bem",)):
    model = MagicMock()
    model.transcribe.return_value = (iter([Segment(text) for text in segments]), MagicMock())
    return model


def make_transcriber(model):
    factory = MagicMock(return_value=model)
    return WhisperTranscriber(model_factory=factory), factory


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    calls = {}

    def fake_run(cmd, *args, **kwargs):
        calls["cmd"] = cmd
        return MagicMock(returncode=0)

    monkeypatch.setattr(transcriber_module.subprocess, "run", fake_run)
    return calls


class TestWhisperTranscriber:
    def test_transcribe_joins_segments_and_converts_via_ffmpeg(self, fake_ffmpeg):
        model = make_model()
        transcriber, factory = make_transcriber(model)
        text = transcriber.transcribe(b"audio-bytes")
        assert text == "olá tudo bem"
        factory.assert_called_once_with("small", device="cpu", compute_type="int8")
        assert fake_ffmpeg["cmd"][0] == "ffmpeg"
        assert fake_ffmpeg["cmd"][-1].endswith("output.wav")

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(subprocess.CalledProcessError(1, "ffmpeg"), id="ffmpeg-fail"),
            pytest.param(OSError("no ffmpeg binary"), id="ffmpeg-missing"),
        ],
    )
    def test_conversion_errors_raise_audio_conversion_error(self, error, monkeypatch):
        transcriber, _ = make_transcriber(make_model())
        monkeypatch.setattr(transcriber_module.subprocess, "run", MagicMock(side_effect=error))
        with pytest.raises(AudioConversionError):
            transcriber.transcribe(b"audio-bytes")

    def test_empty_audio_raises_conversion_error(self):
        transcriber, _ = make_transcriber(make_model())
        with pytest.raises(AudioConversionError):
            transcriber.transcribe(b"")

    def test_model_error_raises_transcription_error(self, fake_ffmpeg):
        model = make_model()
        model.transcribe.side_effect = RuntimeError("cuda oom")
        transcriber, _ = make_transcriber(model)
        with pytest.raises(TranscriptionError):
            transcriber.transcribe(b"audio-bytes")

    def test_missing_faster_whisper_raises(self, fake_ffmpeg, monkeypatch):
        monkeypatch.setattr(transcriber_module, "_HAS_FASTER_WHISPER", False)
        transcriber = WhisperTranscriber()
        with pytest.raises(TranscriptionError):
            transcriber.transcribe(b"audio-bytes")

    def test_real_model_class_used_when_available(self, fake_ffmpeg, monkeypatch):
        model = make_model()
        model_cls = MagicMock(return_value=model)
        monkeypatch.setattr(transcriber_module, "_HAS_FASTER_WHISPER", True)
        monkeypatch.setattr(transcriber_module, "_WhisperModel", model_cls)
        transcriber = WhisperTranscriber()
        assert transcriber.transcribe(b"audio-bytes") == "olá tudo bem"
        model_cls.assert_called_once_with("small", device="cpu", compute_type="int8")

    def test_model_loaded_once_across_calls(self, fake_ffmpeg):
        model = make_model()
        transcriber, factory = make_transcriber(model)
        transcriber.transcribe(b"a")
        transcriber.transcribe(b"b")
        assert factory.call_count == 1

    def test_language_pt_passed_to_model(self, fake_ffmpeg):
        model = make_model()
        transcriber, _ = make_transcriber(model)
        transcriber.transcribe(b"audio-bytes")
        assert model.transcribe.call_args.kwargs["language"] == "pt"

    def test_empty_transcript_returns_empty_string(self, fake_ffmpeg):
        transcriber, _ = make_transcriber(make_model(segments=()))
        assert transcriber.transcribe(b"audio-bytes") == ""
