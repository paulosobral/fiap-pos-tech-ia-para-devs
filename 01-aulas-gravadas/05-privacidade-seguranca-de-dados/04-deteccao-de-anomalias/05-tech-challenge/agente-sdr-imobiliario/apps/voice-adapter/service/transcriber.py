from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from typing import Any, Protocol

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel as _WhisperModel

    _HAS_FASTER_WHISPER = True
except ImportError:
    _WhisperModel = None
    _HAS_FASTER_WHISPER = False


class Transcriber(Protocol):
    def transcribe(self, audio_bytes: bytes) -> str: ...


class TranscriptionError(Exception):
    pass


class AudioConversionError(TranscriptionError):
    pass


class WhisperTranscriber:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "pt",
        model_factory: Any | None = None,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._model_factory = model_factory
        self._model: Any | None = None

    def transcribe(self, audio_bytes: bytes) -> str:
        if not audio_bytes:
            raise AudioConversionError("empty audio payload")
        with tempfile.TemporaryDirectory() as workdir:
            wav_path = self._convert_to_wav(audio_bytes, workdir)
            model = self._load_model()
            try:
                segments, _info = model.transcribe(wav_path, language=self._language)
                text = " ".join(segment.text for segment in segments).strip()
            except Exception as exc:
                logger.error("whisper transcription failed: %s", exc)
                raise TranscriptionError("whisper transcription failed") from exc
        logger.info(json.dumps({"event": "transcription_finished", "chars": len(text)}))
        return text

    def _convert_to_wav(self, audio_bytes: bytes, workdir: str) -> str:
        source = os.path.join(workdir, "input.audio")
        target = os.path.join(workdir, "output.wav")
        try:
            with open(source, "wb") as fh:
                fh.write(audio_bytes)
            subprocess.run(
                ["ffmpeg", "-y", "-i", source, "-ar", "16000", "-ac", "1", target],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.error("ffmpeg conversion failed: %s", exc.stderr)
            raise AudioConversionError("ffmpeg conversion failed") from exc
        except OSError as exc:
            logger.error("ffmpeg unavailable: %s", exc)
            raise AudioConversionError("ffmpeg unavailable") from exc
        return target

    def _load_model(self) -> Any:
        if self._model is None:
            if self._model_factory is not None:
                self._model = self._model_factory(
                    self._model_size, device=self._device, compute_type=self._compute_type
                )
            elif _HAS_FASTER_WHISPER:
                self._model = _WhisperModel(
                    self._model_size, device=self._device, compute_type=self._compute_type
                )
            else:
                raise TranscriptionError("faster-whisper not installed; provide model_factory")
        return self._model
