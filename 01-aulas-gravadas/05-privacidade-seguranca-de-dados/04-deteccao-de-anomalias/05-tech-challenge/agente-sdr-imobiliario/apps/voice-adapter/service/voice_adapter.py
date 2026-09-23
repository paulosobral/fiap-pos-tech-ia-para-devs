from __future__ import annotations

import json
import logging
from typing import Any

from service.pii import PiiMasker
from service.router_gateway import RouterError, RouterGateway
from service.telegram_gateway import FALLBACK_MESSAGE, GatewayError, TelegramGateway
from service.transcriber import AudioConversionError, Transcriber, TranscriptionError

logger = logging.getLogger(__name__)

OUTCOME_OK = "ok"
OUTCOME_RETRY = "retry"
OUTCOME_DROP = "drop"

REQUIRED_FIELDS = ("message_id", "telegram_user_id", "voice_file_id", "session_id", "timestamp")


def log_event(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, default=str))


class VoiceAdapter:
    def __init__(
        self,
        telegram: TelegramGateway,
        transcriber: Transcriber,
        router: RouterGateway,
        sessions: Any | None = None,
        masker: PiiMasker | None = None,
    ) -> None:
        self.telegram = telegram
        self.transcriber = transcriber
        self.router = router
        self.sessions = sessions
        self.masker = masker or PiiMasker()

    def handle_event(self, event: dict[str, Any]) -> dict[str, Any]:
        failures: list[dict[str, str]] = []
        for record in event.get("Records", []):
            identifier = record.get("messageId", "")
            outcome = self._process_record(record)
            log_event("record_processed", message_id=identifier, outcome=outcome)
            if outcome == OUTCOME_RETRY:
                failures.append({"itemIdentifier": identifier})
        return {"batchItemFailures": failures}

    def _process_record(self, record: dict[str, Any]) -> str:
        try:
            message = json.loads(record.get("body") or "{}")
        except json.JSONDecodeError:
            log_event("invalid_body", body_preview=str(record.get("body"))[:120])
            return OUTCOME_DROP
        try:
            return self.process_message(message)
        except Exception as exc:
            logger.error("unexpected failure: %s", exc)
            return OUTCOME_RETRY

    def process_message(self, message: Any) -> str:
        error = self._validate(message)
        if error:
            log_event("message_rejected", reason=error)
            return OUTCOME_DROP
        session_id = message["session_id"]
        telegram_user_id = message["telegram_user_id"]
        if self.sessions is not None and self.sessions.get_session(session_id, telegram_user_id) is None:
            log_event("session_not_found", session_id=session_id)
            self._send_fallback(telegram_user_id)
            return OUTCOME_DROP
        try:
            file_path = self.telegram.get_file_path(message["voice_file_id"])
            audio_bytes = self.telegram.download(file_path)
        except GatewayError as exc:
            log_event("audio_unreachable", session_id=session_id, error=str(exc))
            self._send_fallback(telegram_user_id)
            return OUTCOME_DROP
        try:
            transcript = self.transcriber.transcribe(audio_bytes)
        except AudioConversionError as exc:
            log_event("audio_invalid", session_id=session_id, error=str(exc))
            self._send_fallback(telegram_user_id)
            return OUTCOME_DROP
        except TranscriptionError as exc:
            log_event("transcription_failed", session_id=session_id, error=str(exc))
            return OUTCOME_RETRY
        if not transcript.strip():
            log_event("empty_transcript", session_id=session_id)
            self._send_fallback(telegram_user_id)
            return OUTCOME_DROP
        masked = self.masker.mask(transcript)
        try:
            response_text = self.router.reinject(telegram_user_id, session_id, masked)
        except RouterError as exc:
            log_event("reinject_failed", session_id=session_id, error=str(exc))
            return OUTCOME_RETRY
        if response_text:
            self.telegram.send_message(telegram_user_id, response_text)
        log_event(
            "voice_transcribed",
            session_id=session_id,
            message_id=message["message_id"],
            chars=len(masked),
        )
        return OUTCOME_OK

    @staticmethod
    def _validate(message: Any) -> str | None:
        if not isinstance(message, dict):
            return "payload is not an object"
        for field in REQUIRED_FIELDS:
            if not message.get(field):
                return f"missing field {field}"
        if isinstance(message["telegram_user_id"], bool) or not isinstance(message["telegram_user_id"], int):
            return "telegram_user_id must be an integer"
        return None

    def _send_fallback(self, telegram_user_id: int) -> None:
        try:
            self.telegram.send_message(telegram_user_id, FALLBACK_MESSAGE)
        except Exception as exc:
            logger.error("fallback message failed: %s", exc)
