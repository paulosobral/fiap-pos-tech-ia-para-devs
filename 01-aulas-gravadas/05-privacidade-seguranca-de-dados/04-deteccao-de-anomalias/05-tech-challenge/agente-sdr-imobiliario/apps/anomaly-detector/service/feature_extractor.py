from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    from zoneinfo import ZoneInfo

    BUSINESS_TIMEZONE = ZoneInfo("America/Sao_Paulo")
except Exception:
    BUSINESS_TIMEZONE = timezone(timedelta(hours=-3), name="America/Sao_Paulo")

logger = logging.getLogger(__name__)

FEATURE_VOLUME = "message_volume"
FEATURE_LENGTH = "avg_message_length"
FEATURE_SENTIMENT = "negative_sentiment_ratio"
FEATURE_HOURS = "atypical_hour_ratio"

FEATURE_KEYS = (FEATURE_VOLUME, FEATURE_LENGTH, FEATURE_SENTIMENT, FEATURE_HOURS)

BUSINESS_HOUR_START = 8
BUSINESS_HOUR_END = 19

NEGATIVE_WORDS = (
    "reclama",
    "absurdo",
    "golpe",
    "péssimo",
    "horrível",
    "ódio",
    "idiota",
    "burro",
    "denúncia",
    "protesto",
    "engano",
    "mentira",
    "não quero",
    "nunca mais",
)

_TEXT_FIELDS = ("text", "content", "message")
_TIME_FIELDS = ("at", "ts", "timestamp", "created_at", "sent_at")


def _message_text(message: dict[str, Any]) -> str:
    for field in _TEXT_FIELDS:
        value = message.get(field)
        if isinstance(value, str):
            return value
    return ""


def _message_hour(message: dict[str, Any]) -> int | None:
    """Hora local do lead: lê o campo real `at` da U1 (handler.py grava
    {"role", "text", "at"} com ISO UTC) e converte para America/Sao_Paulo.
    `ts` e demais variantes ficam como compatibilidade; timestamps naive são
    interpretados como UTC antes da conversão.
    """
    for field in _TIME_FIELDS:
        raw = message.get(field)
        if isinstance(raw, str) and raw:
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(BUSINESS_TIMEZONE).hour
    return None


class ConversationFeatureExtractor:
    """Extrator de features por conversa (FR9.1), 100% determinístico.

    Volume de mensagens, comprimento médio, sentimento heurístico (palavras
    negativas em pt-BR) e razão de horários atípicos (fora da janela útil
    08:00–18:59 em America/Sao_Paulo, fuso dos leads +55).
    Função pura: mesmas entradas → mesmas features, sem clock interno — o
    clock injetável do job diário vive no orquestrador (`AnomalyDetector`).
    """

    def extract(self, conversation: dict[str, Any]) -> dict[str, float | int]:
        messages = conversation.get("messages")
        if not isinstance(messages, list):
            messages = []
        texts = [_message_text(m) for m in messages if isinstance(m, dict)]
        texts = [t for t in texts if t]
        hours = [_message_hour(m) for m in messages if isinstance(m, dict)]
        hours = [h for h in hours if h is not None]

        volume = len(messages)
        avg_length = round(sum(len(t) for t in texts) / len(texts), 4) if texts else 0.0
        negative = sum(1 for t in texts if self._is_negative(t))
        negative_ratio = round(negative / len(texts), 4) if texts else 0.0
        atypical = sum(1 for h in hours if not (BUSINESS_HOUR_START <= h < BUSINESS_HOUR_END))
        atypical_ratio = round(atypical / len(hours), 4) if hours else 0.0

        return {
            FEATURE_VOLUME: volume,
            FEATURE_LENGTH: avg_length,
            FEATURE_SENTIMENT: negative_ratio,
            FEATURE_HOURS: atypical_ratio,
        }

    @staticmethod
    def _is_negative(text: str) -> bool:
        lowered = text.lower()
        return any(word in lowered for word in NEGATIVE_WORDS)
