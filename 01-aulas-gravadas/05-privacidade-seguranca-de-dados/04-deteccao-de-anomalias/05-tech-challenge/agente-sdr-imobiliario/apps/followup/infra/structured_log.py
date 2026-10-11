from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def log_event(event: str, **fields: Any) -> None:
    """Log estruturado JSON (NFR5.1) — único formato de log da unidade.

    Campos nomeados por evento; nunca carrega texto de conversa ou contato
    (PII-safe por construção).
    """
    logger.info(json.dumps({"event": event, **fields}, default=str))
