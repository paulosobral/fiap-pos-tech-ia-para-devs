from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def log_event(event: str, **fields: Any) -> None:
    """Log estruturado JSON no padrão NFR5.1 (estilo u2/u3/u4) — PII-safe por contrato."""
    logger.info(json.dumps({"event": event, **fields}, default=str))
