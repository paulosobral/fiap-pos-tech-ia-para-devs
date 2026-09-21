from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Log estruturado em JSON (NFR5.1) — formato único de log da Lambda U7.

    Usado por handler, stores (infra) e KpiService; nunca registre texto
    livre de mensagens — apenas contadores, identificadores e erros.
    """
    logger.log(level, json.dumps({"event": event, **fields}, default=str))
