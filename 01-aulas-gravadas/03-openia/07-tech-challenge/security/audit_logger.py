"""
audit_logger.py
===============
Structured JSON audit logger for the medical assistant.
All LLM interactions, agent steps, and high-urgency alerts are recorded.

Log file: data/logs/audit.jsonl (one JSON object per line)
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "audit.jsonl"


# ── Handler customizado de log JSON ──────────────────────────────────────────

class _JsonFileHandler(logging.Handler):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path

    def emit(self, record: logging.LogRecord) -> None:
        entry: dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", record.getMessage()),
        }
        # Mescla quaisquer campos extras anexados ao record.
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            ) and not key.startswith("_"):
                entry[key] = value

        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            self.handleError(record)


def get_audit_logger() -> logging.Logger:
    logger = logging.getLogger("medical_assistant.audit")
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(_JsonFileHandler(LOG_FILE))
        logger.propagate = False
    return logger


# ── Helpers de alto nível ─────────────────────────────────────────────────────

_audit = get_audit_logger()


def log_query(
    *,
    patient_id: int,
    query: str,
    response: str,
    sources: list[str],
    agent_steps: list[str],
    latency_ms: float,
    urgency_level: str = "",
) -> None:
    _audit.info(
        "llm_query",
        extra={
            "event": "llm_query",
            "patient_id": patient_id,
            "query": query[:500],
            "response_preview": response[:300],
            "sources": sources,
            "agent_steps": agent_steps,
            "urgency_level": urgency_level,
            "latency_ms": round(latency_ms, 2),
        },
    )


def log_agent_step(agent_name: str, patient_id: int, steps: list[str]) -> None:
    _audit.debug(
        "agent_step",
        extra={
            "event": "agent_step",
            "agent": agent_name,
            "patient_id": patient_id,
            "steps": steps,
        },
    )


def load_recent_logs(n: int = 50) -> list[dict[str, Any]]:
    """Read the last `n` log entries from the JSONL file."""
    if not LOG_FILE.exists():
        return []
    lines: list[str] = []
    try:
        with LOG_FILE.open(encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    recent = lines[-n:] if len(lines) > n else lines
    entries: list[dict] = []
    for line in reversed(recent):
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries
