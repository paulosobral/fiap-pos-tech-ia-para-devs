from __future__ import annotations

import logging
import os
from typing import Any

from infra.alert_store import AlertStore
from infra.conversation_store import ConversationStore
from service.anomaly_detector import AnomalyDetector, utc_now
from service.feature_extractor import ConversationFeatureExtractor
from service.scorer import HeuristicScorer, SklearnScorer

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def build_scorer() -> Any:
    name = os.environ.get("ANOMALY_SCORER", "heuristic").strip().lower()
    threshold = float(os.environ.get("ANOMALY_THRESHOLD", "0.7"))
    if name == "sklearn":
        return SklearnScorer(threshold=threshold)
    if name != "heuristic":
        raise ValueError(f"unknown ANOMALY_SCORER {name!r}: use 'heuristic' or 'sklearn'")
    return HeuristicScorer(threshold=threshold)


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    import boto3

    if not isinstance(event, dict):
        logger.warning("unexpected EventBridge payload; running scheduled job anyway")
    client = boto3.client("dynamodb")
    alerts = AlertStore(client, os.environ.get("ALERTS_TABLE", "sdr-alerts"))
    detector = AnomalyDetector(
        conversations=ConversationStore(
            client, os.environ.get("SESSIONS_TABLE", "sdr-sessions")
        ),
        alerts=alerts,
        extractor=ConversationFeatureExtractor(),
        scorer=build_scorer(),
        gate=build_gate(alerts),
        now_fn=utc_now,
    )
    return detector.run()


def build_gate(alerts: AlertStore) -> Any:
    from service.scheduler_gate import SchedulingGate

    return SchedulingGate(alerts)
