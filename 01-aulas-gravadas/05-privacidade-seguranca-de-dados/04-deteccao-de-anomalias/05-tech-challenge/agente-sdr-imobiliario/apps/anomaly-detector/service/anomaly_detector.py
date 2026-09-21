from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable

from infra.alert_store import ACTION_ALERT_ISSUED, ALERT_STATUS_OPEN
from infra.conversation_store import ConversationStoreError
from service.feature_extractor import FEATURE_KEYS, ConversationFeatureExtractor
from service.scorer import AnomalyScorer, ScoringResult

logger = logging.getLogger(__name__)


def log_event(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, default=str))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AnomalyDetector:
    """Job diário de detecção de anomalias (U5, FR9).

    Fluxo: ler conversas (Contrato 6) → extrair features por conversa
    (FR9.1) → score (FR9.2) → salvar alerta (Contrato 7, FR9.3) → restringir
    agendamento do lead suspeito (FR9.4). `anomaly_id` determinístico
    (`session_id#data-do-job`) torna o reprocesso do mesmo dia idempotente
    (put_item sobrescreve). Falha de leitura/scoring propaga (retry async da
    Lambda); falha pontual de extração é contada e não derruba o job.
    """

    def __init__(
        self,
        conversations: Any,
        alerts: Any,
        scorer: AnomalyScorer,
        extractor: ConversationFeatureExtractor | None = None,
        gate: Any | None = None,
        now_fn: Callable[[], datetime] = utc_now,
    ) -> None:
        self.conversations = conversations
        self.alerts = alerts
        self.scorer = scorer
        self.extractor = extractor or ConversationFeatureExtractor()
        self.gate = gate
        self.now_fn = now_fn

    def run(self) -> dict[str, Any]:
        log_event("job_started")
        try:
            conversations = self.conversations.list_conversations()
        except ConversationStoreError:
            log_event("conversation_read_failed")
            raise
        rows: list[dict[str, Any]] = []
        errors = 0
        for conversation in conversations:
            lead_id = conversation.get("lead_id")
            try:
                features = self.extractor.extract(conversation)
                rows.append({"conversation": conversation, "features": features})
            except Exception as exc:
                errors += 1
                log_event("feature_extraction_failed", lead_id=lead_id, error=str(exc))
        results = self.scorer.score_many([row["features"] for row in rows])
        anomalies = 0
        restricted = 0
        for row, result in zip(rows, results):
            if not result.is_anomaly:
                continue
            item = self._alert_item(row, result)
            self.alerts.save_anomaly(item)
            anomalies += 1
            if self.gate is not None:
                self.gate.restrict(str(row["conversation"]["lead_id"]), item["anomaly_id"])
                restricted += 1
            log_event(
                "anomaly_alerted",
                anomaly_id=item["anomaly_id"],
                lead_id=item["lead_id"],
                type=item["type"],
                confidence=item["confidence"],
            )
        summary = {
            "conversations": len(conversations),
            "scored": len(rows),
            "anomalies": anomalies,
            "restricted": restricted,
            "errors": errors,
        }
        log_event("job_completed", **summary)
        return summary

    def _alert_item(self, row: dict[str, Any], result: ScoringResult) -> dict[str, Any]:
        conversation = row["conversation"]
        now = self.now_fn()
        session_id = str(conversation.get("session_id") or conversation.get("lead_id"))
        return {
            "anomaly_id": f"{session_id}#{now.strftime('%Y-%m-%d')}",
            "lead_id": conversation["lead_id"],
            "features": {key: row["features"].get(key) for key in FEATURE_KEYS},
            "confidence": result.score,
            "type": result.kind,
            "detected_at": now.isoformat(),
            "status": ALERT_STATUS_OPEN,
            "action_taken": ACTION_ALERT_ISSUED,
        }
