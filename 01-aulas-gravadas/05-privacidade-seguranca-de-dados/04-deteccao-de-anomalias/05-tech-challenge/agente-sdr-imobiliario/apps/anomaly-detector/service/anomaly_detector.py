from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from infra.alert_store import (
    ACTION_ALERT_ISSUED,
    ALERT_STATUS_OPEN,
    RESTRICTION_AUTO_RESOLVE_REASON,
)
from infra.conversation_store import ConversationStoreError
from infra.logging_utils import log_event
from service.feature_extractor import FEATURE_KEYS, ConversationFeatureExtractor
from service.scorer import AnomalyScorer, ScoringResult


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

    Ciclo de vida da restrição: leads pontuados NORMAIS nesta corrida têm
    restrições antigas `open` auto-resolvidas (`status="resolved"` com motivo
    e timestamp) — falso-positivo não bloqueia o agendamento para sempre.
    Leads ausentes do lote ou com falha de extração NÃO são resolvidos
    (fail-safe: só um sinal positivo e explícito de "normal" libera).
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
        anomalous_leads = {
            str(row["conversation"]["lead_id"])
            for row, result in zip(rows, results)
            if result.is_anomaly
        }
        resolved_at = self.now_fn().isoformat()
        anomalies = 0
        restricted = 0
        resolved = 0
        resolved_leads: set[str] = set()
        for row, result in zip(rows, results):
            if result.is_anomaly:
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
                continue
            lead_id = row["conversation"].get("lead_id")
            if not lead_id or str(lead_id) in anomalous_leads or str(lead_id) in resolved_leads:
                continue
            resolved_leads.add(str(lead_id))
            resolved_ids = self.alerts.resolve_open_restriction(
                str(lead_id), reason=RESTRICTION_AUTO_RESOLVE_REASON, resolved_at=resolved_at
            )
            if resolved_ids:
                resolved += len(resolved_ids)
                log_event(
                    "restriction_auto_resolved",
                    lead_id=str(lead_id),
                    resolved=len(resolved_ids),
                    resolved_at=resolved_at,
                )
        summary = {
            "conversations": len(conversations),
            "scored": len(rows),
            "anomalies": anomalies,
            "restricted": restricted,
            "resolved": resolved,
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
