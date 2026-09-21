from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from service.feature_extractor import (
    FEATURE_HOURS,
    FEATURE_LENGTH,
    FEATURE_SENTIMENT,
    FEATURE_VOLUME,
)

logger = logging.getLogger(__name__)

KIND_NONE = "none"
KIND_ENSEMBLE = "ml_ensemble"
KIND_COMPOSITE = "composite"

KIND_BY_FEATURE = {
    FEATURE_VOLUME: "high_volume",
    FEATURE_LENGTH: "long_messages",
    FEATURE_SENTIMENT: "negative_sentiment",
    FEATURE_HOURS: "atypical_hours",
}


class ScorerDependencyError(Exception):
    """Dependência ML ausente no runtime (ex.: scikit-learn não instalado)."""


@dataclass(frozen=True)
class ScoringResult:
    score: float
    is_anomaly: bool
    kind: str
    details: dict[str, float] = field(default_factory=dict)


class AnomalyScorer(Protocol):
    def score_many(self, feature_rows: list[dict[str, Any]]) -> list[ScoringResult]: ...


class HeuristicScorer:
    """Scorer determinístico 100% stdlib (fallback executável sem sklearn).

    Combinação ponderada de contribuições normalizadas por faixas — postura
    de falso-positivo documentada: thresholds conservadores por feature e
    anomalia declarada somente quando o score agregado cruza o threshold.
    """

    WEIGHTS = {
        FEATURE_VOLUME: 0.3,
        FEATURE_LENGTH: 0.2,
        FEATURE_SENTIMENT: 0.3,
        FEATURE_HOURS: 0.2,
    }
    RANGES = {
        FEATURE_VOLUME: (20.0, 60.0),
        FEATURE_LENGTH: (400.0, 1200.0),
        FEATURE_SENTIMENT: (0.2, 0.8),
        FEATURE_HOURS: (0.2, 0.8),
    }

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    def score_many(self, feature_rows: list[dict[str, Any]]) -> list[ScoringResult]:
        return [self.score(row) for row in feature_rows]

    def score(self, features: dict[str, Any]) -> ScoringResult:
        contributions: dict[str, float] = {}
        for key, (low, high) in self.RANGES.items():
            value = float(features.get(key) or 0.0)
            normalized = min(max((value - low) / (high - low), 0.0), 1.0)
            contributions[key] = round(normalized * self.WEIGHTS[key], 4)
        score = round(sum(contributions.values()), 4)
        is_anomaly = score >= self.threshold
        kind = KIND_NONE
        if is_anomaly:
            top = max(contributions.values())
            if top <= 0.0:
                kind = KIND_COMPOSITE
            else:
                kind = KIND_BY_FEATURE[max(contributions, key=contributions.get)]
        return ScoringResult(score=score, is_anomaly=is_anomaly, kind=kind, details=contributions)


class SklearnScorer:
    """Ensemble Isolation Forest + PCA (erro de reconstrução como sinal
    residual estilo autoencoder) — FR9.2.

    Imports guardados dentro do método: sem scikit-learn no runtime levanta
    `ScorerDependencyError` com mensagem clara. Lotes pequenos (< MIN_BATCH)
    caem para o fallback heurístico: fit não-supervisionado em amostra mínima
    gera falso-positivo evitável (postura documentada no code-summary).
    """

    MIN_BATCH = 5

    def __init__(
        self,
        threshold: float = 0.7,
        contamination: float = 0.1,
        random_state: int = 42,
        fallback: AnomalyScorer | None = None,
    ) -> None:
        self.threshold = threshold
        self.contamination = contamination
        self.random_state = random_state
        self._fallback = fallback or HeuristicScorer(threshold)

    def score_many(self, feature_rows: list[dict[str, Any]]) -> list[ScoringResult]:
        if len(feature_rows) < self.MIN_BATCH:
            logger.info(
                "sklearn scorer skipped: batch of %s below minimum %s; using heuristic fallback",
                len(feature_rows),
                self.MIN_BATCH,
            )
            return self._fallback.score_many(feature_rows)
        try:
            from sklearn.decomposition import PCA
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise ScorerDependencyError(
                "scikit-learn indisponível no runtime: instale scikit-learn>=1.4 "
                "ou configure ANOMALY_SCORER=heuristic para o fallback determinístico"
            ) from exc
        isolation, reconstruction = self._ensemble_scores(
            feature_rows, IsolationForest, PCA, StandardScaler
        )
        results: list[ScoringResult] = []
        for index, _row in enumerate(feature_rows):
            score = round(0.6 * isolation[index] + 0.4 * reconstruction[index], 4)
            is_anomaly = score >= self.threshold
            results.append(
                ScoringResult(
                    score=score,
                    is_anomaly=is_anomaly,
                    kind=KIND_ENSEMBLE if is_anomaly else KIND_NONE,
                    details={
                        "isolation": round(isolation[index], 4),
                        "pca_reconstruction": round(reconstruction[index], 4),
                    },
                )
            )
        return results

    def _ensemble_scores(
        self, rows: list[dict[str, Any]], IsolationForest: Any, PCA: Any, StandardScaler: Any
    ) -> tuple[list[float], list[float]]:
        import numpy as np

        matrix = np.array(
            [[float(row.get(key) or 0.0) for key in (FEATURE_VOLUME, FEATURE_LENGTH, FEATURE_SENTIMENT, FEATURE_HOURS)] for row in rows]
        )
        scaled = StandardScaler().fit_transform(matrix)
        iso_model = IsolationForest(
            contamination=self.contamination, random_state=self.random_state
        ).fit(scaled)
        isolation = self._normalize_to_unit(-iso_model.decision_function(scaled))
        pca_model = PCA(n_components=2, random_state=self.random_state).fit(scaled)
        reconstructed = pca_model.inverse_transform(pca_model.transform(scaled))
        errors = ((scaled - reconstructed) ** 2).mean(axis=1)
        reconstruction = self._normalize_to_unit(errors)
        return isolation, reconstruction

    @staticmethod
    def _normalize_to_unit(values: Any) -> list[float]:
        low, high = float(min(values)), float(max(values))
        span = high - low
        if span <= 0.0:
            return [0.0 for _ in values]
        return [round((float(value) - low) / span, 6) for value in values]
