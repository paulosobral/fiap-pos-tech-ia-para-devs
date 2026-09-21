import sys

import pytest

from service.feature_extractor import FEATURE_HOURS, FEATURE_LENGTH, FEATURE_SENTIMENT, FEATURE_VOLUME
from service.scorer import (
    KIND_ENSEMBLE,
    ScorerDependencyError,
    HeuristicScorer,
    SklearnScorer,
)


def suspicious_features():
    return {FEATURE_VOLUME: 30, FEATURE_LENGTH: 1300.0, FEATURE_SENTIMENT: 1.0, FEATURE_HOURS: 1.0}


def normal_features():
    return {FEATURE_VOLUME: 4, FEATURE_LENGTH: 60.0, FEATURE_SENTIMENT: 0.0, FEATURE_HOURS: 0.0}


def hide_sklearn(monkeypatch):
    for name in ("sklearn", "sklearn.ensemble", "sklearn.decomposition", "sklearn.preprocessing"):
        monkeypatch.setitem(sys.modules, name, None)


class TestHeuristicScorer:
    def test_normal_conversation_scores_low_and_is_not_anomaly(self):
        result = HeuristicScorer().score(normal_features())
        assert result.score == 0.0
        assert result.is_anomaly is False
        assert result.kind == "none"

    def test_suspicious_conversation_is_flagged_with_dominant_kind(self):
        result = HeuristicScorer().score(suspicious_features())
        assert result.is_anomaly is True
        assert result.score == pytest.approx(0.775, abs=1e-4)
        assert result.kind == "negative_sentiment"

    def test_zero_features_score_zero(self):
        result = HeuristicScorer().score({})
        assert result.score == 0.0
        assert result.is_anomaly is False

    def test_score_at_threshold_is_anomaly_inclusive(self):
        features = {FEATURE_VOLUME: 40, FEATURE_LENGTH: 800.0, FEATURE_SENTIMENT: 0.0, FEATURE_HOURS: 0.0}
        result = HeuristicScorer(threshold=0.25).score(features)
        assert result.score == pytest.approx(0.25, abs=1e-4)
        assert result.is_anomaly is True

    def test_score_below_threshold_is_not_anomaly(self):
        features = {FEATURE_VOLUME: 39, FEATURE_LENGTH: 800.0, FEATURE_SENTIMENT: 0.0, FEATURE_HOURS: 0.0}
        result = HeuristicScorer(threshold=0.25).score(features)
        assert result.score < 0.25
        assert result.is_anomaly is False

    def test_score_many_aligns_results_by_index(self):
        results = HeuristicScorer().score_many([normal_features(), suspicious_features()])
        assert [r.is_anomaly for r in results] == [False, True]


class TestSklearnScorer:
    def test_missing_dependency_raises_clear_error(self, monkeypatch):
        hide_sklearn(monkeypatch)
        scorer = SklearnScorer()
        rows = [suspicious_features() for _ in range(6)]
        with pytest.raises(ScorerDependencyError, match="scikit-learn indisponível"):
            scorer.score_many(rows)

    def test_small_batch_falls_back_to_heuristic(self):
        rows = [normal_features(), suspicious_features()]
        fallback = HeuristicScorer()
        results = SklearnScorer(threshold=fallback.threshold).score_many(rows)
        assert results == fallback.score_many(rows)

    def test_missing_dependency_with_small_batch_never_raises(self, monkeypatch):
        hide_sklearn(monkeypatch)
        scorer = SklearnScorer()
        results = scorer.score_many([suspicious_features()])
        assert results[0].is_anomaly is True

    def test_real_sklearn_flags_injected_outlier(self):
        pytest.importorskip("sklearn")
        rng_rows = [normal_features() for _ in range(12)]
        rows = rng_rows + [suspicious_features()]
        results = SklearnScorer().score_many(rows)
        assert len(results) == len(rows)
        assert results[-1].is_anomaly is True
        assert results[-1].kind == KIND_ENSEMBLE
