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

    def test_scores_are_batch_independent(self):
        scorer = HeuristicScorer()
        alone = scorer.score(suspicious_features())
        in_batch = scorer.score_many([normal_features(), suspicious_features(), normal_features()])[1]
        assert in_batch.score == alone.score
        assert in_batch.is_anomaly is True


class TestSklearnScorerCalibration:
    def test_fixed_isolation_calibration_maps_theoretical_range(self):
        assert SklearnScorer._calibrate_isolation(-1.0) == 0.0
        assert SklearnScorer._calibrate_isolation(0.0) == 0.5
        assert SklearnScorer._calibrate_isolation(1.0) == 1.0
        assert SklearnScorer._calibrate_isolation(5.0) == 1.0

    def test_fixed_reconstruction_calibration_maps_error_scale(self):
        assert SklearnScorer._calibrate_reconstruction(0.0) == 0.0
        assert SklearnScorer._calibrate_reconstruction(1.5) == 0.5
        assert SklearnScorer._calibrate_reconstruction(3.0) == 1.0
        assert SklearnScorer._calibrate_reconstruction(30.0) == 1.0

    def test_calibration_never_uses_batch_min_max(self):
        scorer = SklearnScorer()
        batch_margins = [0.0, 0.1, 0.2, 0.3, 0.4]
        calibrated = [scorer._calibrate_isolation(raw) for raw in batch_margins]
        assert calibrated == [scorer._calibrate_isolation(raw) for raw in batch_margins]
        assert max(calibrated) == scorer._calibrate_isolation(max(batch_margins))
        assert calibrated[0] == 0.5

    def test_baseline_rows_are_deterministic(self):
        first = SklearnScorer(random_state=42)._baseline_rows()
        second = SklearnScorer(random_state=42)._baseline_rows()
        assert first == second
        assert len(first) == SklearnScorer.BASELINE_SIZE

    def test_baseline_rows_vary_with_seed(self):
        assert SklearnScorer(random_state=42)._baseline_rows() != SklearnScorer(random_state=7)._baseline_rows()


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

    def test_small_batch_fallback_logs_json_event(self, caplog):
        import json
        import logging

        caplog.set_level(logging.INFO)
        SklearnScorer().score_many([normal_features()])
        record = [r for r in caplog.records if r.name == "infra.logging_utils"][-1]
        event = json.loads(record.getMessage())
        assert event["event"] == "sklearn_scorer_fallback"
        assert event["batch_size"] == 1
        assert event["min_batch"] == SklearnScorer.MIN_BATCH

    def test_real_sklearn_flags_injected_outlier(self):
        pytest.importorskip("sklearn")
        rng_rows = [normal_features() for _ in range(12)]
        rows = rng_rows + [suspicious_features()]
        results = SklearnScorer().score_many(rows)
        assert len(results) == len(rows)
        assert results[-1].is_anomaly is True
        assert results[-1].kind == KIND_ENSEMBLE

    def test_real_sklearn_inlier_batch_declares_no_anomaly(self):
        pytest.importorskip("sklearn")
        import random

        rng = random.Random(7)
        rows = [
            {
                FEATURE_VOLUME: rng.uniform(0, 20),
                FEATURE_LENGTH: rng.uniform(0, 400),
                FEATURE_SENTIMENT: rng.uniform(0, 0.2),
                FEATURE_HOURS: rng.uniform(0, 0.2),
            }
            for _ in range(20)
        ]
        results = SklearnScorer().score_many(rows)
        assert all(result.is_anomaly is False for result in results)

    def test_real_sklearn_same_item_same_score_in_different_batch_sizes(self):
        pytest.importorskip("sklearn")
        scorer = SklearnScorer()
        small_batch = [normal_features() for _ in range(9)] + [suspicious_features()]
        large_batch = [normal_features() for _ in range(39)] + [suspicious_features()]
        small = scorer.score_many(small_batch)[-1]
        large = scorer.score_many(large_batch)[-1]
        assert small.score == large.score
        assert small.details == large.details
