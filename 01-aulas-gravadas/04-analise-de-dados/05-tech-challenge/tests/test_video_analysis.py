"""Tests for video/analysis.py — postural anomaly detection (rolling
z-score over angle/velocity), zone-critical-object alerting, and the
combined, timestamp-ordered deviation report.

Covers scenarios from the video-motion-analysis spec:
- Requirement: Detecção de anomalia postural com sensibilidade ajustável
  - Scenario: Frame com desvio postural acima do threshold
  - Scenario: Ajuste de sensibilidade pelo usuário
- Requirement: Detecção de objeto ou área crítica
  - Scenario: Objeto detectado em área crítica configurada
- Requirement: Relatório automático de desvios do vídeo
  - Scenario: Vídeo processado com anomalias detectadas
  - Scenario: Vídeo processado sem anomalias detectadas

No real YOLO model is involved — tests operate directly on the
``frame_series`` structure produced by ``video.pose.extract_frame_series``
(built here by hand) and on plain bounding boxes.
"""
import pytest
import streamlit as st

from video.analysis import analyze, box_intersects_zone


@pytest.fixture(autouse=True)
def clean_session_state():
    st.session_state.clear()
    yield
    st.session_state.clear()


def _frame(timestamp_s, angle=None, velocity=None, has_pose=True, detections=None):
    return {
        "timestamp_s": timestamp_s,
        "has_pose": has_pose,
        "angle": angle,
        "velocity": velocity,
        "detections": detections or [],
    }


# ── box_intersects_zone ───────────────────────────────────────────────


def test_box_intersects_zone_true_when_boxes_overlap():
    # zone occupies the right half of the frame (relative coords)
    zone = (0.5, 0.0, 1.0, 1.0)
    box_xyxy = [80, 10, 100, 20]  # within right half of a 100x100 frame

    assert box_intersects_zone(box_xyxy, zone, frame_width=100, frame_height=100) is True


def test_box_intersects_zone_false_when_boxes_do_not_overlap():
    zone = (0.5, 0.0, 1.0, 1.0)
    box_xyxy = [0, 10, 10, 20]  # entirely in the left half

    assert box_intersects_zone(box_xyxy, zone, frame_width=100, frame_height=100) is False


def test_box_intersects_zone_true_for_partial_overlap_at_boundary():
    zone = (0.5, 0.0, 1.0, 1.0)
    box_xyxy = [40, 10, 60, 20]  # straddles the zone boundary at x=50

    assert box_intersects_zone(box_xyxy, zone, frame_width=100, frame_height=100) is True


# ── analyze(): postural anomaly (angle/velocity z-score) ─────────────


def test_analyze_flags_postural_anomaly_above_threshold_and_generates_alert():
    # Stable angle series with a single clear spike.
    frames = [_frame(t, angle=90.0, velocity=1.0) for t in range(8)]
    frames.append(_frame(8.0, angle=10.0, velocity=1.0))  # sudden angle change
    frames += [_frame(t, angle=90.0, velocity=1.0) for t in range(9, 13)]

    result = analyze(frames, threshold=2.0, window=6)

    assert result["angle_anomalies"].iloc[8] in (True, 1)
    postural_alerts = [a for a in result["alerts"] if "postural" in a.description.lower() or "ângulo" in a.description.lower() or "velocidade" in a.description.lower()]
    assert len(postural_alerts) >= 1


def test_analyze_lower_threshold_detects_more_anomalies_than_higher_threshold():
    frames = [_frame(t, angle=90.0, velocity=float(t % 3)) for t in range(20)]
    frames[10] = _frame(10.0, angle=140.0, velocity=6.0)  # moderate deviation

    strict = analyze(frames, threshold=5.0, window=6)
    sensitive = analyze(frames, threshold=0.5, window=6)

    strict_count = int(strict["angle_anomalies"].sum() + strict["velocity_anomalies"].sum())
    sensitive_count = int(sensitive["angle_anomalies"].sum() + sensitive["velocity_anomalies"].sum())
    assert sensitive_count >= strict_count


def test_analyze_handles_frames_without_pose_data_without_raising():
    frames = [_frame(t, angle=90.0, velocity=1.0) for t in range(5)]
    frames.append(_frame(5.0, has_pose=False))
    frames += [_frame(t, angle=90.0, velocity=1.0) for t in range(6, 10)]

    result = analyze(frames, threshold=3.0, window=4)

    assert len(result["angle_anomalies"]) == len(frames)
    assert result["angle_anomalies"].iloc[5] in (False, 0)  # no data -> not anomalous


# ── analyze(): zone-critical-object alert ─────────────────────────────


def test_analyze_generates_immediate_alert_when_critical_object_enters_zone():
    frames = [
        _frame(
            0.0,
            angle=90.0,
            velocity=1.0,
            detections=[{"cls": 0, "label": "person", "xyxy": [80, 10, 100, 20]}],
        )
    ]

    zone = (0.5, 0.0, 1.0, 1.0)
    result = analyze(frames, threshold=3.0, window=4, zone=zone, frame_width=100, frame_height=100)

    zone_alerts = [a for a in result["alerts"] if "zona" in a.description.lower()]
    assert len(zone_alerts) == 1
    assert "person" in zone_alerts[0].description.lower()


def test_analyze_zone_alert_is_independent_of_zscore_threshold():
    # Even with a very high (insensitive) postural threshold, the zone
    # alert must still fire — it is a separate detection path.
    frames = [
        _frame(
            0.0,
            angle=90.0,
            velocity=1.0,
            detections=[{"cls": 0, "label": "person", "xyxy": [80, 10, 100, 20]}],
        )
    ]
    zone = (0.5, 0.0, 1.0, 1.0)

    result = analyze(frames, threshold=999.0, window=4, zone=zone, frame_width=100, frame_height=100)

    zone_alerts = [a for a in result["alerts"] if "zona" in a.description.lower()]
    assert len(zone_alerts) == 1


def test_analyze_no_zone_alert_when_no_detection_intersects_zone():
    frames = [
        _frame(
            0.0,
            angle=90.0,
            velocity=1.0,
            detections=[{"cls": 0, "label": "person", "xyxy": [0, 0, 10, 10]}],
        )
    ]
    zone = (0.5, 0.0, 1.0, 1.0)

    result = analyze(frames, threshold=3.0, window=4, zone=zone, frame_width=100, frame_height=100)

    zone_alerts = [a for a in result["alerts"] if "zona" in a.description.lower()]
    assert zone_alerts == []


# ── deviation report ordering ─────────────────────────────────────────


def test_deviation_report_is_ordered_by_timestamp_combining_both_alert_types():
    frames = [_frame(t, angle=90.0, velocity=1.0) for t in range(8)]
    frames.append(_frame(8.0, angle=10.0, velocity=1.0))  # postural anomaly at t=8
    frames += [_frame(t, angle=90.0, velocity=1.0) for t in range(9, 12)]
    # zone alert at an earlier timestamp than the postural anomaly
    frames[2]["detections"] = [{"cls": 0, "label": "person", "xyxy": [80, 10, 100, 20]}]

    zone = (0.5, 0.0, 1.0, 1.0)
    result = analyze(frames, threshold=2.0, window=6, zone=zone, frame_width=100, frame_height=100)

    report = result["deviation_report"]
    timestamps = [row["timestamp_s"] for row in report]
    assert timestamps == sorted(timestamps)
    assert len(report) >= 2
    kinds = {row["kind"] for row in report}
    assert "zona_critica" in kinds
    assert "postural" in kinds


def test_deviation_report_is_empty_when_no_anomaly_detected():
    frames = [_frame(t, angle=90.0, velocity=1.0) for t in range(10)]

    result = analyze(frames, threshold=5.0, window=4)

    assert result["deviation_report"] == []
