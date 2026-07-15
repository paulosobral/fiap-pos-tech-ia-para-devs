"""Tests for video/analysis.py — per-joint postural anomaly detection
(rolling z-score over each joint-angle series and the global velocity
series), grouping of consecutive irregular frames into events, one alert
per event, the event summary (most-affected joint), and the optional
zone-critical grouping.

Covers scenarios from the redesenho-aba-video-postura
video-motion-analysis spec:
- Requirement: Detecção de anomalia postural por articulação
  - Scenario: Frame com desvio postural acima do threshold em alguma articulação
  - Scenario: Feedback do efeito da sensibilidade escolhida
- Requirement: Agrupamento de frames irregulares consecutivos em eventos
  - Scenario: Sequência de frames irregulares da mesma articulação vira um evento
  - Scenario: Feed unificado recebe um alerta por evento, não por frame
- Requirement: Detecção opcional de zona crítica
  - Scenario: Zona crítica desativada por padrão
  - Scenario: Detecção de zona agrupada em evento
- Requirement: Relatório visual de desvios do vídeo
  - Scenario: resumo com contagem de eventos e articulação mais afetada

No real YOLO model is involved — tests operate directly on the
``frame_series`` structure produced by ``video.pose.extract_frame_series``
(built here by hand: ``angles`` dict, ``velocity``, ``keypoints_xy``,
``detections``) and on plain bounding boxes.
"""
import pytest
import streamlit as st

from video.analysis import (
    DEFAULT_ZONE,
    FALLBACK_SENSITIVITY_THRESHOLD,
    JOINT_CATEGORY,
    JOINT_LABELS,
    VELOCITY_SECTION_LABEL,
    ZONE_SECTION_LABEL,
    _event_description,
    _group_consecutive,
    analyze,
    box_intersects_zone,
    estimate_flagged_fraction,
    event_category,
    group_events_for_display,
    joint_label,
    suggest_sensitivity_threshold,
)


@pytest.fixture(autouse=True)
def clean_session_state():
    st.session_state.clear()
    yield
    st.session_state.clear()


def _frame(timestamp_s, angles=None, velocity=None, has_pose=True, detections=None, keypoints_xy=None):
    """Build a frame in the new (Task 1) shape."""
    return {
        "timestamp_s": timestamp_s,
        "has_pose": has_pose,
        "angles": angles if angles is not None else {},
        "velocity": velocity,
        "keypoints_xy": keypoints_xy,
        "detections": detections or [],
    }


# ── box_intersects_zone (unchanged behaviour) ─────────────────────────


def test_box_intersects_zone_true_when_boxes_overlap():
    zone = (0.5, 0.0, 1.0, 1.0)
    box_xyxy = [80, 10, 100, 20]
    assert box_intersects_zone(box_xyxy, zone, frame_width=100, frame_height=100) is True


def test_box_intersects_zone_false_when_boxes_do_not_overlap():
    zone = (0.5, 0.0, 1.0, 1.0)
    box_xyxy = [0, 10, 10, 20]
    assert box_intersects_zone(box_xyxy, zone, frame_width=100, frame_height=100) is False


def test_box_intersects_zone_true_for_partial_overlap_at_boundary():
    zone = (0.5, 0.0, 1.0, 1.0)
    box_xyxy = [40, 10, 60, 20]
    assert box_intersects_zone(box_xyxy, zone, frame_width=100, frame_height=100) is True


def test_default_zone_matches_documented_right_hand_fifth_of_frame():
    assert DEFAULT_ZONE == (0.7, 0.0, 1.0, 1.0)


# ── _group_consecutive: grouping + gap tolerance ──────────────────────


def test_group_consecutive_merges_strictly_consecutive_indices():
    assert _group_consecutive([3, 4, 5]) == [[3, 4, 5]]


def test_group_consecutive_merges_across_small_gap():
    # gap tolerance 1: a single missing frame (index 4) between 3 and 5
    # must not fragment the event.
    assert _group_consecutive([3, 5, 6]) == [[3, 5, 6]]


def test_group_consecutive_splits_on_large_gap():
    # A gap larger than the tolerance starts a new event.
    assert _group_consecutive([3, 4, 20, 21]) == [[3, 4], [20, 21]]


def test_group_consecutive_empty():
    assert _group_consecutive([]) == []


# ── per-joint detection ───────────────────────────────────────────────


def _spike_frames(joint, baseline, spike, n=16, spike_at=(10, 11, 12), other=None, velocity=1.0):
    """Frames where ``joint`` sits at ``baseline`` except ``spike_at`` at ``spike``.

    Optionally carries a second constant joint (``other``) that must never
    be flagged (constant series -> zero variance -> no z-score).
    """
    frames = []
    for t in range(n):
        angles = {joint: spike if t in spike_at else baseline}
        if other is not None:
            angles[other] = 90.0
        frames.append(_frame(float(t), angles=angles, velocity=velocity))
    return frames


def test_analyze_flags_only_the_joint_with_the_spike():
    frames = _spike_frames("joelho_direito", baseline=170.0, spike=90.0, other="cotovelo_direito")

    result = analyze(frames, threshold=1.3, window=6)

    postura_joints = {e["articulacao"] for e in result["events"] if e["tipo"] == "postura"}
    assert "joelho_direito" in postura_joints
    assert "cotovelo_direito" not in postura_joints  # constant -> never flagged


def test_analyze_event_has_expected_schema_and_worst_frame_value():
    frames = _spike_frames("joelho_direito", baseline=170.0, spike=90.0)

    result = analyze(frames, threshold=1.3, window=6)

    postura = [e for e in result["events"] if e["tipo"] == "postura"]
    assert postura, "expected at least one postura event"
    event = postura[0]
    assert set(event.keys()) == {
        "tipo",
        "articulacao",
        "t_inicio",
        "t_fim",
        "frame_index_pior",
        "valor_pior",
        "z_pior",
        "event_id",
    }
    assert event["articulacao"] == "joelho_direito"
    # The worst frame (max |z|) is the first frame of the drop (index 10),
    # whose value is the spike value.
    assert event["frame_index_pior"] == 10
    assert event["valor_pior"] == pytest.approx(90.0)
    assert abs(event["z_pior"]) > 1.3


# ── event grouping ────────────────────────────────────────────────────


def test_consecutive_anomalies_group_into_one_event_with_correct_worst_frame():
    # Velocity series: flat 1.0, a run of 10.0, then flat again. Only the
    # first two frames of the run exceed threshold=1.3; they are one event
    # and the worst frame (max |z|) is the first (index 10).
    frames = [_frame(float(t), velocity=1.0) for t in range(10)]
    frames += [_frame(float(t), velocity=10.0) for t in range(10, 13)]
    frames += [_frame(float(t), velocity=1.0) for t in range(13, 16)]

    result = analyze(frames, threshold=1.3, window=6)

    vel_events = [e for e in result["events"] if e["tipo"] == "velocidade"]
    assert len(vel_events) == 1
    assert vel_events[0]["frame_index_pior"] == 10
    assert vel_events[0]["t_inicio"] == 10.0


def test_one_alert_per_event():
    frames = _spike_frames("joelho_direito", baseline=170.0, spike=90.0)

    result = analyze(frames, threshold=1.3, window=6)

    assert len(result["events"]) >= 1
    assert len(result["alerts"]) == len(result["events"])


def test_alert_description_is_human_readable_with_joint_and_interval():
    frames = _spike_frames("joelho_direito", baseline=170.0, spike=90.0)

    result = analyze(frames, threshold=1.3, window=6)

    descriptions = " ".join(a.description for a in result["alerts"])
    assert "Joelho direito" in descriptions
    assert "entre" in descriptions and "s" in descriptions
    # origin is always the Vídeo tab
    assert all(a.origin == "Vídeo" for a in result["alerts"])


def test_events_sorted_by_start_time():
    # A zone event early and a postural event late; events must come out
    # ordered by t_inicio.
    frames = [_frame(float(t), angles={"joelho_direito": 170.0}, velocity=1.0) for t in range(10)]
    frames += [_frame(float(t), angles={"joelho_direito": 90.0}, velocity=1.0) for t in range(10, 13)]
    frames += [_frame(float(t), angles={"joelho_direito": 170.0}, velocity=1.0) for t in range(13, 16)]
    # zone intersection at the very first frames
    for t in (0, 1):
        frames[t]["detections"] = [{"cls": 0, "label": "person", "xyxy": [80, 10, 100, 20]}]

    zone = (0.5, 0.0, 1.0, 1.0)
    result = analyze(frames, threshold=1.3, window=6, zone=zone, frame_width=100, frame_height=100)

    starts = [e["t_inicio"] for e in result["events"]]
    assert starts == sorted(starts)
    tipos = {e["tipo"] for e in result["events"]}
    assert "zona_critica" in tipos
    assert "postura" in tipos


# ── summary ───────────────────────────────────────────────────────────


def _two_joint_frames():
    """joelho_direito has two separate events; cotovelo_direito has one."""
    joelho = [170.0] * 40
    cotovelo = [90.0] * 40
    for i in (10, 11, 12, 24, 25, 26):
        joelho[i] = 90.0
    for i in (10, 11, 12):
        cotovelo[i] = 40.0
    return [
        _frame(
            float(t),
            angles={"joelho_direito": joelho[t], "cotovelo_direito": cotovelo[t]},
            velocity=1.0,
        )
        for t in range(40)
    ]


def test_summary_reports_most_affected_joint():
    result = analyze(_two_joint_frames(), threshold=1.3, window=6)

    summary = result["summary"]
    assert summary["most_affected_joint"] == "joelho_direito"
    assert summary["most_affected_label"] == "Joelho direito"
    assert summary["total_events"] == len(result["events"])


def test_summary_when_no_events():
    frames = [_frame(float(t), angles={"joelho_direito": 90.0}, velocity=1.0) for t in range(10)]

    result = analyze(frames, threshold=5.0, window=4)

    assert result["events"] == []
    assert result["alerts"] == []
    assert result["summary"]["total_events"] == 0
    assert result["summary"]["most_affected_joint"] is None
    assert result["summary"]["most_affected_label"] is None


def test_analyze_handles_frames_without_pose_without_raising():
    frames = [_frame(float(t), angles={"joelho_direito": 90.0}, velocity=1.0) for t in range(5)]
    frames.append(_frame(5.0, angles={"joelho_direito": None}, velocity=None, has_pose=False))
    frames += [_frame(float(t), angles={"joelho_direito": 90.0}, velocity=1.0) for t in range(6, 10)]

    result = analyze(frames, threshold=3.0, window=4)  # must not raise
    assert "events" in result


# ── zone-critical grouping ────────────────────────────────────────────


def test_zone_intersections_group_into_one_event():
    box = [80, 10, 100, 20]
    frames = [
        _frame(float(t), velocity=1.0, detections=[{"cls": 0, "label": "person", "xyxy": box}])
        for t in range(5)
    ]
    zone = (0.5, 0.0, 1.0, 1.0)

    result = analyze(frames, threshold=3.0, window=4, zone=zone, frame_width=100, frame_height=100)

    zone_events = [e for e in result["events"] if e["tipo"] == "zona_critica"]
    assert len(zone_events) == 1
    assert zone_events[0]["t_inicio"] == 0.0
    assert zone_events[0]["t_fim"] == 4.0
    zone_alerts = [a for a in result["alerts"] if "zona" in a.description.lower()]
    assert len(zone_alerts) == 1  # one alert per event, not per frame


def test_zone_alert_is_independent_of_zscore_threshold():
    box = [80, 10, 100, 20]
    frames = [
        _frame(0.0, velocity=1.0, detections=[{"cls": 0, "label": "person", "xyxy": box}])
    ]
    zone = (0.5, 0.0, 1.0, 1.0)

    result = analyze(frames, threshold=999.0, window=4, zone=zone, frame_width=100, frame_height=100)

    zone_events = [e for e in result["events"] if e["tipo"] == "zona_critica"]
    assert len(zone_events) == 1


def test_zero_zone_events_when_zone_is_none():
    box = [80, 10, 100, 20]
    frames = [
        _frame(0.0, velocity=1.0, detections=[{"cls": 0, "label": "person", "xyxy": box}])
    ]

    result = analyze(frames, threshold=3.0, window=4, zone=None, frame_width=100, frame_height=100)

    assert [e for e in result["events"] if e["tipo"] == "zona_critica"] == []


def test_no_zone_event_when_no_detection_intersects_zone():
    box = [0, 0, 10, 10]  # left corner, outside the right-half zone
    frames = [
        _frame(0.0, velocity=1.0, detections=[{"cls": 0, "label": "person", "xyxy": box}])
    ]
    zone = (0.5, 0.0, 1.0, 1.0)

    result = analyze(frames, threshold=3.0, window=4, zone=zone, frame_width=100, frame_height=100)

    assert [e for e in result["events"] if e["tipo"] == "zona_critica"] == []


# ── joint label mapping ───────────────────────────────────────────────


def test_joint_label_maps_internal_keys_to_human_labels():
    assert JOINT_LABELS["joelho_direito"] == "Joelho direito"
    assert joint_label("cotovelo_esquerdo") == "Cotovelo esquerdo"
    assert joint_label("pescoco") == "Pescoço"
    # velocity (None) and unknown keys degrade gracefully
    assert joint_label(None)  # returns a non-empty label


# ── estimate_flagged_fraction ─────────────────────────────────────────


def test_estimate_flagged_fraction_between_zero_and_one():
    frames = _two_joint_frames()
    frac = estimate_flagged_fraction(frames, threshold=1.3, window=6)
    assert 0.0 <= frac <= 1.0
    assert frac > 0.0  # this video does have flagged frames


def test_estimate_flagged_fraction_is_higher_for_lower_threshold():
    frames = _two_joint_frames()
    frac_sensitive = estimate_flagged_fraction(frames, threshold=0.5, window=6)
    frac_strict = estimate_flagged_fraction(frames, threshold=5.0, window=6)
    assert frac_sensitive >= frac_strict


def test_estimate_flagged_fraction_empty_series_is_zero():
    assert estimate_flagged_fraction([], threshold=1.0, window=6) == 0.0


# ── suggest_sensitivity_threshold (over ALL joint series) ─────────────


def test_suggest_sensitivity_threshold_falls_back_when_no_frame_has_pose():
    frames = [
        _frame(float(t), angles={"joelho_direito": None}, velocity=None, has_pose=False)
        for t in range(20)
    ]

    assert suggest_sensitivity_threshold(frames, window=6) == FALLBACK_SENSITIVITY_THRESHOLD


def test_suggest_sensitivity_threshold_falls_back_when_series_shorter_than_window():
    frames = [
        _frame(float(t), angles={"joelho_direito": 90.0 + t}, velocity=1.0 + t) for t in range(4)
    ]

    assert suggest_sensitivity_threshold(frames, window=6) == FALLBACK_SENSITIVITY_THRESHOLD


_JITTER = (0.0, 0.3, -0.2, 0.1, -0.1, 0.2)


def test_suggest_sensitivity_threshold_within_bounds_for_stable_motion():
    frames = [
        _frame(
            float(t),
            angles={
                "cotovelo_direito": 90.0 + _JITTER[t % len(_JITTER)],
                "joelho_direito": 170.0 + _JITTER[t % len(_JITTER)],
            },
            velocity=1.0 + _JITTER[t % len(_JITTER)] * 0.05,
        )
        for t in range(40)
    ]

    suggestion = suggest_sensitivity_threshold(frames, window=6)

    assert 0.5 <= suggestion <= 5.0
    assert suggestion != FALLBACK_SENSITIVITY_THRESHOLD


def test_suggest_sensitivity_threshold_higher_for_more_variable_motion():
    stable = [
        _frame(
            float(t),
            angles={"cotovelo_direito": 90.0 + _JITTER[t % len(_JITTER)]},
            velocity=1.0 + _JITTER[t % len(_JITTER)] * 0.05,
        )
        for t in range(40)
    ]
    variable = [
        _frame(
            float(t),
            angles={
                "cotovelo_direito": 90.0 + (60.0 if t % 8 == 0 else _JITTER[t % len(_JITTER)])
            },
            velocity=1.0 + (8.0 if t % 8 == 0 else _JITTER[t % len(_JITTER)] * 0.05),
        )
        for t in range(40)
    ]

    assert suggest_sensitivity_threshold(variable, window=6) > suggest_sensitivity_threshold(
        stable, window=6
    )


def test_suggest_sensitivity_threshold_ignores_frames_without_pose():
    frames = [
        _frame(float(t), angles={"cotovelo_direito": 90.0}, velocity=1.0) for t in range(15)
    ]
    frames += [
        _frame(float(t), angles={"cotovelo_direito": None}, velocity=None, has_pose=False)
        for t in range(15, 25)
    ]

    suggestion = suggest_sensitivity_threshold(frames, window=6)  # must not raise
    assert 0.5 <= suggestion <= 5.0


# ── group_events_for_display (grouping for the gallery UI) ────────────


def _event(tipo, articulacao=None, t_inicio=0.0, z_pior=0.0, valor_pior=0.0):
    """Minimal event dict in the shape analyze() produces."""
    return {
        "tipo": tipo,
        "articulacao": articulacao,
        "t_inicio": t_inicio,
        "t_fim": t_inicio + 1.0,
        "frame_index_pior": int(t_inicio),
        "valor_pior": valor_pior,
        "z_pior": z_pior,
    }


def test_group_events_for_display_empty_input_returns_empty_list():
    assert group_events_for_display([]) == []


def test_group_events_for_display_groups_postura_by_joint():
    events = [
        _event("postura", "joelho_direito", t_inicio=0.0, z_pior=2.0),
        _event("postura", "joelho_direito", t_inicio=5.0, z_pior=3.0),
        _event("postura", "cotovelo_esquerdo", t_inicio=1.0, z_pior=4.0),
    ]

    sections = group_events_for_display(events)

    by_chave = {s["chave"]: s for s in sections}
    assert set(by_chave) == {"joelho_direito", "cotovelo_esquerdo"}
    assert by_chave["joelho_direito"]["total"] == 2
    assert by_chave["joelho_direito"]["label"] == "Joelho direito"
    assert by_chave["cotovelo_esquerdo"]["total"] == 1


def test_group_events_for_display_velocity_and_zone_each_one_section():
    events = [
        _event("velocidade", None, t_inicio=0.0, z_pior=2.0),
        _event("velocidade", None, t_inicio=3.0, z_pior=1.0),
        _event("zona_critica", None, t_inicio=1.0, z_pior=float("nan"), valor_pior=50.0),
    ]

    sections = group_events_for_display(events)
    by_chave = {s["chave"]: s for s in sections}

    assert by_chave["velocidade"]["total"] == 2
    assert by_chave["velocidade"]["label"] == VELOCITY_SECTION_LABEL
    assert by_chave["zona_critica"]["total"] == 1
    assert by_chave["zona_critica"]["label"] == ZONE_SECTION_LABEL


def test_group_events_for_display_orders_events_by_severity_within_section():
    events = [
        _event("postura", "joelho_direito", t_inicio=0.0, z_pior=1.0),
        _event("postura", "joelho_direito", t_inicio=5.0, z_pior=-4.0),
        _event("postura", "joelho_direito", t_inicio=2.0, z_pior=2.5),
    ]

    section = group_events_for_display(events)[0]

    zs = [abs(e["z_pior"]) for e in section["eventos"]]
    assert zs == sorted(zs, reverse=True)
    assert zs[0] == 4.0  # most severe first (uses abs)


def test_group_events_for_display_severity_tie_broken_by_t_inicio():
    events = [
        _event("postura", "joelho_direito", t_inicio=9.0, z_pior=2.0),
        _event("postura", "joelho_direito", t_inicio=1.0, z_pior=2.0),
    ]

    section = group_events_for_display(events)[0]

    assert [e["t_inicio"] for e in section["eventos"]] == [1.0, 9.0]


def test_group_events_for_display_truncates_to_top_n_preserving_total():
    events = [
        _event("postura", "joelho_direito", t_inicio=float(i), z_pior=float(i))
        for i in range(15)
    ]

    section = group_events_for_display(events, top_n=10)[0]

    assert section["total"] == 15
    assert len(section["eventos"]) == 10
    # kept the 10 most severe (highest z), i.e. i = 14..5
    assert [e["z_pior"] for e in section["eventos"]] == [float(i) for i in range(14, 4, -1)]


def test_group_events_for_display_orders_sections_by_count_desc():
    events = [
        _event("postura", "cotovelo_esquerdo", t_inicio=0.0, z_pior=1.0),
        _event("postura", "joelho_direito", t_inicio=1.0, z_pior=1.0),
        _event("postura", "joelho_direito", t_inicio=2.0, z_pior=1.0),
        _event("postura", "joelho_direito", t_inicio=3.0, z_pior=1.0),
    ]

    sections = group_events_for_display(events)

    assert [s["chave"] for s in sections] == ["joelho_direito", "cotovelo_esquerdo"]


def test_group_events_for_display_velocity_zone_sections_last_on_tie():
    # Each section has one event: tie on count must place joints first (in
    # JOINT_LABELS order), then velocity/zone last.
    events = [
        _event("velocidade", None, t_inicio=0.0, z_pior=1.0),
        _event("zona_critica", None, t_inicio=1.0, z_pior=float("nan"), valor_pior=5.0),
        _event("postura", "joelho_direito", t_inicio=2.0, z_pior=1.0),
        _event("postura", "cotovelo_direito", t_inicio=3.0, z_pior=1.0),
    ]

    chaves = [s["chave"] for s in group_events_for_display(events)]

    # joints in canonical JOINT_LABELS order first, then velocidade/zona
    assert chaves.index("cotovelo_direito") < chaves.index("joelho_direito")
    assert chaves.index("joelho_direito") < chaves.index("velocidade")
    assert chaves.index("joelho_direito") < chaves.index("zona_critica")


def test_group_events_for_display_zone_section_with_nan_z_orders_by_valor_pior():
    events = [
        _event("zona_critica", None, t_inicio=0.0, z_pior=float("nan"), valor_pior=10.0),
        _event("zona_critica", None, t_inicio=1.0, z_pior=float("nan"), valor_pior=99.0),
        _event("zona_critica", None, t_inicio=2.0, z_pior=float("nan"), valor_pior=42.0),
    ]

    section = group_events_for_display(events)[0]  # must not crash on NaN

    assert [e["valor_pior"] for e in section["eventos"]] == [99.0, 42.0, 10.0]


# ── event_id assignment (unique, deterministic, sequential) ───────────


def _multi_event_frames():
    """Frames producing several events across joints + velocity, so the
    resulting event list has more than one entry to id."""
    joelho = [170.0] * 40
    cotovelo = [90.0] * 40
    for i in (10, 11, 12, 24, 25, 26):
        joelho[i] = 90.0
    for i in (5, 6, 7):
        cotovelo[i] = 40.0
    vel = [1.0] * 40
    for i in (30, 31, 32):
        vel[i] = 12.0
    return [
        _frame(
            float(t),
            angles={"joelho_direito": joelho[t], "cotovelo_direito": cotovelo[t]},
            velocity=vel[t],
        )
        for t in range(40)
    ]


def test_analyze_assigns_event_id_to_every_event():
    result = analyze(_multi_event_frames(), threshold=1.3, window=6)

    assert result["events"], "expected several events"
    assert all("event_id" in e for e in result["events"])


def test_event_ids_are_unique_across_events():
    result = analyze(_multi_event_frames(), threshold=1.3, window=6)

    ids = [e["event_id"] for e in result["events"]]
    assert len(ids) == len(set(ids))


def test_event_ids_are_sequential_in_t_inicio_order():
    result = analyze(_multi_event_frames(), threshold=1.3, window=6)

    events = result["events"]
    # events come sorted by t_inicio; ids must be #V01, #V02, ... in that order
    assert [e["event_id"] for e in events] == [f"#V{i:02d}" for i in range(1, len(events) + 1)]
    # sanity: still sorted by t_inicio
    assert [e["t_inicio"] for e in events] == sorted(e["t_inicio"] for e in events)


def test_event_ids_are_deterministic_across_runs():
    frames = _multi_event_frames()
    first = [e["event_id"] for e in analyze(frames, threshold=1.3, window=6)["events"]]
    st.session_state.clear()
    second = [e["event_id"] for e in analyze(frames, threshold=1.3, window=6)["events"]]
    assert first == second


# ── event_category mapping ────────────────────────────────────────────


def test_joint_category_maps_regions():
    assert JOINT_CATEGORY["pescoco"] == "Cabeça"
    assert JOINT_CATEGORY["cotovelo_esquerdo"] == "Braços"
    assert JOINT_CATEGORY["cotovelo_direito"] == "Braços"
    assert JOINT_CATEGORY["quadril_esquerdo"] == "Tronco"
    assert JOINT_CATEGORY["quadril_direito"] == "Tronco"
    assert JOINT_CATEGORY["joelho_esquerdo"] == "Pernas"
    assert JOINT_CATEGORY["joelho_direito"] == "Pernas"


def test_event_category_for_postura_uses_joint_region():
    assert event_category(_event("postura", "pescoco")) == "Cabeça"
    assert event_category(_event("postura", "cotovelo_direito")) == "Braços"
    assert event_category(_event("postura", "quadril_esquerdo")) == "Tronco"
    assert event_category(_event("postura", "joelho_direito")) == "Pernas"


def test_event_category_for_velocity_and_zone():
    assert event_category(_event("velocidade", None)) == "Corpo"
    assert event_category(_event("zona_critica", None)) == "Zona de risco"


# ── description includes id + category + interval ─────────────────────


def test_event_description_prepends_id_and_category():
    event = _event("postura", "cotovelo_direito", t_inicio=5.2)
    event["t_fim"] = 6.4
    event["event_id"] = "#V03"

    text = _event_description(event)

    assert text.startswith("#V03 [Braços] ")
    assert "Cotovelo direito" in text
    assert "entre 5.2s e 6.4s" in text


def test_alert_descriptions_contain_id_and_category_and_interval():
    result = analyze(_multi_event_frames(), threshold=1.3, window=6)

    for event, alert in zip(result["events"], result["alerts"]):
        assert event["event_id"] in alert.description
        assert f"[{event_category(event)}]" in alert.description
        assert "entre" in alert.description and "s e " in alert.description
