"""Video pipeline: per-joint postural anomaly detection, movement-velocity
anomaly detection, optional zone-critical grouping, and the event-based
report for the "Vídeo" tab.

Redesign ``redesenho-aba-video-postura`` (design.md D2) replaces the old
one-point / one-alert-per-frame behaviour:

1. Postural anomaly, per joint: for every joint tracked by
   ``video.pose.extract_frame_series`` (the six ``JOINTS`` plus the
   ``NECK_JOINT`` head angle) the layer builds that joint's angle time
   series and applies ``anomaly.zscore.detect_anomalies`` (reused, never
   reimplemented). A frame is "irregular for joint X" when the |z-score|
   of X's angle exceeds the user's sensitivity ``threshold``. The global
   center-of-mass ``velocity`` series is analysed the same way.
2. Grouping into events: consecutive irregular frames of the *same*
   origin (a joint, velocity, or the zone) are merged — with a small gap
   tolerance so a single dropped-pose frame does not fragment an event —
   into one event ``{tipo, articulacao, t_inicio, t_fim,
   frame_index_pior, valor_pior, z_pior}``. The "worst" (``pior``) frame
   is the group's max-|z-score| frame (via ``rolling_zscore``); for zone
   events it is the frame of maximum box/zone intersection area. Exactly
   one ``Alert`` (origin "Vídeo") is generated per event.
3. Zone-critical-object: a configurable rectangular zone in relative
   frame coordinates (``x_min, y_min, x_max, y_max``, each in ``[0, 1]``)
   checked against every person box in the same YOLOv8-pose forward pass.
   Run only when ``zone is not None`` (default: skipped entirely); the
   Vídeo tab keeps it off by default.

Documented choice on "critical objects": the pose-specific weights
(``yolov8n-pose.pt``) are trained on a single object class, "person"
(COCO keypoints dataset), so ``result.boxes`` from that same forward pass
only ever contains person detections. For this demo, the zone rule is
therefore defined as "a **person** entering the configured critical
zone" (e.g. a restricted area of an operating room). ``ZONE_CRITICAL_CLASSES``
is kept configurable (by class id) so a future revision can swap in a
general detector without changing this module's zone-intersection logic.

The rolling z-score remains "deviation from THIS video's own recent
behaviour" (not an absolute clinical norm), now computed per joint — see
design.md Non-Goals.

Spec: openspec/changes/monitoramento-multimodal-pacientes/specs/video-motion-analysis/spec.md
Spec: openspec/changes/redesenho-aba-video-postura/specs/video-motion-analysis/spec.md
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from alerts.feed import add_alert
from anomaly.zscore import detect_anomalies, rolling_zscore

ORIGIN = "Vídeo"

# Rolling window for the z-score layer over angle/velocity series. Video
# frame rates are much higher than the vital-signs sampling rate, so a
# slightly larger window than vital_signs' default (6) smooths out
# per-frame detector jitter while still reacting within roughly half a
# second of 10-15 fps demo footage.
DEFAULT_WINDOW = 10

# Maximum number of consecutive non-irregular ("gap") frames tolerated
# inside a single event before it is split into two. A single frame where
# the pose was momentarily lost (or dipped just under threshold) should
# not fragment one continuous irregular movement into several events, so
# indices whose gap is at most this many frames are kept in the same
# group. One frame of tolerance matches design.md D2's "1-2 frames".
GAP_TOLERANCE = 1

# Object class ids (COCO, as produced by yolov8n-pose.pt's single-class
# person detector) considered "critical" when found inside the
# configured zone. Only "person" (id 0) is available from the pose
# model's own forward pass — see module docstring.
ZONE_CRITICAL_CLASSES = {0}

# Default critical zone in relative frame coordinates
# (x_min, y_min, x_max, y_max), each in [0, 1]. ``analyze``'s ``zone``
# parameter still defaults to ``None`` ("skip the zone path entirely");
# ``DEFAULT_ZONE`` exists so callers that *do* want a pre-filled starting
# zone (the Vídeo tab's zone controls in app.py) share one canonical
# value. The right-hand-fifth-of-frame value matches the bundled demo
# video's framing of the critical area.
DEFAULT_ZONE: Tuple[float, float, float, float] = (0.7, 0.0, 1.0, 1.0)

_CLASS_LABELS = {0: "person"}

# Human-readable labels for the internal joint keys (from video.pose's
# ``JOINTS`` / ``NECK_JOINT``), used in plain-language alert descriptions
# and the UI. Kept here (not imported from video.pose) so the analysis /
# presentation vocabulary lives with the layer that presents it.
JOINT_LABELS: Dict[str, str] = {
    "cotovelo_esquerdo": "Cotovelo esquerdo",
    "cotovelo_direito": "Cotovelo direito",
    "joelho_esquerdo": "Joelho esquerdo",
    "joelho_direito": "Joelho direito",
    "quadril_esquerdo": "Quadril esquerdo",
    "quadril_direito": "Quadril direito",
    "pescoco": "Pescoço",
}

# Label used for the global velocity ("brusque movement") series, which
# is not tied to a single joint (``articulacao is None``).
VELOCITY_LABEL = "Movimento brusco"

# Fallback sensitivity suggestion when no series has enough valid data to
# compute one (e.g. no frame in the video has a detected pose) — matches
# the slider's own hard-coded default in app.py before this change.
FALLBACK_SENSITIVITY_THRESHOLD = 2.0

# Percentile of the observed |z-score| distribution used as the
# suggested threshold. Chosen so the suggestion flags roughly the top
# 5% most unusual frames of THIS video as a starting point — sensitive
# enough to be useful, not so sensitive that a naturally variable but
# healthy movement floods the report on first upload.
_SUGGESTION_PERCENTILE = 95.0

# Slider bounds from the Vídeo tab (app.py) — the suggestion is clamped
# to this range so it's always a valid starting value for that control.
_SUGGESTION_MIN = 0.5
_SUGGESTION_MAX = 5.0


def joint_label(joint_key: Optional[str]) -> str:
    """Human-readable label for an internal joint key.

    Args:
        joint_key: An internal joint key (e.g. ``"joelho_direito"``), or
            ``None`` for the velocity/movement series.

    Returns:
        A non-empty human label (e.g. ``"Joelho direito"``). Falls back to
        ``VELOCITY_LABEL`` for ``None`` and to the raw key for any
        unmapped joint.
    """
    if joint_key is None:
        return VELOCITY_LABEL
    return JOINT_LABELS.get(joint_key, joint_key)


def _joint_names(frame_series: List[Dict[str, Any]]) -> List[str]:
    """Every joint key present in ``frame_series``, in a stable order.

    The union of keys across every frame's ``angles`` dict; ordered by
    ``JOINT_LABELS`` (known joints first, in their canonical order) then
    any extra keys alphabetically, so the output is deterministic.
    """
    seen = set()
    for frame in frame_series:
        seen.update(frame.get("angles", {}).keys())
    known = [name for name in JOINT_LABELS if name in seen]
    extra = sorted(seen - set(JOINT_LABELS))
    return known + extra


def _series_for_joint(frame_series: List[Dict[str, Any]], joint: str) -> pd.Series:
    """Time series of one joint's angle, missing frames as ``NaN``."""
    return pd.Series(
        [frame.get("angles", {}).get(joint) for frame in frame_series], dtype="float64"
    )


def _velocity_series(frame_series: List[Dict[str, Any]]) -> pd.Series:
    return pd.Series([frame.get("velocity") for frame in frame_series], dtype="float64")


def _group_consecutive(indices: Sequence[int], gap_tolerance: int = GAP_TOLERANCE) -> List[List[int]]:
    """Group sorted indices into runs, tolerating small gaps.

    Two consecutive (in ``indices``) positions belong to the same group
    when at most ``gap_tolerance`` frames are missing between them, i.e.
    ``next - current <= gap_tolerance + 1``. A larger jump starts a new
    group. This keeps one continuous irregular movement as a single
    event even if the pose dropped (or dipped under threshold) for a
    frame or two.

    Args:
        indices: Sorted, unique frame indices flagged as irregular.
        gap_tolerance: Max number of non-flagged frames tolerated inside a
            group.

    Returns:
        List of groups, each a list of the original indices, in order.
    """
    groups: List[List[int]] = []
    current: List[int] = []
    for idx in indices:
        if current and idx - current[-1] <= gap_tolerance + 1:
            current.append(idx)
        else:
            if current:
                groups.append(current)
            current = [idx]
    if current:
        groups.append(current)
    return groups


def _build_series_events(
    frame_series: List[Dict[str, Any]],
    values: pd.Series,
    z_abs: pd.Series,
    z_raw: pd.Series,
    threshold: float,
    window: int,
    tipo: str,
    articulacao: Optional[str],
) -> List[Dict[str, Any]]:
    """Group a single (joint or velocity) series' irregular frames into events.

    Args:
        frame_series: The full per-frame series (for timestamps).
        values: The underlying value series (angle or velocity).
        z_abs / z_raw: ``|z|`` and signed z magnitudes (from
            ``rolling_zscore``), used to pick the worst frame per group.
        threshold: Sensitivity threshold.
        window: Rolling window (for ``detect_anomalies``).
        tipo: ``"postura"`` or ``"velocidade"``.
        articulacao: Joint key for postura events, ``None`` for velocity.

    Returns:
        One event dict per group of consecutive irregular frames.
    """
    flags = detect_anomalies(values, window=window, threshold=threshold)
    anomalous = list(flags[flags].index)
    events: List[Dict[str, Any]] = []
    for group in _group_consecutive(anomalous):
        # Worst frame = max |z-score| within the group.
        worst = max(group, key=lambda i: (z_abs.iloc[i] if pd.notna(z_abs.iloc[i]) else 0.0))
        events.append(
            {
                "tipo": tipo,
                "articulacao": articulacao,
                "t_inicio": float(frame_series[group[0]]["timestamp_s"]),
                "t_fim": float(frame_series[group[-1]]["timestamp_s"]),
                "frame_index_pior": int(worst),
                "valor_pior": float(values.iloc[worst]),
                "z_pior": float(z_raw.iloc[worst]),
            }
        )
    return events


def suggest_sensitivity_threshold(
    frame_series: List[Dict[str, Any]], window: int = DEFAULT_WINDOW
) -> float:
    """Suggest a starting sensitivity threshold from this video's own motion.

    Computes the rolling z-score of **every** tracked joint's angle series
    and the velocity series (the same series ``analyze`` uses), pools their
    |z-score| magnitudes, and suggests a threshold near the top of that
    combined distribution for this specific video (see
    ``_SUGGESTION_PERCENTILE``), clamped to the Vídeo tab's slider range.
    Pooling all joints (not just one) means the suggestion reflects the
    most variable part of the body, matching the multi-joint detection.

    Args:
        frame_series: Per-frame dicts from ``video.pose.extract_frame_series``.
        window: Rolling window size, matching what will be passed to
            ``analyze``.

    Returns:
        Suggested threshold in ``[_SUGGESTION_MIN, _SUGGESTION_MAX]``.
        Falls back to ``FALLBACK_SENSITIVITY_THRESHOLD`` when there isn't
        enough valid data to compute any rolling z-score.
    """
    series_list = [_series_for_joint(frame_series, joint) for joint in _joint_names(frame_series)]
    series_list.append(_velocity_series(frame_series))

    z_parts = [rolling_zscore(series, window).abs() for series in series_list]
    z_values = pd.concat(z_parts).dropna() if z_parts else pd.Series(dtype="float64")

    if z_values.empty:
        return FALLBACK_SENSITIVITY_THRESHOLD

    suggestion = float(z_values.quantile(_SUGGESTION_PERCENTILE / 100.0))
    if pd.isna(suggestion):
        return FALLBACK_SENSITIVITY_THRESHOLD

    return max(_SUGGESTION_MIN, min(_SUGGESTION_MAX, suggestion))


def estimate_flagged_fraction(
    frame_series: List[Dict[str, Any]], threshold: float, window: int = DEFAULT_WINDOW
) -> float:
    """Approximate fraction of frames that would be marked irregular.

    A frame counts as flagged if **any** tracked joint's angle or the
    velocity exceeds ``threshold`` at that frame (logical OR across all
    series). Powers the Vídeo tab's "~X% do vídeo" sensitivity caption
    (spec: "Feedback do efeito da sensibilidade escolhida").

    Args:
        frame_series: Per-frame dicts from ``video.pose.extract_frame_series``.
        threshold: Z-score magnitude above which a reading is irregular.
        window: Rolling window size (matching ``analyze``).

    Returns:
        Fraction in ``[0, 1]``: flagged frames / total frames. ``0.0``
        for an empty series.
    """
    total = len(frame_series)
    if total == 0:
        return 0.0

    flagged = pd.Series(False, index=range(total), dtype=bool)
    for joint in _joint_names(frame_series):
        flagged |= detect_anomalies(_series_for_joint(frame_series, joint), window, threshold)
    flagged |= detect_anomalies(_velocity_series(frame_series), window, threshold)

    return float(flagged.sum()) / total


def box_intersects_zone(
    box_xyxy: Sequence[float],
    zone: Tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> bool:
    """Whether a detection's bounding box intersects the critical zone.

    Args:
        box_xyxy: Detection bounding box in absolute pixel coordinates
            ``(x1, y1, x2, y2)``.
        zone: Critical zone in relative coordinates
            ``(x_min, y_min, x_max, y_max)``, each in ``[0, 1]``.
        frame_width: Frame width in pixels, used to convert ``zone`` to
            absolute coordinates.
        frame_height: Frame height in pixels.

    Returns:
        ``True`` if the two axis-aligned rectangles overlap (including
        touching at the boundary), ``False`` otherwise.
    """
    zone_x1 = zone[0] * frame_width
    zone_y1 = zone[1] * frame_height
    zone_x2 = zone[2] * frame_width
    zone_y2 = zone[3] * frame_height

    box_x1, box_y1, box_x2, box_y2 = box_xyxy

    no_overlap = box_x2 < zone_x1 or box_x1 > zone_x2 or box_y2 < zone_y1 or box_y1 > zone_y2
    return not no_overlap


def _zone_intersection_area(
    box_xyxy: Sequence[float],
    zone: Tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> float:
    """Overlap area (pixels²) between a detection box and the zone, else 0."""
    zone_abs = (
        zone[0] * frame_width,
        zone[1] * frame_height,
        zone[2] * frame_width,
        zone[3] * frame_height,
    )
    box_x1, box_y1, box_x2, box_y2 = box_xyxy
    ix1 = max(box_x1, zone_abs[0])
    iy1 = max(box_y1, zone_abs[1])
    ix2 = min(box_x2, zone_abs[2])
    iy2 = min(box_y2, zone_abs[3])
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def _zone_events(
    frame_series: List[Dict[str, Any]],
    zone: Tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> List[Dict[str, Any]]:
    """Group consecutive zone-intersecting frames into ``zona_critica`` events.

    A frame intersects the zone when any of its "critical" detections
    (class in ``ZONE_CRITICAL_CLASSES``) overlaps the zone (including a
    zero-area boundary touch, per ``box_intersects_zone``). Consecutive
    intersecting frames (same ``_group_consecutive`` gap tolerance) become
    one event; the worst frame is the one of maximum overlap area, exposed
    as ``valor_pior`` (``z_pior`` is ``nan`` — zone events have no z-score).
    """
    per_frame_area: Dict[int, float] = {}
    for idx, frame in enumerate(frame_series):
        best_area: Optional[float] = None
        for detection in frame.get("detections", []):
            if detection["cls"] not in ZONE_CRITICAL_CLASSES:
                continue
            if not box_intersects_zone(detection["xyxy"], zone, frame_width, frame_height):
                continue
            area = _zone_intersection_area(detection["xyxy"], zone, frame_width, frame_height)
            best_area = area if best_area is None else max(best_area, area)
        if best_area is not None:
            per_frame_area[idx] = best_area

    intersecting = sorted(per_frame_area)
    events: List[Dict[str, Any]] = []
    for group in _group_consecutive(intersecting):
        worst = max(group, key=lambda i: per_frame_area[i])
        events.append(
            {
                "tipo": "zona_critica",
                "articulacao": None,
                "t_inicio": float(frame_series[group[0]]["timestamp_s"]),
                "t_fim": float(frame_series[group[-1]]["timestamp_s"]),
                "frame_index_pior": int(worst),
                "valor_pior": float(per_frame_area[worst]),
                "z_pior": float("nan"),
            }
        )
    return events


def _event_description(event: Dict[str, Any]) -> str:
    """Plain-language alert description for one event (design.md D2)."""
    interval = f"entre {event['t_inicio']:.1f}s e {event['t_fim']:.1f}s"
    if event["tipo"] == "postura":
        return f"{joint_label(event['articulacao'])} irregular {interval}."
    if event["tipo"] == "velocidade":
        return f"{VELOCITY_LABEL} irregular {interval}."
    # zona_critica
    return f"Pessoa na zona de risco {interval}."


def _summarize(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the report summary: event count and most-affected joint.

    Most-affected joint rule (documented): among ``postura`` events, the
    joint with the most events; ties broken by the joint's total span of
    irregular frames (sum of ``t_fim - t_inicio``), then by label for
    determinism. ``None`` when there are no postural events (velocity and
    zone events are not tied to a joint).
    """
    postura = [e for e in events if e["tipo"] == "postura"]
    if not postura:
        return {"total_events": len(events), "most_affected_joint": None, "most_affected_label": None}

    stats: Dict[str, Tuple[int, float]] = {}
    for e in postura:
        joint = e["articulacao"]
        count, span = stats.get(joint, (0, 0.0))
        stats[joint] = (count + 1, span + (e["t_fim"] - e["t_inicio"]))

    def rank(item: Tuple[str, Tuple[int, float]]):
        joint, (count, span) = item
        return (count, span, joint)

    most_joint = max(stats.items(), key=rank)[0]
    return {
        "total_events": len(events),
        "most_affected_joint": most_joint,
        "most_affected_label": joint_label(most_joint),
    }


def analyze(
    frame_series: List[Dict[str, Any]],
    threshold: float,
    window: int = DEFAULT_WINDOW,
    zone: Optional[Tuple[float, float, float, float]] = None,
    frame_width: int = 1,
    frame_height: int = 1,
) -> Dict[str, Any]:
    """Detect per-joint / velocity / zone irregular events and build the report.

    Applies ``detect_anomalies`` to every tracked joint's angle series and
    to the velocity series, groups consecutive irregular frames into
    events (design.md D2), optionally does the same for zone
    intersections, and generates exactly one ``Alert`` per event.

    Args:
        frame_series: Per-frame dicts from ``video.pose.extract_frame_series``
            (keys: ``timestamp_s``, ``has_pose``, ``angles`` dict,
            ``velocity``, ``keypoints_xy``, ``detections``).
        threshold: Z-score magnitude above which an angle/velocity reading
            is irregular (user-adjustable sensitivity slider).
        window: Rolling window size for the z-score layer.
        zone: Optional critical zone in relative coordinates
            ``(x_min, y_min, x_max, y_max)``. When ``None``, the zone path
            is skipped entirely (no zone events).
        frame_width: Frame width in pixels (to interpret ``zone`` and
            detection boxes; ignored when ``zone`` is ``None``).
        frame_height: Frame height in pixels.

    Returns:
        Dict with:
            - ``events``: list of event dicts (``tipo``, ``articulacao``,
              ``t_inicio``, ``t_fim``, ``frame_index_pior``, ``valor_pior``,
              ``z_pior``), sorted by ``t_inicio`` ascending.
            - ``alerts``: list of ``Alert`` objects, one per event (also
              pushed to the shared feed), same order as ``events``.
            - ``summary``: dict with ``total_events``,
              ``most_affected_joint`` and ``most_affected_label``.
    """
    events: List[Dict[str, Any]] = []

    # Postural events, per joint.
    for joint in _joint_names(frame_series):
        values = _series_for_joint(frame_series, joint)
        z_raw = rolling_zscore(values, window)
        events.extend(
            _build_series_events(
                frame_series, values, z_raw.abs(), z_raw, threshold, window, "postura", joint
            )
        )

    # Velocity ("brusque movement") events.
    velocity = _velocity_series(frame_series)
    v_z_raw = rolling_zscore(velocity, window)
    events.extend(
        _build_series_events(
            frame_series, velocity, v_z_raw.abs(), v_z_raw, threshold, window, "velocidade", None
        )
    )

    # Zone-critical events (opt-in).
    if zone is not None:
        events.extend(_zone_events(frame_series, zone, frame_width, frame_height))

    events.sort(key=lambda e: (e["t_inicio"], e["tipo"], e["articulacao"] or ""))

    alerts = [add_alert(origin=ORIGIN, description=_event_description(event)) for event in events]

    return {
        "events": events,
        "alerts": alerts,
        "summary": _summarize(events),
    }
