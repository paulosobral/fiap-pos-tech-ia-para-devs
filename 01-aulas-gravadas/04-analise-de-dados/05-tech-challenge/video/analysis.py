"""Video pipeline: postural anomaly detection, zone-critical-object
alerting, and the combined deviation report for the "Vídeo" tab.

Two independent detection paths feed the same deviation report, per the
video-motion-analysis spec:

1. Postural anomaly: rolling z-score (``anomaly.zscore.detect_anomalies``)
   over the per-frame joint-angle and movement-velocity series produced
   by ``video.pose.extract_frame_series``. The ``threshold`` parameter is
   user-adjustable via the Streamlit sensitivity slider (design.md D4).
2. Zone-critical-object: a configurable rectangular zone in relative
   frame coordinates (``x_min, y_min, x_max, y_max``, each in ``[0, 1]``)
   is checked against every object bounding box detected in the same
   YOLOv8-pose forward pass. Any intersection raises an immediate
   ``Alert``, independent of the z-score threshold above.

Documented choice on "critical objects": the pose-specific weights
(``yolov8n-pose.pt``) are trained on a single object class, "person"
(COCO keypoints dataset), so ``result.boxes`` from that same forward pass
only ever contains person detections. For this demo, the zone rule is
therefore defined as "a **person** entering the configured critical
zone" (e.g. a restricted area of an operating room) rather than a
non-person object (e.g. a scalpel) — plain ``yolov8n.pt`` general object
detection (80 COCO classes) would be needed to alert on non-person
objects, which design.md D1 explicitly avoided adding as a second model
to keep a single forward pass. ``ZONE_CRITICAL_CLASSES`` is kept
configurable (by class id) so a future revision can swap in the general
detector without changing this module's zone-intersection logic.

Spec: openspec/changes/monitoramento-multimodal-pacientes/specs/video-motion-analysis/spec.md
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

# Object class ids (COCO, as produced by yolov8n-pose.pt's single-class
# person detector) considered "critical" when found inside the
# configured zone. Only "person" (id 0) is available from the pose
# model's own forward pass — see module docstring.
ZONE_CRITICAL_CLASSES = {0}

# Default critical zone in relative frame coordinates
# (x_min, y_min, x_max, y_max), each in [0, 1]. This module never applies
# it implicitly — ``analyze``'s ``zone`` parameter still defaults to
# ``None``, meaning "skip the zone-critical-object path entirely" for
# callers that don't pass a zone. ``DEFAULT_ZONE`` exists so callers that
# *do* want a pre-filled starting zone (e.g. the Vídeo tab's zone
# controls in app.py) share one canonical value instead of hard-coding
# it again. The right-hand-fifth-of-frame value below matches the
# bundled demo video's framing of the critical area.
DEFAULT_ZONE: Tuple[float, float, float, float] = (0.7, 0.0, 1.0, 1.0)

_CLASS_LABELS = {0: "person"}

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


def suggest_sensitivity_threshold(
    frame_series: List[Dict[str, Any]], window: int = DEFAULT_WINDOW
) -> float:
    """Suggest a starting sensitivity threshold from this video's own motion.

    Computes the rolling z-score of the angle and velocity series (same
    series used by ``analyze``'s postural-anomaly path) and suggests a
    threshold near the top of the observed |z-score| distribution for
    this specific video (see ``_SUGGESTION_PERCENTILE``), clamped to the
    Vídeo tab's slider range. This gives each video a starting point
    calibrated to its own natural amount of movement noise, instead of
    one fixed value for every video.

    Args:
        frame_series: Per-frame dicts as produced by
            ``video.pose.extract_frame_series``.
        window: Rolling window size, matching whatever will be passed to
            ``analyze`` for the real detection.

    Returns:
        Suggested threshold in ``[_SUGGESTION_MIN, _SUGGESTION_MAX]``.
        Falls back to ``FALLBACK_SENSITIVITY_THRESHOLD`` when no frame
        has pose data, or when there isn't enough valid data to compute
        any rolling z-score (e.g. fewer than ``window`` frames with pose).
    """
    angle_series = pd.Series([frame["angle"] for frame in frame_series], dtype="float64")
    velocity_series = pd.Series([frame["velocity"] for frame in frame_series], dtype="float64")

    z_values = pd.concat(
        [rolling_zscore(angle_series, window).abs(), rolling_zscore(velocity_series, window).abs()]
    ).dropna()

    if z_values.empty:
        return FALLBACK_SENSITIVITY_THRESHOLD

    suggestion = float(z_values.quantile(_SUGGESTION_PERCENTILE / 100.0))
    if pd.isna(suggestion):
        return FALLBACK_SENSITIVITY_THRESHOLD

    return max(_SUGGESTION_MIN, min(_SUGGESTION_MAX, suggestion))


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


def _postural_alert_description(kind: str, value: float, timestamp_s: float, threshold: float) -> str:
    label = "ângulo" if kind == "angle" else "velocidade"
    return (
        f"Anomalia postural ({label}) em t={timestamp_s:.2f}s: valor={value:.2f} "
        f"(|z-score| > {threshold})."
    )


def analyze(
    frame_series: List[Dict[str, Any]],
    threshold: float,
    window: int = DEFAULT_WINDOW,
    zone: Optional[Tuple[float, float, float, float]] = None,
    frame_width: int = 1,
    frame_height: int = 1,
) -> Dict[str, Any]:
    """Run both detection paths over ``frame_series`` and build the report.

    Args:
        frame_series: Per-frame dicts as produced by
            ``video.pose.extract_frame_series`` (keys: ``timestamp_s``,
            ``has_pose``, ``angle``, ``velocity``, ``detections``).
        threshold: Z-score magnitude above which an angle/velocity
            reading is anomalous (user-adjustable sensitivity slider).
        window: Rolling window size for the z-score layer.
        zone: Optional critical zone in relative coordinates
            ``(x_min, y_min, x_max, y_max)``. When ``None``, the
            zone-critical-object path is skipped entirely.
        frame_width: Frame width in pixels (required to interpret
            ``zone`` and detection boxes together; ignored if ``zone`` is
            ``None``).
        frame_height: Frame height in pixels.

    Returns:
        Dict with:
            - ``angle_anomalies`` / ``velocity_anomalies``: boolean
              Series aligned with ``frame_series`` order, ``True`` where
              that frame's angle/velocity is anomalous. Frames without
              pose data are always ``False`` (no data to flag).
            - ``alerts``: list of ``Alert`` objects generated for every
              postural anomaly and every zone-critical intersection
              (also pushed to the shared feed).
            - ``deviation_report``: list of dicts (one per alert), each
              with ``timestamp_s``, ``kind`` (``"postural"`` or
              ``"zona_critica"``) and ``description``, ordered by
              ``timestamp_s`` ascending.
    """
    timestamps = [frame["timestamp_s"] for frame in frame_series]
    angle_series = pd.Series([frame["angle"] for frame in frame_series], dtype="float64")
    velocity_series = pd.Series([frame["velocity"] for frame in frame_series], dtype="float64")

    angle_anomalies = detect_anomalies(angle_series, window=window, threshold=threshold)
    velocity_anomalies = detect_anomalies(velocity_series, window=window, threshold=threshold)

    alerts = []
    deviation_report: List[Dict[str, Any]] = []

    for kind, flags, values in (
        ("angle", angle_anomalies, angle_series),
        ("velocity", velocity_anomalies, velocity_series),
    ):
        for idx in flags[flags].index:
            timestamp_s = timestamps[idx]
            value = values.iloc[idx]
            description = _postural_alert_description(kind, value, timestamp_s, threshold)
            alert = add_alert(origin=ORIGIN, description=description)
            alerts.append(alert)
            deviation_report.append(
                {"timestamp_s": timestamp_s, "kind": "postural", "description": description}
            )

    if zone is not None:
        for frame in frame_series:
            for detection in frame.get("detections", []):
                if detection["cls"] not in ZONE_CRITICAL_CLASSES:
                    continue
                if not box_intersects_zone(detection["xyxy"], zone, frame_width, frame_height):
                    continue
                label = detection.get("label") or _CLASS_LABELS.get(detection["cls"], str(detection["cls"]))
                timestamp_s = frame["timestamp_s"]
                description = (
                    f"Objeto crítico ({label}) detectado na zona configurada em t={timestamp_s:.2f}s."
                )
                alert = add_alert(origin=ORIGIN, description=description)
                alerts.append(alert)
                deviation_report.append(
                    {"timestamp_s": timestamp_s, "kind": "zona_critica", "description": description}
                )

    deviation_report.sort(key=lambda row: row["timestamp_s"])

    return {
        "angle_anomalies": angle_anomalies,
        "velocity_anomalies": velocity_anomalies,
        "alerts": alerts,
        "deviation_report": deviation_report,
    }
