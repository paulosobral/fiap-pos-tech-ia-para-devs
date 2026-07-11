"""YOLOv8-pose keypoint extraction, joint-angle and velocity calculation.

Per design.md D1, a single ``yolov8n-pose.pt`` model (loaded via
``ultralytics``) is used for both human pose keypoints and object
detection in the same forward pass — the pose model's ``Results`` object
exposes both ``.keypoints`` (17 COCO keypoints per detected person) and
``.boxes`` (bounding boxes with a class id per detection, restricted to
the "person" class since the pose-specific weights are only trained on
that one class). ``video/analysis.py`` uses the zone-critical-object
feature against these person detections; a separate plain-detection
model is not needed for that requirement — see the module docstring in
``video/analysis.py`` for the documented trade-off this implies.

Spec: openspec/changes/monitoramento-multimodal-pacientes/specs/video-motion-analysis/spec.md
"""
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# COCO 17-keypoint order, as produced by ultralytics pose models.
KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Joint chosen for angle calculation: right elbow (shoulder-elbow-wrist).
# Documented choice (design.md D1/D4 leave the exact joint underspecified):
# the elbow is chosen over the knee because it is visible in a wider
# variety of camera framings for a physiotherapy/exercise demo video
# (upper-body shots are common; full-body shots showing the knee are not
# guaranteed). Any of the two would satisfy the spec's "articulação
# clinicamente relevante" requirement equally well.
RIGHT_ELBOW_TRIPLET = ("right_shoulder", "right_elbow", "right_wrist")

# Reference point used for the movement-velocity series: the wrist, the
# distal end of the same limb whose angle is tracked at the elbow. The
# wrist travels further per unit of joint rotation than the elbow itself
# (which is closer to the torso and moves less), so it is a more
# sensitive/representative point for detecting movement-velocity
# anomalies (e.g. a sudden, uncontrolled arm motion).
VELOCITY_REFERENCE_POINT = "right_wrist"

# A keypoint at exactly (0, 0) is how ultralytics represents "not
# detected" for a given point in an otherwise-detected person (zero
# confidence keypoints collapse to the origin). Treat it as missing.
_MISSING_KEYPOINT = np.array([0.0, 0.0])


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a torch Tensor (possibly on GPU) or array-like to numpy.

    ``ultralytics`` results carry torch Tensors that may live on a CUDA
    device; ``np.asarray``/``np.array`` cannot convert those directly
    (raises ``TypeError``), so tensors are moved to CPU first via
    ``.cpu().numpy()`` when available.
    """
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _keypoint_xy(keypoints_xy: np.ndarray, name: str) -> Optional[np.ndarray]:
    point = np.asarray(keypoints_xy[KEYPOINT_NAMES.index(name)], dtype=float)
    if np.allclose(point, _MISSING_KEYPOINT):
        return None
    return point


def compute_joint_angle(keypoints_xy: np.ndarray, triplet: Sequence[str]) -> Optional[float]:
    """Compute the angle (in degrees) at the middle point of ``triplet``.

    Args:
        keypoints_xy: Array of shape ``(17, 2)`` with COCO keypoint (x, y)
            coordinates for a single detected person.
        triplet: Three keypoint names ``(a, vertex, b)``; the angle is
            measured at ``vertex`` between rays to ``a`` and ``b``.

    Returns:
        Angle in degrees in ``[0, 180]``, or ``None`` if any of the three
        keypoints is missing (not detected) or the vertex coincides with
        one of the endpoints (degenerate triangle).
    """
    a_name, vertex_name, b_name = triplet
    a = _keypoint_xy(keypoints_xy, a_name)
    vertex = _keypoint_xy(keypoints_xy, vertex_name)
    b = _keypoint_xy(keypoints_xy, b_name)
    if a is None or vertex is None or b is None:
        return None

    vec_a = a - vertex
    vec_b = b - vertex
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return None

    cos_angle = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    cos_angle = max(-1.0, min(1.0, cos_angle))  # clamp for float rounding
    return math.degrees(math.acos(cos_angle))


def compute_velocity(
    current_keypoints_xy: Optional[np.ndarray],
    previous_keypoints_xy: Optional[np.ndarray],
    reference_point: str = VELOCITY_REFERENCE_POINT,
) -> Optional[float]:
    """Euclidean displacement of ``reference_point`` between two frames.

    Args:
        current_keypoints_xy: Keypoints of the current frame, or ``None``
            if no person was detected in the current frame.
        previous_keypoints_xy: Keypoints of the previous frame with a
            detected person, or ``None`` if there is no previous frame
            (first frame) or no person was detected in it.
        reference_point: Keypoint name whose displacement is measured.

    Returns:
        Displacement in pixel units, or ``None`` if either frame lacks
        the reference point or there is no previous frame to compare to.
    """
    if current_keypoints_xy is None or previous_keypoints_xy is None:
        return None

    current_point = _keypoint_xy(current_keypoints_xy, reference_point)
    previous_point = _keypoint_xy(previous_keypoints_xy, reference_point)
    if current_point is None or previous_point is None:
        return None

    return float(np.linalg.norm(current_point - previous_point))


def _extract_detections(result: Any) -> List[Dict[str, Any]]:
    """Turn a single frame's ``result.boxes`` into plain dicts.

    Kept independent of keypoints so the same structure could later be
    populated from a separate general-purpose detector if the zone-alert
    feature needs classes beyond "person" (see analysis.py docstring).
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    xyxy = _to_numpy(boxes.xyxy)
    cls = _to_numpy(boxes.cls)
    detections = []
    for box, class_id in zip(xyxy, cls):
        detections.append({"cls": int(class_id), "xyxy": [float(v) for v in box]})
    return detections


def extract_frame_series(
    model: Any,
    frames: Sequence[np.ndarray],
    fps: float,
    on_frame_processed: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Run ``model`` over ``frames`` and derive the per-frame pose series.

    For each frame, records whether a person was detected, the right-elbow
    angle, the velocity of the wrist relative to the previous frame with a
    detected person, and the raw object detections from the same forward
    pass (used by the zone-critical-object feature).

    A frame with no detected person does not stop processing: it is
    recorded with ``has_pose = False`` and ``angle`` / ``velocity`` set to
    ``None`` (per the spec's "frame sem pessoa detectada" scenario).

    Args:
        model: A loaded pose model exposing ``.predict(frame, verbose=False)``
            and returning a list with one ``Results``-like object per
            frame (the real ``ultralytics.YOLO`` model, or a test double).
        frames: Sequence of frames (as read by e.g. ``cv2.VideoCapture``).
        fps: Frames per second of the source video, used to derive each
            frame's timestamp in seconds.
        on_frame_processed: Optional callback invoked as
            ``on_frame_processed(frame_index, total_frames)`` after each
            frame is processed, so callers (e.g. the Streamlit UI) can
            show progress without this module depending on Streamlit.

    Returns:
        List of per-frame dicts with keys ``timestamp_s``, ``has_pose``,
        ``angle``, ``velocity``, ``detections``.
    """
    series: List[Dict[str, Any]] = []
    previous_keypoints_xy: Optional[np.ndarray] = None
    total_frames = len(frames)

    for frame_index, frame in enumerate(frames):
        result = model.predict(frame, verbose=False)[0]
        keypoints_xy_all = _to_numpy(result.keypoints.xy)

        has_pose = len(keypoints_xy_all) > 0
        current_keypoints_xy = keypoints_xy_all[0] if has_pose else None

        angle = compute_joint_angle(current_keypoints_xy, RIGHT_ELBOW_TRIPLET) if has_pose else None
        velocity = compute_velocity(current_keypoints_xy, previous_keypoints_xy)

        series.append(
            {
                "timestamp_s": frame_index / fps,
                "has_pose": has_pose,
                "angle": angle,
                "velocity": velocity,
                "detections": _extract_detections(result),
            }
        )

        if has_pose:
            previous_keypoints_xy = current_keypoints_xy

        if on_frame_processed is not None:
            on_frame_processed(frame_index, total_frames)

    return series
