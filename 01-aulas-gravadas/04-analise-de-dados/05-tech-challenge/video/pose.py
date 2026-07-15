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

The multi-joint tracking (``JOINTS`` + the neck angle) and the
center-of-mass velocity generalization are described in the redesign
change ``redesenho-aba-video-postura`` (design.md D1): instead of a
single tracked point (right elbow + right wrist), the extraction layer
now produces one angle series per body joint (both sides) plus a
head/neck angle, and a single global "brusque movement" velocity signal
from the approximate center of mass. Each frame also retains its raw
keypoints so a later layer can draw the pose skeleton over the frame.

Spec: openspec/changes/monitoramento-multimodal-pacientes/specs/video-motion-analysis/spec.md
Spec: openspec/changes/redesenho-aba-video-postura/specs/video-motion-analysis/spec.md
"""
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

# Body joints tracked per frame, as the exact keypoint triplets
# ``(a, vertex, b)`` from design.md D1 of the redesign change. Each is a
# plain triplet consumed by the generic ``compute_joint_angle`` — both
# body sides are tracked so a postural irregularity anywhere (not just
# the right arm, as before) is picked up. The head/neck angle is NOT a
# plain triplet (it needs computed midpoints) and is handled separately
# by ``compute_neck_angle`` under the ``NECK_JOINT`` name.
JOINTS: Dict[str, Tuple[str, str, str]] = {
    "cotovelo_esquerdo": ("left_shoulder", "left_elbow", "left_wrist"),
    "cotovelo_direito": ("right_shoulder", "right_elbow", "right_wrist"),
    "joelho_esquerdo": ("left_hip", "left_knee", "left_ankle"),
    "joelho_direito": ("right_hip", "right_knee", "right_ankle"),
    "quadril_esquerdo": ("left_shoulder", "left_hip", "left_knee"),
    "quadril_direito": ("right_shoulder", "right_hip", "right_knee"),
}

# Name of the head/neck angle in the per-frame ``angles`` dict. Kept as a
# constant so callers (analysis/UI) reference it symbolically.
NECK_JOINT = "pescoco"

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


def compute_neck_angle(keypoints_xy: np.ndarray) -> Optional[float]:
    """Head/neck angle from computed midpoints (design.md D1).

    Unlike the ``JOINTS`` entries, the neck angle is not a plain keypoint
    triplet: the vertex is the *shoulder midpoint* (mean of both
    shoulders) and the angle is measured between the trunk vertical
    (shoulder midpoint → hip midpoint, hip midpoint being the mean of both
    hips) and the direction shoulder midpoint → nose. This captures a
    dropped or tilted head. A perfectly upright head (nose straight up,
    opposed to the downward trunk vector) gives ~180°; a sideways tilt
    drops the angle toward 90°.

    Args:
        keypoints_xy: Array of shape ``(17, 2)`` with COCO keypoint (x, y)
            coordinates for a single detected person.

    Returns:
        Angle in degrees in ``[0, 180]``, or ``None`` if any required
        keypoint (nose, both shoulders, both hips) is missing/undetected,
        or the geometry is degenerate (zero-length vector).
    """
    nose = _keypoint_xy(keypoints_xy, "nose")
    left_shoulder = _keypoint_xy(keypoints_xy, "left_shoulder")
    right_shoulder = _keypoint_xy(keypoints_xy, "right_shoulder")
    left_hip = _keypoint_xy(keypoints_xy, "left_hip")
    right_hip = _keypoint_xy(keypoints_xy, "right_hip")
    if any(p is None for p in (nose, left_shoulder, right_shoulder, left_hip, right_hip)):
        return None

    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    hip_mid = (left_hip + right_hip) / 2.0

    trunk_vec = hip_mid - shoulder_mid  # trunk vertical, pointing down
    head_vec = nose - shoulder_mid  # direction to the head
    norm_trunk = np.linalg.norm(trunk_vec)
    norm_head = np.linalg.norm(head_vec)
    if norm_trunk == 0 or norm_head == 0:
        return None

    cos_angle = float(np.dot(trunk_vec, head_vec) / (norm_trunk * norm_head))
    cos_angle = max(-1.0, min(1.0, cos_angle))  # clamp for float rounding
    return math.degrees(math.acos(cos_angle))


def compute_center_of_mass(keypoints_xy: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Approximate center of mass = mean of the *detected* keypoints.

    Missing keypoints (collapsed to the origin ``(0, 0)`` by ultralytics,
    see ``_MISSING_KEYPOINT``) are excluded so they do not drag the mean
    toward the origin.

    Args:
        keypoints_xy: Array of shape ``(17, 2)`` for a single detected
            person, or ``None`` if no person was detected.

    Returns:
        A length-2 ``np.ndarray`` ``[x, y]`` of the mean detected point,
        or ``None`` if no keypoints were detected (all at the origin).
    """
    if keypoints_xy is None:
        return None
    points = np.asarray(keypoints_xy, dtype=float)
    detected_mask = ~np.all(np.isclose(points, _MISSING_KEYPOINT), axis=1)
    if not np.any(detected_mask):
        return None
    return points[detected_mask].mean(axis=0)


def compute_center_of_mass_velocity(
    current_keypoints_xy: Optional[np.ndarray],
    previous_keypoints_xy: Optional[np.ndarray],
) -> Optional[float]:
    """Displacement of the approximate center of mass between two frames.

    Generalizes the previous single-point (right wrist) velocity to a
    global "brusque movement" signal: the euclidean distance between the
    mean of the detected keypoints of ``current`` and of ``previous``
    (design.md D1). Captures sudden whole-body motion rather than only the
    right arm.

    Args:
        current_keypoints_xy: Keypoints of the current frame, or ``None``
            if no person was detected in it.
        previous_keypoints_xy: Keypoints of the previous frame with a
            detected person, or ``None`` if there is no previous frame.

    Returns:
        Displacement in pixel units, or ``None`` when there is no previous
        frame or either frame has no detectable keypoints.
    """
    current_com = compute_center_of_mass(current_keypoints_xy)
    previous_com = compute_center_of_mass(previous_keypoints_xy)
    if current_com is None or previous_com is None:
        return None
    return float(np.linalg.norm(current_com - previous_com))


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


def _compute_frame_angles(
    keypoints_xy: Optional[np.ndarray],
) -> Dict[str, Optional[float]]:
    """Angle of every tracked joint for a single frame.

    Returns a dict keyed by every ``JOINTS`` name plus ``NECK_JOINT``.
    Each value is the computed angle, or ``None`` when the person was not
    detected or that specific joint's required keypoints are missing — a
    missing keypoint for one joint never suppresses the others.
    """
    if keypoints_xy is None:
        return {name: None for name in list(JOINTS) + [NECK_JOINT]}

    angles: Dict[str, Optional[float]] = {
        name: compute_joint_angle(keypoints_xy, triplet) for name, triplet in JOINTS.items()
    }
    angles[NECK_JOINT] = compute_neck_angle(keypoints_xy)
    return angles


def extract_frame_series(
    model: Any,
    frames: Sequence[np.ndarray],
    fps: float,
    on_frame_processed: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Run ``model`` over ``frames`` and derive the per-frame pose series.

    For each frame, records whether a person was detected, the angle of
    every tracked joint (the six ``JOINTS`` plus the ``NECK_JOINT`` head
    angle), the center-of-mass velocity relative to the previous frame
    with a detected person, the raw keypoints (for later skeleton
    drawing), and the object detections from the same forward pass (used
    by the zone-critical-object feature).

    A frame with no detected person does not stop processing: it is
    recorded with ``has_pose = False``, every angle ``None``, ``velocity``
    ``None`` and ``keypoints_xy`` ``None`` (per the spec's "frame sem
    pessoa detectada" scenario).

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
        ``angles`` (dict ``{joint_name: float|None}`` over all joints and
        the neck), ``velocity``, ``keypoints_xy`` (a plain list of 17
        ``[x, y]`` pairs when a person is detected, else ``None`` —
        JSON/pickle-friendly, never a numpy array or torch tensor), and
        ``detections``.
    """
    series: List[Dict[str, Any]] = []
    previous_keypoints_xy: Optional[np.ndarray] = None
    total_frames = len(frames)

    for frame_index, frame in enumerate(frames):
        result = model.predict(frame, verbose=False)[0]
        keypoints_xy_all = _to_numpy(result.keypoints.xy)

        has_pose = len(keypoints_xy_all) > 0
        current_keypoints_xy = keypoints_xy_all[0] if has_pose else None

        angles = _compute_frame_angles(current_keypoints_xy)
        velocity = compute_center_of_mass_velocity(current_keypoints_xy, previous_keypoints_xy)
        keypoints_xy = (
            [[float(x), float(y)] for x, y in np.asarray(current_keypoints_xy, dtype=float)]
            if has_pose
            else None
        )

        series.append(
            {
                "timestamp_s": frame_index / fps,
                "has_pose": has_pose,
                "angles": angles,
                "velocity": velocity,
                "keypoints_xy": keypoints_xy,
                "detections": _extract_detections(result),
            }
        )

        if has_pose:
            previous_keypoints_xy = current_keypoints_xy

        if on_frame_processed is not None:
            on_frame_processed(frame_index, total_frames)

    return series
