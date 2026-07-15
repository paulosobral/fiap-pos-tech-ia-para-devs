"""Pure rendering of the pose skeleton and the critical-zone rectangle
over a video frame (redesign ``redesenho-aba-video-postura``, design.md D3).

The redesigned "Vídeo" tab presents each irregular event as an annotated
key frame — the frame's pose skeleton drawn over it with the affected
joint highlighted — instead of a raw text list (spec: "Relatório visual de
desvios do vídeo com esqueleto anotado"). This module is that renderer.

It is deliberately a separate module from ``video/pose.py`` (extraction)
and ``video/analysis.py`` (detection): drawing is a distinct
responsibility, testable in isolation with a synthetic frame + mock
keypoints, and free of any ``ultralytics``/Streamlit/analysis dependency.
Only ``cv2`` and ``numpy`` are used, plus the COCO keypoint constants from
``video.pose``.

Spec: openspec/changes/redesenho-aba-video-postura/specs/video-motion-analysis/spec.md
"""
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from video.pose import JOINTS, KEYPOINT_NAMES, NECK_JOINT

# A keypoint at exactly (0, 0) is how ultralytics marks "not detected"
# (zero-confidence points collapse to the origin) — mirrors
# ``video.pose._MISSING_KEYPOINT`` (kept local so this pure renderer does
# not import a private name from ``pose.py``). Such points and any edge
# touching them are not drawn.
_MISSING_XY: Tuple[float, float] = (0.0, 0.0)

# Colors are BGR (the channel order of ``cv2.VideoCapture`` frames and of
# ``cv2`` drawing primitives), so Streamlit can display the result with
# ``channels="BGR"``.
NEUTRAL_COLOR: Tuple[int, int, int] = (0, 255, 0)  # green — non-highlighted skeleton
HIGHLIGHT_COLOR: Tuple[int, int, int] = (0, 0, 255)  # red — affected joint
ZONE_COLOR: Tuple[int, int, int] = (255, 0, 0)  # blue — critical-zone rectangle

_LINE_THICKNESS = 2
_POINT_RADIUS = 3
_ZONE_THICKNESS = 2

# COCO 17-keypoint skeleton as pairs of keypoint indices (into
# ``KEYPOINT_NAMES``): face, shoulders/arms, torso, hips/legs. Standard
# ultralytics/COCO connectivity. Each edge is only drawn when both of its
# endpoints were actually detected.
COCO_SKELETON: List[Tuple[int, int]] = [
    # face
    (0, 1),  # nose - left_eye
    (0, 2),  # nose - right_eye
    (1, 3),  # left_eye - left_ear
    (2, 4),  # right_eye - right_ear
    # arms
    (5, 7),  # left_shoulder - left_elbow
    (7, 9),  # left_elbow - left_wrist
    (6, 8),  # right_shoulder - right_elbow
    (8, 10),  # right_elbow - right_wrist
    # shoulders / torso
    (5, 6),  # left_shoulder - right_shoulder
    (5, 11),  # left_shoulder - left_hip
    (6, 12),  # right_shoulder - right_hip
    # hips / legs
    (11, 12),  # left_hip - right_hip
    (11, 13),  # left_hip - left_knee
    (13, 15),  # left_knee - left_ankle
    (12, 14),  # right_hip - right_knee
    (14, 16),  # right_knee - right_ankle
]


def _is_detected(point: Sequence[float]) -> bool:
    """A keypoint is "detected" unless it sits on the (0, 0) sentinel."""
    return not (float(point[0]) == _MISSING_XY[0] and float(point[1]) == _MISSING_XY[1])


def _px(point: Sequence[float]) -> Tuple[int, int]:
    """Round a keypoint (x, y) to an integer pixel coordinate tuple."""
    return (int(round(float(point[0]))), int(round(float(point[1]))))


def _highlight_edges_and_points(
    highlight_joint: str,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Keypoint-index edges and vertices to highlight for ``highlight_joint``.

    For a plain ``JOINTS`` triplet ``(a, vertex, b)`` the two meeting
    segments ``(a, vertex)`` and ``(vertex, b)`` are highlighted along with
    all three vertices. ``NECK_JOINT`` is handled separately by the caller
    (its segment uses a computed shoulder midpoint, not a single keypoint),
    so it returns no index-based edges here.
    """
    if highlight_joint in JOINTS:
        a, vertex, b = (KEYPOINT_NAMES.index(name) for name in JOINTS[highlight_joint])
        return [(a, vertex), (vertex, b)], [a, vertex, b]
    return [], []


def draw_pose_on_frame(
    frame_bgr: np.ndarray,
    keypoints_xy: Sequence[Sequence[float]],
    highlight_joint: Optional[str] = None,
) -> np.ndarray:
    """Draw the COCO pose skeleton over a copy of ``frame_bgr``.

    Connecting lines and keypoint dots are drawn in ``NEUTRAL_COLOR``.
    Keypoints on the ``(0, 0)`` "not detected" sentinel are skipped, and so
    is any edge touching such a point. When ``highlight_joint`` is given,
    that joint's segment(s) and vertex(es) are redrawn in ``HIGHLIGHT_COLOR``
    (red) on top of the neutral skeleton.

    For the six ``JOINTS`` triplets the two segments meeting at the joint
    vertex are highlighted. For ``NECK_JOINT`` (``"pescoco"``) — which has
    no single-keypoint vertex — the highlight is the *shoulder-midpoint →
    nose* segment (the head/neck direction ``video.pose.compute_neck_angle``
    measures), drawn as a red line with red dots at the midpoint and nose.

    Args:
        frame_bgr: ``(H, W, 3)`` BGR image (as ``cv2.VideoCapture`` yields).
            Never mutated — the annotations are drawn on a copy.
        keypoints_xy: The frame's 17 COCO keypoints as ``[x, y]`` pairs (the
            plain-list shape stored by ``extract_frame_series``).
        highlight_joint: Optional joint key (e.g. ``"joelho_direito"`` or
            ``"pescoco"``) to emphasize in red. ``None`` draws every point
            in the neutral color.

    Returns:
        The annotated copy of ``frame_bgr`` (BGR).
    """
    annotated = frame_bgr.copy()
    points = [list(p) for p in keypoints_xy]

    highlight_edges, highlight_points = (
        _highlight_edges_and_points(highlight_joint) if highlight_joint else ([], [])
    )
    highlight_edge_set = {frozenset(edge) for edge in highlight_edges}

    # 1) Neutral skeleton: every edge whose endpoints are both detected.
    for i, j in COCO_SKELETON:
        if frozenset((i, j)) in highlight_edge_set:
            continue  # drawn red in the highlight pass below
        if _is_detected(points[i]) and _is_detected(points[j]):
            cv2.line(annotated, _px(points[i]), _px(points[j]), NEUTRAL_COLOR, _LINE_THICKNESS)

    # 2) Neutral keypoint dots for every detected point.
    for idx, point in enumerate(points):
        if idx in highlight_points:
            continue  # drawn red below
        if _is_detected(point):
            cv2.circle(annotated, _px(point), _POINT_RADIUS, NEUTRAL_COLOR, -1)

    # 3a) Highlighted joint (triplet): red segments + vertices on top.
    for i, j in highlight_edges:
        if _is_detected(points[i]) and _is_detected(points[j]):
            cv2.line(annotated, _px(points[i]), _px(points[j]), HIGHLIGHT_COLOR, _LINE_THICKNESS)
    for idx in highlight_points:
        if _is_detected(points[idx]):
            cv2.circle(annotated, _px(points[idx]), _POINT_RADIUS, HIGHLIGHT_COLOR, -1)

    # 3b) Highlighted neck/head: shoulder-midpoint -> nose segment in red.
    if highlight_joint == NECK_JOINT:
        left_shoulder = points[KEYPOINT_NAMES.index("left_shoulder")]
        right_shoulder = points[KEYPOINT_NAMES.index("right_shoulder")]
        nose = points[KEYPOINT_NAMES.index("nose")]
        if _is_detected(left_shoulder) and _is_detected(right_shoulder) and _is_detected(nose):
            midpoint = (
                (float(left_shoulder[0]) + float(right_shoulder[0])) / 2.0,
                (float(left_shoulder[1]) + float(right_shoulder[1])) / 2.0,
            )
            cv2.line(annotated, _px(midpoint), _px(nose), HIGHLIGHT_COLOR, _LINE_THICKNESS)
            cv2.circle(annotated, _px(midpoint), _POINT_RADIUS, HIGHLIGHT_COLOR, -1)
            cv2.circle(annotated, _px(nose), _POINT_RADIUS, HIGHLIGHT_COLOR, -1)

    return annotated


def draw_zone_on_frame(
    frame_bgr: np.ndarray,
    zone_rel: Tuple[float, float, float, float],
) -> np.ndarray:
    """Draw the critical-zone rectangle over a copy of ``frame_bgr``.

    Powers the Vídeo tab's zone preview: the user sees where the relative
    zone falls on a real frame before processing (spec: "Prévia da zona
    sobre o frame ao ativar").

    Args:
        frame_bgr: ``(H, W, 3)`` BGR image. Never mutated (drawn on a copy).
        zone_rel: ``(x_min, y_min, x_max, y_max)`` in relative ``[0, 1]``
            coordinates; converted to pixels using the frame's width/height.

    Returns:
        The annotated copy of ``frame_bgr`` with the zone rectangle in
        ``ZONE_COLOR``.
    """
    annotated = frame_bgr.copy()
    height, width = annotated.shape[:2]
    x_min, y_min, x_max, y_max = zone_rel

    pt1 = (int(round(x_min * width)), int(round(y_min * height)))
    pt2 = (int(round(x_max * width)), int(round(y_max * height)))
    cv2.rectangle(annotated, pt1, pt2, ZONE_COLOR, _ZONE_THICKNESS)
    return annotated
