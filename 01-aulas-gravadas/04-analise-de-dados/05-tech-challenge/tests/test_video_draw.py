"""Tests for video/draw.py — pure cv2/numpy rendering of the pose skeleton
and the critical-zone rectangle over a video frame.

Covers the redesign spec's "Relatório visual de desvios do vídeo com
esqueleto anotado" requirement (design.md D3): the UI draws annotated key
frames instead of a raw text list. These tests use only small synthetic
frames (``np.zeros``) and hand-built keypoint lists — no real video or
YOLOv8-pose model.

Spec: openspec/changes/redesenho-aba-video-postura/specs/video-motion-analysis/spec.md
"""
import numpy as np

from video.draw import draw_pose_on_frame, draw_zone_on_frame
from video.pose import KEYPOINT_NAMES

RED = (0, 0, 255)  # BGR highlight color


def _keypoints(**overrides):
    """Build a 17-element list of ``[x, y]`` (COCO order), all missing at
    the origin except the ones overridden by keypoint name."""
    points = [[0.0, 0.0] for _ in KEYPOINT_NAMES]
    for name, xy in overrides.items():
        points[KEYPOINT_NAMES.index(name)] = [float(xy[0]), float(xy[1])]
    return points


def _has_color(region, color):
    return bool(np.any(np.all(region == np.array(color), axis=-1)))


# ── draw_pose_on_frame: copy-not-mutate ──────────────────────────────


def test_draw_pose_does_not_mutate_original_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    original = frame.copy()
    keypoints = _keypoints(left_shoulder=(50, 50), right_shoulder=(70, 50))

    result = draw_pose_on_frame(frame, keypoints)

    # caller's array is untouched, and something was actually drawn
    assert np.array_equal(frame, original)
    assert not np.array_equal(result, original)


# ── draw_pose_on_frame: missing keypoints (0,0) are not drawn ─────────


def test_missing_keypoints_at_origin_are_not_drawn():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # two detected, connected keypoints (torso edge); everything else at
    # the (0, 0) "missing" sentinel and must stay invisible.
    keypoints = _keypoints(left_shoulder=(50, 50), right_shoulder=(70, 50))

    result = draw_pose_on_frame(frame, keypoints)

    # a detected keypoint's location becomes non-zero (a dot was drawn)
    assert result[50, 50].sum() > 0
    # the top-left origin region, where all missing keypoints collapse to,
    # stays black: no dots and no lines reaching (0, 0)
    assert result[0:8, 0:8].sum() == 0


# ── draw_pose_on_frame: highlight only affects the target joint ──────


def test_highlight_joint_draws_red_only_on_the_target_joint():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # right elbow triplet (highlighted) + left elbow triplet (neutral)
    keypoints = _keypoints(
        right_shoulder=(50, 50),
        right_elbow=(60, 60),
        right_wrist=(70, 50),
        left_shoulder=(20, 20),
        left_elbow=(30, 30),
        left_wrist=(40, 20),
    )

    result = draw_pose_on_frame(frame, keypoints, highlight_joint="cotovelo_direito")

    # the highlighted joint's segments/vertex have red pixels
    assert _has_color(result[54:67, 54:67], RED)
    # the non-highlighted left elbow region has no red pixels
    assert not _has_color(result[24:37, 24:37], RED)


def test_highlight_pescoco_draws_red_on_shoulder_midpoint_to_nose():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # shoulders at y=50 (midpoint x=50), nose straight above at (50, 20):
    # the pescoço highlight is the shoulder-midpoint -> nose segment.
    keypoints = _keypoints(
        left_shoulder=(40, 50),
        right_shoulder=(60, 50),
        nose=(50, 20),
    )

    result = draw_pose_on_frame(frame, keypoints, highlight_joint="pescoco")

    # red somewhere along the vertical mid->nose segment (around y=35, x=50)
    assert _has_color(result[30:41, 45:56], RED)


# ── draw_zone_on_frame ───────────────────────────────────────────────


def test_draw_zone_on_frame_draws_rectangle_in_the_expected_pixel_bounds():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # right half of the frame: x in [50, 100], full height
    result = draw_zone_on_frame(frame, (0.5, 0.0, 1.0, 1.0))

    # the left edge of the rectangle falls on the x=50 column (some pixels
    # of that column are drawn), while the far-left of the frame is empty
    assert result[:, 48:52].sum() > 0
    assert result[:, 0:40].sum() == 0


def test_draw_zone_on_frame_does_not_mutate_original_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    original = frame.copy()

    result = draw_zone_on_frame(frame, (0.5, 0.0, 1.0, 1.0))

    assert np.array_equal(frame, original)
    assert not np.array_equal(result, original)
