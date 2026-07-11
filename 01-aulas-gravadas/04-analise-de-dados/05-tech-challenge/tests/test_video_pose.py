"""Tests for video/pose.py — joint-angle/velocity extraction from mock
YOLOv8-pose keypoints, and per-frame processing that never stops on a
frame with no detected person.

Covers scenarios from the video-motion-analysis spec:
- Requirement: Extração de série de ângulo e velocidade de movimento
  - Scenario: Cálculo de série a partir de keypoints válidos
  - Scenario: Frame sem pessoa detectada

The real YOLOv8-pose model is never loaded in these tests — the frame
processing loop is exercised against a fake "model" object whose
``predict``/``__call__`` returns hand-built fake ``Results``-like objects,
so no network access or GPU is required.
"""
import math

import numpy as np
import pytest

from video.pose import (
    KEYPOINT_NAMES,
    RIGHT_ELBOW_TRIPLET,
    compute_joint_angle,
    compute_velocity,
    extract_frame_series,
)


def _keypoints(**overrides):
    """Build a 17x2 keypoint array (COCO order) with all points at the
    origin except for the ones overridden by name."""
    points = np.zeros((17, 2), dtype=float)
    for name, xy in overrides.items():
        points[KEYPOINT_NAMES.index(name)] = xy
    return points


# ── compute_joint_angle ──────────────────────────────────────────────


def test_compute_joint_angle_returns_180_for_straight_arm():
    # shoulder, elbow, wrist collinear -> fully extended arm -> 180 degrees
    # (all points offset away from the origin: (0, 0) is the "missing
    # keypoint" sentinel, see _MISSING_KEYPOINT in video/pose.py)
    keypoints = _keypoints(
        right_shoulder=(5.0, 5.0),
        right_elbow=(6.0, 5.0),
        right_wrist=(7.0, 5.0),
    )

    angle = compute_joint_angle(keypoints, RIGHT_ELBOW_TRIPLET)

    assert angle == pytest.approx(180.0, abs=1e-6)


def test_compute_joint_angle_returns_90_for_right_angle_bend():
    keypoints = _keypoints(
        right_shoulder=(5.0, 5.0),
        right_elbow=(6.0, 5.0),
        right_wrist=(6.0, 6.0),
    )

    angle = compute_joint_angle(keypoints, RIGHT_ELBOW_TRIPLET)

    assert angle == pytest.approx(90.0, abs=1e-6)


def test_compute_joint_angle_returns_none_when_any_triplet_point_is_missing():
    # A (0, 0) keypoint is how "not detected" is represented by ultralytics
    # when confidence is zero; treat it as missing rather than a real point.
    keypoints = _keypoints(right_shoulder=(5.0, 5.0), right_elbow=(6.0, 5.0))
    # right_wrist left at (0, 0) -> missing

    angle = compute_joint_angle(keypoints, RIGHT_ELBOW_TRIPLET)

    assert angle is None


# ── compute_velocity ─────────────────────────────────────────────────


def test_compute_velocity_is_euclidean_distance_between_reference_points():
    previous = _keypoints(right_wrist=(1.0, 1.0))
    current = _keypoints(right_wrist=(4.0, 5.0))

    velocity = compute_velocity(current, previous, reference_point="right_wrist")

    assert velocity == pytest.approx(5.0)


def test_compute_velocity_returns_none_when_reference_point_missing_in_either_frame():
    previous = _keypoints()  # right_wrist at origin -> "missing"
    current = _keypoints(right_wrist=(3.0, 4.0))

    velocity = compute_velocity(current, previous, reference_point="right_wrist")

    assert velocity is None


def test_compute_velocity_returns_none_when_previous_frame_is_none():
    current = _keypoints(right_wrist=(3.0, 4.0))

    velocity = compute_velocity(current, None, reference_point="right_wrist")

    assert velocity is None


# ── extract_frame_series (mocked YOLO model) ─────────────────────────


class _FakeKeypoints:
    def __init__(self, xy):
        self.xy = xy  # torch-like: indexable, len() == number of detected persons


class _FakeBoxes:
    def __init__(self, xyxy, cls):
        self.xyxy = xyxy
        self.cls = cls


class _FakeResult:
    def __init__(self, keypoints_xy=None, boxes_xyxy=None, boxes_cls=None):
        self.keypoints = _FakeKeypoints(keypoints_xy if keypoints_xy is not None else [])
        self.boxes = _FakeBoxes(
            boxes_xyxy if boxes_xyxy is not None else [],
            boxes_cls if boxes_cls is not None else [],
        )


class _FakePoseModel:
    """Stands in for ``ultralytics.YOLO`` loaded with yolov8n-pose.pt."""

    def __init__(self, results_per_frame):
        self._results_per_frame = results_per_frame
        self.calls = 0

    def predict(self, frame, verbose=False):
        result = self._results_per_frame[self.calls]
        self.calls += 1
        return [result]


def _person_keypoints(right_shoulder, right_elbow, right_wrist):
    kp = _keypoints(right_shoulder=right_shoulder, right_elbow=right_elbow, right_wrist=right_wrist)
    return np.array([kp])  # one detected person


def test_extract_frame_series_computes_angle_and_velocity_for_frames_with_person():
    results = [
        _FakeResult(keypoints_xy=_person_keypoints((5, 5), (6, 5), (7, 5))),  # straight arm, frame 0
        _FakeResult(keypoints_xy=_person_keypoints((5, 5), (6, 5), (6, 6))),  # bent arm, frame 1
    ]
    model = _FakePoseModel(results)
    frames = [np.zeros((10, 10, 3), dtype="uint8") for _ in results]

    series = extract_frame_series(model, frames, fps=10.0)

    assert len(series) == 2
    assert series[0]["has_pose"] is True
    assert series[0]["angle"] == pytest.approx(180.0)
    assert series[0]["velocity"] is None  # no previous frame yet
    assert series[1]["has_pose"] is True
    assert series[1]["angle"] == pytest.approx(90.0)
    # wrist moved (7,5)->(6,6): euclidean distance sqrt(2)
    assert series[1]["velocity"] == pytest.approx(math.sqrt(2))


def test_extract_frame_series_marks_frame_without_person_as_no_pose_data_and_continues():
    results = [
        _FakeResult(keypoints_xy=_person_keypoints((5, 5), (6, 5), (7, 5))),
        _FakeResult(keypoints_xy=np.zeros((0, 17, 2))),  # no person detected
        _FakeResult(keypoints_xy=_person_keypoints((5, 5), (6, 5), (7, 5))),
    ]
    model = _FakePoseModel(results)
    frames = [np.zeros((10, 10, 3), dtype="uint8") for _ in results]

    series = extract_frame_series(model, frames, fps=10.0)

    assert len(series) == 3  # processing did not stop
    assert series[1]["has_pose"] is False
    assert series[1]["angle"] is None
    assert series[1]["velocity"] is None
    # third frame resumes normal processing
    assert series[2]["has_pose"] is True
    assert series[2]["angle"] == pytest.approx(180.0)


def test_extract_frame_series_assigns_increasing_timestamps_from_fps():
    results = [_FakeResult(keypoints_xy=np.zeros((0, 17, 2))) for _ in range(3)]
    model = _FakePoseModel(results)
    frames = [np.zeros((10, 10, 3), dtype="uint8") for _ in results]

    series = extract_frame_series(model, frames, fps=2.0)

    assert [round(f["timestamp_s"], 3) for f in series] == [0.0, 0.5, 1.0]


def test_extract_frame_series_returns_object_detections_per_frame():
    detection_result = _FakeResult(
        keypoints_xy=np.zeros((0, 17, 2)),
        boxes_xyxy=[[10, 10, 20, 20], [30, 30, 40, 40]],
        boxes_cls=[0, 1],
    )
    model = _FakePoseModel([detection_result])
    frames = [np.zeros((10, 10, 3), dtype="uint8")]

    series = extract_frame_series(model, frames, fps=1.0)

    assert len(series[0]["detections"]) == 2
    assert series[0]["detections"][0]["cls"] == 0
    assert series[0]["detections"][0]["xyxy"] == [10, 10, 20, 20]
