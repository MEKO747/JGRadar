import queue
from types import SimpleNamespace

import numpy as np
import pytest

from jg_radar.detect.armor_detector import ArmorHit
from jg_radar.detect.factory import CascadeResult
from jg_radar.detect.digit_classifier import DigitVote
from jg_radar.detect.protocol import DetectionRecord
from tools.vision_preview import _CheckedBackend, _draw, _latest_frames, _put_latest, _run_detection, main


def test_preview_draws_actual_armor_records_without_changing_the_source_image():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    result = CascadeResult(
        cars=[DetectionRecord(0, .9, (10, 70, 280, 190), "car")],
        rois=np.array([[10, 70, 280, 190]]),
        armors=[ArmorHit(0, DetectionRecord(2, .8, (20, 90, 80, 150), "blue"))],
        digits=[],
    )
    preview = _draw(image, result, 12, 80, 90)
    assert preview.shape == image.shape
    assert preview[90, 20].tolist() == [255, 120, 40]
    assert not image.any()


def test_inference_failure_remains_visible_even_when_the_detector_catches_it():
    class Broken:
        def infer(self, images):
            raise RuntimeError("inference failed")

    backend = _CheckedBackend(Broken())
    with pytest.raises(RuntimeError, match="inference failed"):
        backend.infer([np.zeros((2, 2, 3))])
    assert isinstance(backend.error, RuntimeError)


def test_latest_frame_replaces_old_frames_when_inference_is_slow():
    mailbox = queue.Queue(maxsize=1)
    for number in range(10):
        _put_latest(mailbox, number)
    assert mailbox.get_nowait() == 9
    assert mailbox.empty()


def test_camera_read_failure_is_propagated_to_the_consumer():
    class BrokenSource:
        def read(self):
            raise RuntimeError("camera disconnected")

    frames = _latest_frames(BrokenSource())
    with pytest.raises(RuntimeError, match="camera disconnected"):
        next(frames)


def test_simulated_backend_cannot_pass_a_real_vision_check(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.yaml"
    path.write_text("detect:\n  backend: stub\n")

    def unexpected_camera_open(**kwargs):
        raise AssertionError("camera should not be opened")

    monkeypatch.setattr("tools.vision_preview.HikRobotCameraSource", unexpected_camera_open)
    assert main(["--config", str(path), "--no-display", "--max-frames", "1"]) == 1
    assert "不能使用模拟后端" in capsys.readouterr().err


def test_independent_armor_detection_runs_without_any_car_and_has_no_car_identity():
    calls = []
    record = DetectionRecord(2, .9, (20, 20, 50, 50), "blue")

    def detect(image, rois):
        calls.append(np.asarray(rois))
        return [ArmorHit(0, record)]

    cascade = SimpleNamespace(
        car=SimpleNamespace(detect_rois=lambda image: ([], np.empty((0, 4)))),
        armor=SimpleNamespace(detect=detect),
    )
    result = _run_detection(cascade, np.zeros((100, 200, 3)), "independent")
    assert result.cars == []
    np.testing.assert_equal(calls[0], [[0, 0, 200, 100]])
    assert result.armors == [ArmorHit(-1, record)]
    assert result.digits == []


def test_armor_only_mode_crops_the_requested_region_without_running_car_detection():
    calls = []

    def unexpected_car(image):
        raise AssertionError("armor-only mode must not call car detection")

    cascade = SimpleNamespace(
        car=SimpleNamespace(detect_rois=unexpected_car),
        armor=SimpleNamespace(detect=lambda image, rois: calls.append(rois) or []),
    )
    _run_detection(cascade, np.zeros((100, 200, 3)), "armor", (.1, .2, .6, .8))
    np.testing.assert_equal(calls[0], [[20, 20, 120, 80]])


def test_cascade_mode_keeps_the_normal_detection_path():
    sentinel = object()
    cascade = SimpleNamespace(run=lambda image: sentinel)
    assert _run_detection(cascade, np.zeros((100, 200, 3)), "cascade") is sentinel


def test_independent_armor_can_read_digit_without_claiming_a_vehicle_identity():
    record = DetectionRecord(2, .9, (20, 20, 50, 50), "blue")
    calls = []
    vote = DigitVote(2, .95, "3")
    cascade = SimpleNamespace(
        armor=SimpleNamespace(detect=lambda image, rois: [ArmorHit(0, record)]),
        digit=SimpleNamespace(classify=lambda image, boxes: calls.append(boxes) or [vote]),
    )
    result = _run_detection(cascade, np.zeros((100, 200, 3)), "armor")
    assert result.armors[0].car_index == -1
    assert result.cars == []
    assert result.digits == [vote]
    assert calls == [[record.bbox]]
