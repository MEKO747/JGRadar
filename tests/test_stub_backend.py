"""Stub 后端：整套流水线测试的地基。

它的全部价值在于**确定性**——同一帧两次推理必须完全一致，
否则回放测试无法用金标准比对。
"""

from __future__ import annotations

import numpy as np
import pytest

from jg_radar.detect import DetectionRecord, DetectorBackend
from jg_radar.detect.backends import EmptyBackend, StubBackend


def _img(seed: int, h: int = 64, w: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_satisfies_protocol():
    backend = StubBackend()
    assert isinstance(backend, DetectorBackend)
    assert not backend.is_ready
    backend.load()
    assert backend.is_ready


def test_infer_requires_load():
    with pytest.raises(RuntimeError, match="load"):
        StubBackend().infer([_img(0)])


def test_batch_length_matches_input():
    backend = StubBackend(dets_per_image=2)
    backend.load()
    images = [_img(i) for i in range(5)]
    out = backend.infer(images)
    assert len(out) == 5
    assert all(len(d) == 2 for d in out)


def test_deterministic_across_calls():
    backend = StubBackend(dets_per_image=4)
    backend.load()
    img = _img(7)
    a = backend.infer([img])[0]
    b = backend.infer([img])[0]
    assert a == b


def test_deterministic_across_instances():
    img = _img(7)
    outs = []
    for _ in range(2):
        backend = StubBackend(dets_per_image=4)
        backend.load()
        outs.append(backend.infer([img])[0])
    assert outs[0] == outs[1]


def test_different_content_gives_different_detections():
    backend = StubBackend(dets_per_image=3)
    backend.load()
    a = backend.infer([_img(1)])[0]
    b = backend.infer([_img(2)])[0]
    assert a != b


def test_boxes_inside_image():
    backend = StubBackend(dets_per_image=8)
    backend.load()
    h, w = 48, 72
    for det in backend.infer([_img(3, h, w)])[0]:
        x1, y1, x2, y2 = det.bbox
        assert 0 <= x1 <= x2 <= w
        assert 0 <= y1 <= y2 <= h
        assert det.area > 0


def test_class_ids_in_range():
    backend = StubBackend(num_classes=6, dets_per_image=16)
    backend.load()
    for det in backend.infer([_img(4)])[0]:
        assert 0 <= det.class_id < 6


def test_empty_backend_always_empty():
    backend = EmptyBackend()
    backend.load()
    out = backend.infer([_img(i) for i in range(3)])
    assert out == [[], [], []]


def test_warmup_does_not_raise():
    backend = StubBackend()
    backend.warmup(2)
    assert backend.is_ready


def test_empty_image_safe():
    backend = StubBackend()
    backend.load()
    assert backend.infer([np.zeros((0, 0, 3), dtype=np.uint8)]) == [[]]


def test_detection_record_geometry():
    det = DetectionRecord(class_id=1, confidence=0.9, bbox=(10.0, 20.0, 30.0, 60.0))
    assert det.center == (20.0, 40.0)
    assert det.width == 20.0
    assert det.height == 40.0
    assert det.area == 800.0


def test_detection_record_clipped():
    det = DetectionRecord(class_id=1, confidence=0.9, bbox=(-5.0, -5.0, 200.0, 300.0))
    clipped = det.clipped(96, 64)
    assert clipped.bbox == (0.0, 0.0, 96.0, 64.0)
    assert det.bbox == (-5.0, -5.0, 200.0, 300.0)  # 原对象不变
