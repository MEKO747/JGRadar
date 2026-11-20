"""数字分类的 ONNX 运行时：预处理必须和 torch 版逐条对齐。

torch 版做的是「补成正方形（黑边）→ resize → RGB → ImageNet 归一化」。
这里用假 onnxruntime 断言送进网络的张量形状与数值范围——预处理错位不会报错，
只会让识别率莫名下降，所以必须由测试兜住。
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from jg_radar.detect.backends.onnx_digit_backend import OnnxDigitBackend


class FakeSession:
    def __init__(self, logits):
        self.logits = logits
        self.seen: list[np.ndarray] = []

    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _outputs, feed):
        self.seen.append(feed["images"])
        return [np.asarray(self.logits, dtype=np.float32)]


def make_backend(tmp_path, logits, monkeypatch, *, num_classes=6):
    model = tmp_path / "digit.onnx"
    model.write_bytes(b"fake")
    session = FakeSession(logits)

    runtime = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=lambda *args, **kwargs: session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    monkeypatch.setattr("jg_radar.integration.gpu_env.preload_cuda_libs",
                        lambda ort: {"cudnn_dir": None, "preloaded": False, "reason": "test"})
    backend = OnnxDigitBackend(model, input_size=64, num_classes=num_classes)
    backend.load()
    return backend, session


def test_loads_and_reports_ready(tmp_path, monkeypatch):
    backend, _ = make_backend(tmp_path, [[0.1] * 6], monkeypatch)
    assert backend.is_ready
    assert backend.reason == ""


def test_missing_model_stays_not_ready(tmp_path, monkeypatch):
    backend = OnnxDigitBackend(tmp_path / "nope.onnx")
    backend.load()
    assert not backend.is_ready
    assert "不存在" in backend.reason


def test_input_is_square_chw_and_normalized(tmp_path, monkeypatch):
    backend, session = make_backend(tmp_path, [[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]], monkeypatch)
    crop = np.full((100, 20, 3), 128, dtype=np.uint8)
    backend.infer([crop, crop])
    batch = session.seen[0]
    assert batch.shape == (2, 3, 64, 64)
    # ImageNet 归一化后，128 灰大致落在 (-1, 1)；未归一化会是 0~255
    assert -3.0 < float(batch.mean()) < 3.0
    assert abs(float(batch.std())) > 0.05


def test_returns_softmax_probabilities_and_argmax(tmp_path, monkeypatch):
    backend, _ = make_backend(tmp_path, [[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]], monkeypatch)
    votes = backend.infer([np.zeros((10, 10, 3), dtype=np.uint8)])
    assert len(votes) == 1
    record = votes[0][0]
    assert record.class_id == 2
    assert record.confidence > 0.9  # softmax 把 5.0 推到接近 1


def test_empty_input_returns_empty(tmp_path, monkeypatch):
    backend, _ = make_backend(tmp_path, [[0.1] * 6], monkeypatch)
    assert backend.infer([]) == []


def test_infer_before_load_raises(tmp_path, monkeypatch):
    backend = OnnxDigitBackend(tmp_path / "digit.onnx")
    with pytest.raises(RuntimeError):
        backend.infer([np.zeros((8, 8, 3), dtype=np.uint8)])


def test_warmup_runs_the_model(tmp_path, monkeypatch):
    backend, session = make_backend(tmp_path, [[0.1] * 6], monkeypatch)
    backend.warmup(2)
    assert len(session.seen) == 2
