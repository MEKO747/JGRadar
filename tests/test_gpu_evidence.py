"""provider 自证：``get_providers()`` 不等于"真在 GPU 上算"。

CUDA 执行器能建起来却一个节点都分不到时，它照样排在列表最前面。所以证据里必须有
**每个 provider 实际分到的节点数**，否则 JSON 自己站不住。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from jg_radar.integration.gpu_evidence import parse_profile, provider_node_counts


def node(provider: str) -> dict:
    return {"cat": "Node", "name": "op", "args": {"provider": provider}}


def test_parses_the_json_array_format(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps([node("CUDAExecutionProvider"), node("CUDAExecutionProvider"),
                                node("CPUExecutionProvider"), {"cat": "Kernel"}]),
                    encoding="utf-8")
    assert parse_profile(path) == {"CUDAExecutionProvider": 2, "CPUExecutionProvider": 1}


def test_parses_the_jsonl_format(tmp_path):
    path = tmp_path / "profile.jsonl"
    path.write_text("\n".join(json.dumps(event) for event in
                              (node("CUDAExecutionProvider"), node("CPUExecutionProvider"))),
                    encoding="utf-8")
    assert parse_profile(path) == {"CUDAExecutionProvider": 1, "CPUExecutionProvider": 1}


def test_ignores_events_without_a_provider(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps([{"cat": "Node", "args": {}}, node("CPUExecutionProvider")]),
                    encoding="utf-8")
    assert parse_profile(path) == {"unknown": 1, "CPUExecutionProvider": 1}


def test_missing_model_file_is_reported_as_empty(tmp_path):
    assert provider_node_counts(tmp_path / "nope.onnx", input_shape=(1, 3, 8, 8)) == {}


class FakeSession:
    def __init__(self, profile_path):
        self.profile_path = profile_path

    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def run(self, _outputs, feed):
        assert feed["images"].shape == (1, 3, 8, 8)
        return [np.zeros((1, 2), dtype=np.float32)]

    def end_profiling(self):
        return str(self.profile_path)


def test_counts_nodes_through_a_session(tmp_path, monkeypatch):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"fake")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps([node("CUDAExecutionProvider"), node("CPUExecutionProvider")]),
                       encoding="utf-8")

    runtime = SimpleNamespace(
        SessionOptions=lambda: SimpleNamespace(),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        InferenceSession=lambda *args, **kwargs: FakeSession(profile),
    )
    counts = provider_node_counts(model, input_shape=(1, 3, 8, 8), ort=runtime)
    assert counts == {"CUDAExecutionProvider": 1, "CPUExecutionProvider": 1}


def test_failure_does_not_raise(tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"fake")

    def boom(*args, **kwargs):
        raise RuntimeError("profiling 不支持")

    runtime = SimpleNamespace(
        SessionOptions=lambda: SimpleNamespace(),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        InferenceSession=boom,
    )
    assert provider_node_counts(model, input_shape=(1, 3, 8, 8), ort=runtime) == {}
