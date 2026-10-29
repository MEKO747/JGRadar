"""GPU 运行库必须在会话创建前加载；CPU 环境仍可使用同一后端。"""

import sys
from types import SimpleNamespace

import pytest

from jg_radar.detect.backends.onnx_backend import OnnxBackend


@pytest.mark.parametrize("cuda", [True, False])
def test_cuda_libraries_load_before_creating_a_session(tmp_path, monkeypatch, cuda):
    events = []
    providers = ["CPUExecutionProvider"]
    if cuda:
        providers.insert(0, "CUDAExecutionProvider")

    def preload(**kwargs):
        assert kwargs == {"cuda": True, "cudnn": True, "msvc": False}
        events.append("preload")

    def session(path, **kwargs):
        assert kwargs["providers"] == providers
        events.append("session")
        return SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="images")])

    runtime = SimpleNamespace(get_available_providers=lambda: providers,
                              preload_dlls=preload, InferenceSession=session)
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    model = tmp_path / "model.onnx"
    model.touch()
    backend = OnnxBackend(model)
    backend.load()
    assert backend.is_ready
    assert events == (["preload", "session"] if cuda else ["session"])


def test_linux_cudnn_in_another_site_packages_is_preloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    library = tmp_path / "user-site" / "nvidia" / "cudnn" / "lib"
    library.mkdir(parents=True)
    (library / "libcudnn.so.9").touch()
    monkeypatch.setattr("importlib.util.find_spec", lambda name: SimpleNamespace(
        submodule_search_locations=[str(library.parent)]))
    events = []

    def session(*args, **kwargs):
        events.append("session")
        return SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="images")])

    runtime = SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        preload_dlls=lambda **kwargs: events.append(kwargs), InferenceSession=session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    model = tmp_path / "model.onnx"
    model.touch()
    backend = OnnxBackend(model)
    backend.load()
    assert backend.is_ready
    assert events == [
        {"cuda": True, "cudnn": False, "msvc": False},
        {"cuda": False, "cudnn": True, "msvc": False, "directory": str(library)},
        "session",
    ]


def test_old_cpu_runtime_without_preload_api_remains_usable(tmp_path, monkeypatch):
    runtime = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=lambda *args, **kwargs: SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(name="images")]),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    model = tmp_path / "model.onnx"
    model.touch()
    backend = OnnxBackend(model)
    backend.load()
    assert backend.is_ready
