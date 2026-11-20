"""GPU 运行库必须在会话创建前加载；CPU 环境仍可使用同一后端。"""

import sys
from types import SimpleNamespace

import pytest

from jg_radar.detect.backends.onnx_backend import OnnxBackend
from jg_radar.integration import gpu_env


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
    # 这台机器上确实装着 cuDNN；这条测的是"找不到时"的兜底路径
    monkeypatch.setattr(gpu_env, "cudnn_lib_dir", lambda roots=None: None)
    model = tmp_path / "model.onnx"
    model.touch()
    backend = OnnxBackend(model)
    backend.load()
    assert backend.is_ready
    assert events == (["preload", "session"] if cuda else ["session"])


def test_cudnn_outside_the_venv_is_preloaded_by_path(tmp_path, monkeypatch):
    """cuDNN 在用户 site 里也要能加载——按路径找，不按 import 找。

    仓库全程 ``PYTHONNOUSERSITE=1``，用户 site 不在 ``sys.path`` 上，所以
    ``importlib.util.find_spec("nvidia.cudnn")`` 会返回 None，但库文件就在磁盘上。
    """
    library = tmp_path / "user-site" / "nvidia" / "cudnn" / "lib"
    library.mkdir(parents=True)
    (library / gpu_env.CUDNN_SONAME).touch()
    monkeypatch.setattr(gpu_env, "cudnn_lib_dir", lambda roots=None: library)
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
        {"cuda": True, "cudnn": True, "msvc": False, "directory": str(library)},
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
