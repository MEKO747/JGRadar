"""cuDNN 发现与预加载：ONNX 找不到它时会**静默**退 CPU。

本机现场：onnxruntime-gpu 缺 `libcudnn.so.9` 不报错，退 CPU 继续跑；
而 cuDNN 就在用户 site 里，只是被 `PYTHONNOUSERSITE=1` 挡在 `sys.path` 之外。
所以这里测的是"按路径找"，不是"按 import 找"。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jg_radar.integration import gpu_env


def make_root(tmp_path: Path, *, with_cudnn: bool) -> Path:
    root = tmp_path / "site-packages"
    cudnn = root / "nvidia" / "cudnn" / "lib"
    cudnn.mkdir(parents=True)
    if with_cudnn:
        (cudnn / gpu_env.CUDNN_SONAME).write_bytes(b"\x7fELF")
    other = root / "nvidia" / "cublas" / "lib"
    other.mkdir(parents=True)
    return root


def test_discovers_every_nvidia_component(tmp_path):
    root = make_root(tmp_path, with_cudnn=True)
    dirs = gpu_env.nvidia_lib_dirs([root])
    assert [path.parent.name for path in dirs] == ["cublas", "cudnn"]


def test_skips_roots_without_nvidia(tmp_path):
    assert gpu_env.nvidia_lib_dirs([tmp_path / "empty"]) == []


def test_finds_the_directory_holding_cudnn(tmp_path):
    root = make_root(tmp_path, with_cudnn=True)
    found = gpu_env.cudnn_lib_dir([root])
    assert found is not None and found.parent.name == "cudnn"
    assert (found / gpu_env.CUDNN_SONAME).is_file()


def test_reports_when_cudnn_is_missing(tmp_path):
    root = make_root(tmp_path, with_cudnn=False)
    assert gpu_env.cudnn_lib_dir([root]) is None


class FakeOrt:
    """只记录调用参数，不碰真库。"""

    def __init__(self, *, has_preload: bool = True) -> None:
        self.calls: list[dict] = []
        if has_preload:
            self.preload_dlls = self._preload

    def _preload(self, **kwargs):
        self.calls.append(kwargs)


def test_preload_points_ort_at_the_cudnn_directory(tmp_path):
    root = make_root(tmp_path, with_cudnn=True)
    ort = FakeOrt()
    info = gpu_env.preload_cuda_libs(ort, roots=[root])
    assert info["preloaded"] is True
    assert info["cudnn_dir"].endswith("nvidia/cudnn/lib")
    assert ort.calls == [{"cuda": True, "cudnn": True, "msvc": False,
                          "directory": info["cudnn_dir"]}]


def test_preload_is_a_noop_without_cudnn(tmp_path):
    root = make_root(tmp_path, with_cudnn=False)
    ort = FakeOrt()
    info = gpu_env.preload_cuda_libs(ort, roots=[root])
    assert info["preloaded"] is False
    assert "libcudnn" in info["reason"]
    assert ort.calls == []


def test_preload_reports_old_onnxruntime(tmp_path):
    root = make_root(tmp_path, with_cudnn=True)
    info = gpu_env.preload_cuda_libs(FakeOrt(has_preload=False), roots=[root])
    assert info["preloaded"] is False
    assert "preload_dlls" in info["reason"]


def test_preload_surfaces_exceptions(tmp_path):
    root = make_root(tmp_path, with_cudnn=True)

    class Boom(FakeOrt):
        def _preload(self, **kwargs):  # noqa: ARG002
            raise OSError("libcudnn 加载失败")

    info = gpu_env.preload_cuda_libs(Boom(), roots=[root])
    assert info["preloaded"] is False
    assert "libcudnn 加载失败" in info["reason"]


def test_default_roots_include_venv_and_user_site():
    """默认根目录必须同时覆盖 venv 和用户 site——两种装法都有人用。"""
    roots = [str(path) for path in gpu_env._site_roots()]
    assert any(".venv" in path or "site-packages" in path for path in roots)
    assert any(path.endswith(".local/lib/python3.12/site-packages") or ".local" in path
               for path in roots)


@pytest.mark.parametrize("builder", ["venv", "user"])
def test_both_install_layouts_are_found(tmp_path, builder):
    """两种装法都要能找出来：venv 里装 vs 用户 site 里已有。"""
    root = make_root(tmp_path / builder, with_cudnn=True)
    assert gpu_env.cudnn_lib_dir([root]) is not None
