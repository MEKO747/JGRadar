"""让 onnxruntime 找到用户 site 里的 cuDNN，而不是静默退 CPU。

## 问题

`onnxruntime-gpu` 找不到 `libcudnn.so.9` 时**不抛异常**：CUDA 执行器建失败，它退到
CPU 照常返回数字。读的人很容易把 CPU 的耗时当成 GPU 的。

本机有两种现场：

1. cuDNN 由 pip 装进**用户 site**：``~/.local/lib/python3.12/site-packages/nvidia/cudnn/lib``
2. 或者装进 venv：``<venv>/lib/python3.12/site-packages/nvidia/*/lib``

第 1 种被 ``PYTHONNOUSERSITE=1`` 挡在 ``sys.path`` 之外，于是"按模块 import 去找"
（``importlib.util.find_spec("nvidia.cudnn")``）会返回 None——但库文件明明在磁盘上。

## 做法

**按路径找，不按 import 找**：扫上面两个 site-packages 下的 ``nvidia/*/lib``，
把含 ``libcudnn.so.9`` 的目录交给 ``onnxruntime.preload_dlls``。
实测这一步就能把 CUDA 执行器拉起来，不需要 ``LD_LIBRARY_PATH``、不需要 re-exec。
"""

from __future__ import annotations

from pathlib import Path
import sys
import sysconfig

__all__ = ["cudnn_lib_dir", "nvidia_lib_dirs", "preload_cuda_libs"]

#: cuDNN 的 soname。ORT 1.26 要 cuDNN 9 + CUDA 12。
CUDNN_SONAME = "libcudnn.so.9"


def _site_roots() -> list[Path]:
    """可能藏着 ``nvidia/*/lib`` 的 site-packages，按优先级排列。"""
    roots: list[Path] = []

    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        roots.append(Path(purelib))

    # PYTHONNOUSERSITE=1 会让 userbase 变成 None，但"用户 site 不在 sys.path 上"
    # 和"里面的动态库能不能用"是两件事——我们找的是后者。
    user_base = sysconfig.get_paths().get("userbase") or str(Path.home() / ".local")
    tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    root = Path(user_base) / "lib" / tag / "site-packages"
    if root not in roots:
        roots.append(root)

    return roots


def nvidia_lib_dirs(roots: list[Path] | None = None) -> list[Path]:
    """所有存在的 ``nvidia/<组件>/lib`` 目录，按路径排序。"""
    found: list[Path] = []
    for root in (roots if roots is not None else _site_roots()):
        packages = root / "nvidia"
        if not packages.is_dir():
            continue
        for component in sorted(packages.iterdir()):
            lib = component / "lib"
            if lib.is_dir() and lib not in found:
                found.append(lib)
    return found


def cudnn_lib_dir(roots: list[Path] | None = None) -> Path | None:
    """含 ``libcudnn.so.9`` 的目录；没有就返回 None（调用方据此决定要不要告警）。"""
    for lib in nvidia_lib_dirs(roots):
        if (lib / CUDNN_SONAME).is_file():
            return lib
    return None


def preload_cuda_libs(ort=None, *, roots: list[Path] | None = None) -> dict:
    """在建 CUDA 会话之前把 cuDNN 目录告诉 ORT。

    返回 ``{"cudnn_dir": str|None, "preloaded": bool, "reason": str}``——**调用方应该
    把 reason 打出来**。这里不抛异常：装不上 GPU 只是降速，不该让整个节点起不来；
    但静默降速是另外一回事，所以结论要能被看见。
    """
    if ort is None:
        import onnxruntime as ort  # noqa: PLC0415 —— 可选依赖，放在函数里

    directory = cudnn_lib_dir(roots)
    if directory is None:
        return {"cudnn_dir": None, "preloaded": False, "reason": "没找到 libcudnn.so.9"}
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return {"cudnn_dir": str(directory), "preloaded": False,
                "reason": "当前 onnxruntime 没有 preload_dlls"}
    try:
        preload(cuda=True, cudnn=True, msvc=False, directory=str(directory))
    except Exception as exc:  # noqa: BLE001 —— 失败原因要原样带出去
        return {"cudnn_dir": str(directory), "preloaded": False,
                "reason": f"preload_dlls 失败：{type(exc).__name__}: {exc}"}
    return {"cudnn_dir": str(directory), "preloaded": True, "reason": ""}
