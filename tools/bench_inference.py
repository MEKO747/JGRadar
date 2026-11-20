#!/usr/bin/env python3
"""推理测速：同一份 ONNX 在不同 provider 上各跑一遍，输出 ms/帧。

只做测量，不装依赖、不做精度评估。装好 cuDNN 之后它就是"GPU 到底多少 ms"的
唯一答案来源——数字必须来自这里，不来自任何估算。

用法::

    # 单个模型，自动用当前可用的 provider（装好 cuDNN 后会包含 CUDA）
    .venv/bin/python tools/bench_inference.py --model /path/car.onnx --size 1280

    # 从配置里把所有检测模型跑一遍
    .venv/bin/python tools/bench_inference.py --config config/vision_check.yaml

    # 指定 provider / 线程数对比
    .venv/bin/python tools/bench_inference.py --model /path/car.onnx --providers cpu --threads 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def providers_for(spec: str) -> list[str]:
    """``auto`` 用当前真能建起来的 provider；``cpu`` / ``cuda`` 可以点名。"""
    available = ort.get_available_providers()
    if spec == "auto":
        order = ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
        return [name for name in order if name in available]
    wanted = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider",
              "trt": "TensorrtExecutionProvider"}[spec]
    if wanted not in available:
        raise SystemExit(f"请求的 provider {wanted} 不可用；当前可用：{available}")
    # CUDA/TRT 后面挂 CPU 兜底，没被分配的节点要能落下来
    return [wanted] + ([] if wanted == "CPUExecutionProvider" else ["CPUExecutionProvider"])


def bench(model: Path, *, size: int, providers: list[str], threads: int, iters: int) -> dict:
    options = ort.SessionOptions()
    if threads > 0:
        options.intra_op_num_threads = threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
    except Exception as exc:  # noqa: BLE001 —— 建不起来就是结论本身，不该吞成"成功"
        return {"model": str(model), "error": f"{type(exc).__name__}: {exc}"}

    # 模型输入可能是固定的（导出时锁死），按它的形状造输入；动态维再用 --size
    shape = session.get_inputs()[0].shape
    height = size if isinstance(shape[2], str) else int(shape[2])
    width = size if isinstance(shape[3], str) else int(shape[3])
    feed = session.get_inputs()[0].name
    tensor = np.random.rand(1, 3, height, width).astype(np.float32)

    for _ in range(2):  # 预热：第一次调用含内存分配和图优化后的算子选择
        session.run(None, {feed: tensor})
    started = time.perf_counter()
    for _ in range(iters):
        session.run(None, {feed: tensor})
    ms = (time.perf_counter() - started) / iters * 1000
    actual = session.get_providers()
    requested = [name for name in providers if name != "CPUExecutionProvider"]
    fell_back = bool(requested) and not any(name in actual for name in requested)
    return {
        "model": str(model),
        "input": [height, width],
        "providers": actual,
        "fell_back_to_cpu": fell_back,
        "threads": threads,
        "iters": iters,
        "ms_per_frame": round(ms, 1),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="append", default=[], help="ONNX 路径，可重复")
    parser.add_argument("--size", type=int, default=0, help="动态输入时使用的边长")
    parser.add_argument("--config", action="append", default=[], help="从配置里读 car/armor 模型")
    parser.add_argument("--providers", choices=("auto", "cpu", "cuda", "trt"), default="auto")
    parser.add_argument("--threads", type=int, default=4, help="CPU 线程数；0 用 ORT 默认")
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args(argv)

    models = [Path(path) for path in args.model]
    sizes: dict[Path, int] = {}
    if args.config:
        from jg_radar.config import load_config

        cfg = load_config(*args.config)
        for key, model_path, size_key in (
            ("car", getattr(cfg.detect, "car_model", ""), "car_img_size"),
            ("armor", getattr(cfg.detect, "armor_model", ""), "armor_img_size"),
        ):
            if model_path:
                path = Path(model_path)
                models.append(path)
                sizes[path] = int(getattr(cfg.detect, size_key, 0) or 640)
            else:
                print(json.dumps({"skipped": key, "reason": "配置里没有模型路径"}, ensure_ascii=False))
    if not models:
        parser.error("至少给一个 --model 或 --config")

    chosen = providers_for(args.providers)
    print(json.dumps({"available_providers": ort.get_available_providers(),
                      "using": chosen}, ensure_ascii=False))
    exit_code = 0
    for model in models:
        result = bench(model, size=args.size or sizes.get(model, 640), providers=chosen,
                       threads=args.threads, iters=args.iters)
        print(json.dumps(result, ensure_ascii=False))
        if "error" in result:
            exit_code = 1
        elif result.get("fell_back_to_cpu"):
            # ORT 请求 CUDA 但装不上时**不会抛**，它静默退到 CPU。不把这件事
            # 打出来的话，读的人会以为这串 ms 是 GPU 的。
            print(json.dumps({"warning": "请求的加速 provider 没建起来，上面是 CPU 的数"}, ensure_ascii=False))
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
