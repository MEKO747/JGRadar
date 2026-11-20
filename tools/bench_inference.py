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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jg_radar.integration.gpu_env import preload_cuda_libs  # noqa: E402

import onnxruntime as ort  # noqa: E402


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


# ---------------------------------------------------------------------------
# 端到端：走仓库真实级联（letterbox + NMS + 装甲批处理），不是 dummy tensor
# ---------------------------------------------------------------------------
def _read_frame(video: str, resize: str):
    import cv2

    if video:
        cap = cv2.VideoCapture(video)
    else:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit(f"打不开帧来源：{video or '/dev/video0'}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("读不到帧")
    if resize:
        width, height = (int(v) for v in resize.lower().split("x"))
        frame = cv2.resize(frame, (width, height))
    return frame


def _timeit(fn, iters: int):
    fn()  # 预热
    started = time.perf_counter()
    for _ in range(iters):
        result = fn()
    return (time.perf_counter() - started) / iters * 1000, result


def bench_e2e(cfg_paths, *, video: str, resize: str, rois: int, iters: int) -> dict:
    from jg_radar.config import load_config
    from jg_radar.detect.factory import build_cascade

    cfg = load_config(*cfg_paths)
    cascade = build_cascade(cfg.detect)
    cascade.load()
    frame = _read_frame(video, resize)
    height, width = frame.shape[:2]

    car_ms, (cars, car_rois) = _timeit(lambda: cascade.car.detect_rois(frame), iters)

    # 屏幕上没有车时（办公室画面），二级就没 ROI 可跑——那会测出"0 ms"，
    # 等于什么都没测。这里补一组固定 ROI 只为压出装甲级耗时，**并如实标注**。
    synthetic = False
    boxes = car_rois
    if len(boxes) == 0:
        synthetic = True
        step = max(1, width // (rois + 1))
        boxes = np.array(
            [[step * (i + 1) - 40, height * 0.35, step * (i + 1) + 40, height * 0.65]
             for i in range(rois)],
            dtype=np.float64,
        )
    armor_ms, armors = _timeit(lambda: cascade.armor.detect(frame, boxes), iters)

    session = getattr(getattr(cascade.car, "backend", None), "_session", None)
    providers = session.get_providers() if session is not None else []
    # 数字级也要算进整帧：没有真检测时用同一批人工 ROI，否则会报成"0ms"。
    digit_boxes = [hit.bbox for hit in armors] or list(boxes)
    digit = {"ready": bool(getattr(cascade.digit, "is_ready", False)), "ms": None,
             "rois": len(digit_boxes) if cascade.digit.is_ready else 0}
    if digit["ready"] and digit_boxes:
        digit["ms"], _ = _timeit(lambda: cascade.digit.classify(frame, digit_boxes), iters)
    return {
        "mode": "e2e",
        "frame": [width, height],
        "providers": providers,
        "car_ms": round(car_ms, 1),
        "cars": len(cars),
        "armor_ms": round(armor_ms, 1),
        "armor_rois": len(boxes),
        "synthetic_rois": synthetic,
        "armor_hits": len(armors),
        "digit": digit,
        "total_ms": round(car_ms + armor_ms + (digit["ms"] or 0.0), 1),
        "fps": round(1000.0 / (car_ms + armor_ms + (digit["ms"] or 0.0)), 2),
    }


def _maybe_write(out: str, kind: str, payload: dict) -> None:
    """把测量结果落到 out/：口头数字不算证据，文件才算。"""
    if not out:
        return
    import datetime
    import subprocess

    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"{kind}_{stamp}.json"
    path.write_text(json.dumps({"commit": commit, "time": stamp, **payload},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"written": str(path)}, ensure_ascii=False))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="append", default=[], help="ONNX 路径，可重复")
    parser.add_argument("--size", type=int, default=0, help="动态输入时使用的边长")
    parser.add_argument("--config", action="append", default=[], help="从配置里读 car/armor 模型")
    parser.add_argument("--providers", choices=("auto", "cpu", "cuda", "trt"), default="auto")
    parser.add_argument("--threads", type=int, default=4, help="CPU 线程数；0 用 ORT 默认")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--e2e", action="store_true",
                        help="走仓库真实级联（letterbox + NMS + 装甲批处理），不用 dummy tensor")
    parser.add_argument("--video", default="", help="e2e 的帧来源；空则用 /dev/video0")
    parser.add_argument("--resize", default="", help="e2e 时把帧缩放到 WxH，例如 1280x1024")
    parser.add_argument("--rois", type=int, default=6, help="e2e 无检测时补几个 ROI")
    parser.add_argument("--out", default="", help="把结果 JSON 存到该目录（文件名带时间戳）")
    args = parser.parse_args(argv)

    if args.e2e:
        if not args.config:
            parser.error("--e2e 需要 --config（要读模型路径和输入尺寸）")
        preload = preload_cuda_libs(ort)
        print(json.dumps({"available_providers": ort.get_available_providers(),
                          "cudnn": preload}, ensure_ascii=False))
        result = bench_e2e(args.config, video=args.video, resize=args.resize,
                           rois=args.rois, iters=args.iters)
        result["preload"] = preload
        print(json.dumps(result, ensure_ascii=False))
        _maybe_write(args.out, "e2e", result)
        return 0

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

    # 库里找不到 cuDNN 时 ORT 会静默退 CPU；先把能找到的 cuDNN 目录告诉它。
    preload = preload_cuda_libs(ort)
    chosen = providers_for(args.providers)
    print(json.dumps({"available_providers": ort.get_available_providers(),
                      "using": chosen, "cudnn": preload}, ensure_ascii=False))
    exit_code = 0
    for model in models:
        result = bench(model, size=args.size or sizes.get(model, 640), providers=chosen,
                       threads=args.threads, iters=args.iters)
        print(json.dumps(result, ensure_ascii=False))
        _maybe_write(args.out, "single", result)
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
