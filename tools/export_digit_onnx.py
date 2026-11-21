#!/usr/bin/env python3
"""把数字分类模型（MobileNetV2 .pth）导成 ONNX，运行时不再需要 torch。

## 为什么

``detect.digit_backend: mobilenet`` 走的是 torch + torchvision：本机 torch 装在
**用户 site**，而全仓库默认 ``PYTHONNOUSERSITE=1``（为了避开用户 site 里的
numpy 2.5.5 与 GNU Radio/OpenCV 的 ABI 冲突）。于是"要识别数字就得关掉那条
环境变量"，等于把部署环境做坏。

导成 ONNX 之后运行时只需要 onnxruntime——GPU 上 64px 分类器是毫秒级的事。

## 依赖

导出阶段需要 **torch + torchvision + onnx**（onnx 只用于序列化图，约 16 MB）。
运行时不需要任何一个。

用法::

    PYTHONNOUSERSITE= .venv/bin/python tools/export_digit_onnx.py \
        --weights /path/MobileNetv2.2.pth --out models/digit_mobilenetv2.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True, help="MobileNetV2 的 .pth（state_dict）")
    parser.add_argument("--out", required=True, help="输出 .onnx 路径")
    parser.add_argument("--num-classes", type=int, default=6,
                        help="类别数，与 detect.digit_classes 的长度一致")
    parser.add_argument("--img-size", type=int, default=64)
    args = parser.parse_args(argv)

    try:
        import torch
        from torchvision.models import mobilenet_v2
    except ImportError as exc:
        print(json.dumps({"error": f"导出需要 torch + torchvision：{exc}",
                          "hint": "本机 torch 在用户 site，运行时去掉 PYTHONNOUSERSITE=1 即可"},
                         ensure_ascii=False))
        return 2

    weights = Path(args.weights)
    if not weights.is_file():
        print(json.dumps({"error": f"权重不存在：{weights}"}, ensure_ascii=False))
        return 2

    model = mobilenet_v2(weights=None, num_classes=args.num_classes)
    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    # 这份权重是"包了一层"的：键名带 ``backbone.`` 前缀（``backbone.features.0.0.weight``）。
    # 直接灌进 ``mobilenet_v2`` 会报一大串 missing/unexpected key，所以按已知前缀剥一层。
    for prefix in ("backbone.", "model.", "module."):
        if state and all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: value for key, value in state.items()}
            break
    model.load_state_dict(state, strict=True)
    model.eval()

    dummy = torch.randn(1, 3, args.img_size, args.img_size)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            model, dummy, str(out),
            input_names=["images"], output_names=["logits"],
            # **batch 维必须是动态的**。导出成固定 [1,3,64,64] 的话，同一帧里
            # 认出两块装甲板时 `OnnxDigitBackend.infer` 会把裁图堆成 NCHW 送进去，
            # ORT 直接 INVALID_ARGUMENT；`DigitClassifier` 捕到异常就把整批留在
            # unknown——两块装甲的数字同时丢失，而且不报错。
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17, do_constant_folding=True,
            # torch 2.10 默认走 dynamo 导出器，需要 onnxscript；这里用 TorchScript
            # 导出器，只依赖 onnx 包（本机已装 1.18，与 numpy 1.26 兼容）。
            dynamo=False,
        )
    except ImportError as exc:
        print(json.dumps({
            "error": f"导出需要 onnx 包：{exc}",
            "hint": "pip install onnx（约 16 MB，只在导出时用）",
        }, ensure_ascii=False))
        return 2

    dynamic_batch = _check_dynamic_batch(out)
    print(json.dumps({
        "out": str(out),
        "size_mb": round(out.stat().st_size / 1024 / 1024, 2),
        "num_classes": args.num_classes,
        "img_size": args.img_size,
        "dynamic_batch": dynamic_batch,
        "torch": torch.__version__,
    }, ensure_ascii=False))
    return 0 if dynamic_batch else 1


def _check_dynamic_batch(path: Path) -> bool:
    """导出后当场验一遍输入第 0 维是不是符号维。

    不验这一步就会重演「导出成功、两块装甲全 unknown」：图是合法的，
    只是 batch 被钉死成 1。
    """
    try:
        import onnx

        graph = onnx.load(str(path))
        dims = graph.graph.input[0].type.tensor_type.shape.dim
        return bool(dims[0].dim_param) and not dims[0].dim_value
    except Exception:  # noqa: BLE001 - 校验失败按不支持处理，别让导出假装成功
        return False


if __name__ == "__main__":
    raise SystemExit(main())
