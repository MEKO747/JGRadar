#!/usr/bin/env python3
"""把一个 ONNX 模型的输入改成「固定 H/W、batch 保持动态」。

## 为什么要这个

装甲模型的输入原本是 ``[batch, 3, height, width]``——四个维度全动态。动态维度意味着
ORT 每帧都要现算一遍形状链，它把 ``Gather/Unsqueeze/Slice/Concat`` 这类形状算子
整个甩给 CPU：实测装甲模型 **201 个节点在 CPU**（GPU 520）。

固定 H/W 之后形状链被常量折叠掉，CPU 节点归零。**但 batch 必须保持动态**：
管线走 ``RoiBatcher`` 批量推理，固定 batch 会把批处理一起废掉，省下的时间还不够赔。
所以这里只改 height/width，batch 留成 ``dim_param``。

## 与 digit.onnx 一样的规矩

产物是**派生模型**，不进 git；生成命令和 sha256 写进文档。
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

__all__ = ["main"]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="输入模型路径")
    parser.add_argument("--out", required=True, help="输出模型路径")
    parser.add_argument("--size", type=int, required=True, help="固定成 size×size")
    parser.add_argument("--batch-dim", default="batch",
                        help="batch 维度的符号名；固定成常量会把批处理一起废掉，所以默认保持动态")
    parser.add_argument("--fixed-batch", type=int, default=0,
                        help="把 batch 也固定成常量（仅用于对照实验：批处理会一起废掉）")
    args = parser.parse_args(argv)

    import onnx

    source = Path(args.model)
    if not source.is_file():
        parser.error(f"模型不存在：{source}")
    model = onnx.load(str(source))
    inputs = model.graph.input
    if not inputs:
        parser.error("模型没有输入")

    tensor = inputs[0]
    dims = tensor.type.tensor_type.shape.dim
    if len(dims) != 4:
        parser.error(f"期望 4 维输入（NCHW），实际 {len(dims)} 维")
    if args.fixed_batch:
        dims[0].dim_value = int(args.fixed_batch)
        dims[0].ClearField("dim_param")
    elif args.batch_dim:
        dims[0].dim_param = args.batch_dim
        dims[0].ClearField("dim_value")
    for index, value in ((2, args.size), (3, args.size)):
        dims[index].dim_value = int(value)
        dims[index].ClearField("dim_param")
    onnx.checker.check_model(model)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    print(f"已写入 {out}（{out.stat().st_size / 1024 / 1024:.1f} MB）")
    batch_text = args.fixed_batch if args.fixed_batch else (args.batch_dim or "?")
    print(f"输入形状：{batch_text} × 3 × {args.size} × {args.size}")
    print(f"sha256: {sha256_of(out)}")
    print(f"源模型 sha256: {sha256_of(source)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
