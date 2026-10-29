"""检测层对外协议。

检测器被抽象成 ``DetectorBackend``，让权重/推理框架可以随时替换而不动跟踪层。
三个实现要点来自现有方案的经验：

  - ``infer`` 收**一批**图，返回等长的一批结果。EGA 的 ``predict_batch`` 证明
    批量调用是 ROI 级联能跑到 20fps 的关键；单图接口会逼出 N 次调用。
  - ``warmup`` 显式暴露。TensorRT 首次推理含引擎加载与显存分配，不预热会把
    首帧延迟算进计时埋点。
  - ``is_ready`` 让上层能在模型缺失时走降级路径，而不是抛异常把整条流水线打断。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class DetectionRecord:
    """一次检测结果。坐标系是**原图像素**，左上原点。"""

    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]
    """x1, y1, x2, y2。"""
    label: str = ""

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def clipped(self, w: int, h: int) -> "DetectionRecord":
        """裁剪到图像范围内。ROI 映射回原图后常越界，必须裁。"""
        x1, y1, x2, y2 = self.bbox
        return DetectionRecord(
            class_id=self.class_id,
            confidence=self.confidence,
            bbox=(
                min(max(x1, 0.0), float(w)),
                min(max(y1, 0.0), float(h)),
                min(max(x2, 0.0), float(w)),
                min(max(y2, 0.0), float(h)),
            ),
            label=self.label,
        )


@runtime_checkable
class DetectorBackend(Protocol):
    """推理后端协议。"""

    name: str
    input_size: int
    is_ready: bool

    def load(self) -> None:
        """加载权重/引擎。可重复调用，幂等。"""
        ...

    def infer(self, images: Sequence[np.ndarray]) -> list[list[DetectionRecord]]:
        """对一批 BGR 图像推理，返回与输入等长的一批检测列表。"""
        ...

    def warmup(self, n: int = 20) -> None:
        """预热 n 次，避开首帧延迟。"""
        ...
