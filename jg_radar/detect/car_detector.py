"""一级：整车检测。整帧 letterbox 到 1280 后一次推理。

这是级联里唯一看整帧的一级，也是唯一"知道全局"的一级——后面所有阶段都
只在自己那一小块 ROI 里看。所以整车漏检是**无法在后续阶段补救**的：车没
被框出来，它的装甲板就永远不会被送进二级。

全图推理代价最高（方案 §十一 里占 10ms/32ms），所以帧率不够时第一个该
考虑的是降输入尺寸或跳帧，而不是砍二级。
"""

from __future__ import annotations

import numpy as np

from jg_radar.detect.base import BaseDetector
from jg_radar.detect.protocol import DetectionRecord

__all__ = ["CarDetector"]


class CarDetector(BaseDetector):
    """整车检测器。输出**原图像素**坐标的整车框。"""

    stage = "car"

    def detect(self, frame: np.ndarray) -> list[DetectionRecord]:
        """对一帧做整车检测。"""
        img = np.asarray(frame)
        if img.size == 0:
            return []
        src_h, src_w = img.shape[:2]
        canvas, transform = self._infer_canvas(img)
        raw = self._safe_infer([canvas])[0]
        return self._accept(raw, transform, src_w=src_w, src_h=src_h)

    def detect_rois(self, frame: np.ndarray) -> tuple[list[DetectionRecord], np.ndarray]:
        """检测并同时给出送二级用的 ROI 列表 ``(N,4)``。

        两级用**同一批框**，不重新排一遍——否则一级输出的框和二级裁的框
        可能不是同一个（顺序、过滤规则漂移），到时候对不上号还找不到原因。
        """
        cars = self.detect(frame)
        if not cars:
            return [], np.zeros((0, 4), dtype=np.float64)
        rois = np.array([c.bbox for c in cars], dtype=np.float64)
        return cars, rois
