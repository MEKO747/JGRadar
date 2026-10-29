"""二级装甲检测：保留每块装甲所属车辆与坐标映射。

RoiBatcher 描述独立 ROI 的缩放与画布位置。ONNX 等定尺寸模型接收独立瓦片，
动态 batch 沿 N 维堆叠，固定 batch 则按模型声明分批；支持拼图的后端仍可读取画布。
推理异常会降级为空，同时记录 inference_errors 和 last_error。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from jg_radar.detect.base import BaseDetector
from jg_radar.detect.protocol import DetectionRecord
from jg_radar.detect.roi_batch import RoiBatcher, infer_roi_batch

__all__ = ["ArmorDetector", "ArmorHit"]


@dataclass(frozen=True)
class ArmorHit:
    """一块装甲板检测，带它所属整车框的下标。"""

    car_index: int
    """对应输入 ``rois`` 里的下标。``-1`` 表示不在任何车 ROI 内。"""
    record: DetectionRecord

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.record.bbox

    @property
    def class_id(self) -> int:
        return self.record.class_id

    @property
    def confidence(self) -> float:
        return self.record.confidence


class ArmorDetector(BaseDetector):
    """装甲板检测器。输入是整帧 + 一级给出的车 ROI。"""

    stage = "armor"

    def __init__(self, backend, *, batcher: RoiBatcher | None = None, **kw) -> None:
        super().__init__(backend, **kw)
        # 瓦片尺寸跟随后端输入尺寸：模型说它吃 256，就别硬拼 192 的格子
        self.batcher = batcher if batcher is not None else RoiBatcher(
            tile_size=self.input_size, grid=4
        )

    @property
    def capacity(self) -> int:
        """一张画布装得下多少块 ROI。超出部分需要调用方分批。"""
        return self.batcher.capacity

    def detect(
        self, frame: np.ndarray, rois: Sequence[Sequence[float]] | np.ndarray
    ) -> list[ArmorHit]:
        """对每个车 ROI 检测装甲板。ROI 超过一张画布的容量时按批处理。"""
        img = np.asarray(frame)
        boxes = np.asarray(rois, dtype=np.float64).reshape(-1, 4)
        if img.size == 0 or len(boxes) == 0:
            return []

        src_h, src_w = img.shape[:2]
        hits: list[ArmorHit] = []
        for start in range(0, len(boxes), self.capacity):
            chunk = boxes[start:start + self.capacity]
            batch = self.batcher.build(img, chunk)
            if batch.empty:
                continue
            raw = infer_roi_batch(self.backend, batch, infer=self._safe_infer)
            hits.extend(self._split(batch, raw, src_w=src_w, src_h=src_h, offset=start))
        return hits

    # -- 内部 -------------------------------------------------------------
    def _split(self, batch, raw, *, src_w: int, src_h: int, offset: int) -> list[ArmorHit]:
        """把一次批量推理的结果按瓦片拆开，各自映射回原图。

        后端返回的是一个扁平列表，**没有**"这个框属于哪块瓦片"的信息——
        那正是要在这里补上的东西。判据用瓦片的 ``canvas_box``：画布上的格子
        是**互不重叠**的划分，所以归属唯一且是 O(1)。

        早期版本用的是 ``transform.content_region()``（"这个点被图像覆盖了吗"），
        那对瓦片是错的——ROI 常被放大数倍，整张原图映射过来会盖住几乎整张
        画布，于是每块瓦片都认领所有检测。当时靠"取逆映射后离 ROI 中心最近的
        那块"兜底，恰好蒙对了，但那是运气不是设计。
        """
        if not raw:
            return []

        assignment: dict[int, list[DetectionRecord]] = {}
        for rec in raw:
            cx, cy = rec.center
            tile = batch.tile_at(cx, cy)
            if tile is None:  # 落在瓦片之间的缝里，或画布外
                continue
            assignment.setdefault(tile.index, []).append(rec)

        hits: list[ArmorHit] = []
        for index, recs in assignment.items():
            tile = batch.by_index()[index]
            for rec in self._accept(
                recs, tile.transform, src_w=src_w, src_h=src_h, valid_region=tile.valid_box
            ):
                hits.append(ArmorHit(car_index=offset + tile.index, record=rec))
        return hits
