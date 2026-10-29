"""三级数字分类，输出顺序始终与输入装甲框一致。

分类任务读取原始裁图，由后端执行与训练一致的预处理；检测式后端保留瓦片协议。
无效裁图与低可信分类返回 unknown 占位，避免结果错位到另一块装甲。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from jg_radar.detect.labels import ClassMap
from jg_radar.detect.protocol import DetectionRecord, DetectorBackend
from jg_radar.detect.roi_batch import RoiBatcher, infer_roi_batch

__all__ = ["DigitClassifier", "DigitVote"]

UNKNOWN_CLASS = -1
"""没读出数字。**不是** "0 号数字"——用一个真实类别去表示"不知道"，会让
下游的投票把它当成一票投出去。"""


@dataclass(frozen=True)
class DigitVote:
    """一块装甲板的数字结论。``class_id == UNKNOWN_CLASS`` 表示没读出来。"""

    class_id: int
    confidence: float
    label: str = ""

    @property
    def known(self) -> bool:
        return self.class_id != UNKNOWN_CLASS


class DigitClassifier:
    """把装甲板框裁出来，批量读出里面的数字。"""

    def __init__(
        self,
        backend: DetectorBackend,
        *,
        class_map: ClassMap | None = None,
        batcher: RoiBatcher | None = None,
        min_confidence: float = 0.3,
        grid: int = 4,
    ) -> None:
        self.backend = backend
        # 与 ``BaseDetector`` 同一条规则：``None`` = 不过滤，给了表就只认表里的。
        self.class_map: ClassMap | None = class_map
        size = int(backend.input_size)
        self.batcher = batcher if batcher is not None else RoiBatcher(tile_size=size, grid=grid)
        self.min_confidence = float(min_confidence)
        self.inference_errors = 0
        self.last_error = ""

    @property
    def name(self) -> str:
        return f"digit:{getattr(self.backend, 'name', 'unknown')}"

    @property
    def is_ready(self) -> bool:
        return bool(self.backend.is_ready)

    def load(self) -> None:
        self.backend.load()

    def warmup(self, n: int = 20) -> None:
        self.backend.warmup(n)

    def classify(
        self, frame: np.ndarray, boxes: Sequence[Sequence[float]] | np.ndarray
    ) -> list[DigitVote]:
        """对每个框给一个数字结论，返回顺序与输入一致。

        **顺序与输入严格一致**（包括失败的那些也要占位）——调用方是拿它
        和装甲板列表 zip 起来做身份投票的，少一个元素会让后面所有装甲板
        都配上错误的数字，而且不会有任何报错。
        """
        img = np.asarray(frame)
        arr = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        if len(arr) == 0:
            return []
        votes: list[DigitVote] = [
            DigitVote(UNKNOWN_CLASS, 0.0, "unknown") for _ in range(len(arr))
        ]
        if img.size == 0 or not self.is_ready:
            return votes

        if getattr(self.backend, "task", "detect") == "classify":
            # 分类网络读取原始装甲裁图，自行执行训练时的 padding/resize。
            for start in range(0, len(arr), self.batcher.capacity):
                crops, indices = [], []
                for index in range(start, min(start + self.batcher.capacity, len(arr))):
                    box = arr[index]
                    if not np.isfinite(box).all():
                        continue
                    x1, y1 = np.maximum(np.floor(box[:2]).astype(int), 0)
                    x2, y2 = np.minimum(np.ceil(box[2:]).astype(int), (img.shape[1], img.shape[0]))
                    crop = img[y1:y2, x1:x2]
                    if x2 > x1 and y2 > y1 and crop.size:
                        crops.append(crop)
                        indices.append(index)
                if not crops:
                    continue
                try:
                    records = self.backend.infer(crops)
                    if len(records) != len(crops):
                        raise ValueError("数字分类结果数与装甲裁图数不一致")
                    for index, result in zip(indices, records):
                        vote = self._read_one(None, [r for r in result if r.confidence >= self.min_confidence])
                        if vote is not None:
                            votes[index] = vote
                except Exception as exc:
                    self.inference_errors += 1
                    self.last_error = str(exc)
            return votes

        for start in range(0, len(arr), self.batcher.capacity):
            chunk = arr[start:start + self.batcher.capacity]
            batch = self.batcher.build(img, chunk)
            if batch.empty:
                continue
            try:
                raw = infer_roi_batch(self.backend, batch)
            except Exception as exc:
                self.inference_errors += 1
                self.last_error = str(exc)
                continue

            # 先按瓦片的格子把结论分好组（格子互不重叠，归属唯一），
            # 再让每块瓦片在自己的那组里挑最好的一个——不能反过来让每块
            # 瓦片去扫全部输出，那样谁都可能挑到别人的结论。
            grouped: dict[int, list[DetectionRecord]] = {}
            for rec in raw:
                if rec.confidence < self.min_confidence:
                    continue
                cx, cy = rec.center
                tile = batch.tile_at(cx, cy)
                if tile is None or not tile.has_content(cx, cy):
                    continue
                grouped.setdefault(tile.index, []).append(rec)

            for tile in batch.tiles:
                vote = self._read_one(tile, grouped.get(tile.index, ()))
                if vote is not None:
                    votes[start + tile.index] = vote
        return votes

    # -- 内部 -------------------------------------------------------------
    def _read_one(self, tile, candidates: Sequence[DetectionRecord]) -> DigitVote | None:
        """从**已归到这块瓦片**的候选里挑置信度最高的一个结论。"""
        best: DetectionRecord | None = None
        for rec in candidates:
            if best is None or rec.confidence > best.confidence:
                best = rec
        if best is None:
            return None
        if self.class_map is not None and not self.class_map.has(best.class_id):
            # 类别表之外的 id 一律当没读出来。放它过去会让一个训练时的
            # 临时类（比如 "background"）混进身份投票。
            return None
        label = best.label
        if self.class_map is not None:
            label = self.class_map.label(best.class_id)
        return DigitVote(
            class_id=int(best.class_id),
            confidence=float(best.confidence),
            label=label,
        )
