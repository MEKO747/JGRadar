"""按颜色分割的确定性检测后端。**无需权重，但不是假检测。**

``StubBackend`` 的框来自图像内容哈希——确定、可复现，但那些框和画面上
真实有什么**毫无关系**。于是它测不了这个项目最要紧的一类问题：级联的
坐标对不对。框是随机的，就没有真值可比。

这个后端用连通域分析找出画面上真实的色块，返回它们的真实包围盒。于是

    在合成图上放一个 40×40 的方块 → 走完 letterbox / 拼图 / 映射回来
    → 断言检测框的坐标就是那个方块的位置

成为一条可以**在没有任何模型权重的情况下**跑的端到端几何测试。真实权重
到位后，``BlobBackend`` 仍然有用：它是级联坐标系的一条不依赖模型的基准线，
换后端之后回归它，就能分清"坐标错了"和"模型不行"。

只依赖 ``scipy.ndimage``（scipy 本来就是依赖），运行时没有额外负担。
"""

from __future__ import annotations

import importlib.util
from typing import Sequence

import numpy as np

from jg_radar.detect.protocol import DetectionRecord

__all__ = ["BlobBackend"]


class BlobBackend:
    """找出与 ``target`` 颜色接近的连通区域，返回其包围盒。

    参数
    ----
    target:
        目标颜色。三通道给 ``(B,G,R)``；灰度图给标量或单元素序列。
    tol:
        各通道允许的偏差。**注意别把 letterbox 的填充灰(114,114,114)圈进来**：
        那会让画布上出现一整圈假检测，而且因为它在边缘，看起来很像真的。
    min_area:
        小于它的连通域丢弃（像素数，在画布坐标系里）。
    max_dets:
        按面积降序截断。置信度也由相对面积给出，于是 NMS 的排序是确定的。
    class_ids:
        给定后按**面积排名**取模分配 class_id（最大的块拿 ``class_ids[0]``）。
        这是给测试用的：放两个块就能预知它们各自的类别。默认全部用 ``class_id``。
    """

    def __init__(
        self,
        target=(255, 255, 255),
        *,
        tol: float = 40.0,
        min_area: int = 16,
        max_dets: int = 32,
        input_size: int = 1280,
        name: str = "blob",
        class_id: int = 0,
        class_ids: Sequence[int] | None = None,
    ) -> None:
        self.target = np.asarray(target, dtype=np.int16).reshape(-1)
        self.tol = float(tol)
        self.min_area = int(min_area)
        self.max_dets = int(max_dets)
        self.input_size = int(input_size)
        self.name = name
        self.class_id = int(class_id)
        self.class_ids = None if class_ids is None else [int(c) for c in class_ids]
        self._loaded = False

    # -- DetectorBackend --------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if importlib.util.find_spec("scipy.ndimage") is None:  # pragma: no cover
            raise RuntimeError("BlobBackend 需要 scipy.ndimage")
        self._loaded = True

    def warmup(self, n: int = 20) -> None:
        if not self._loaded:
            self.load()
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self.infer([dummy] * max(1, min(n, 4)))

    def infer(self, images: Sequence[np.ndarray]) -> list[list[DetectionRecord]]:
        if not self._loaded:
            raise RuntimeError("BlobBackend.load() 未调用")
        return [self._detect_one(np.asarray(img)) for img in images]

    # -- 内部 -------------------------------------------------------------
    def _detect_one(self, img: np.ndarray) -> list[DetectionRecord]:
        from scipy import ndimage

        if img.size == 0:
            return []
        h, w = img.shape[:2]
        mask = self._mask(img)
        if not mask.any():
            return []

        # 8 连通：缩小时矩形边缘可能出现单像素断口，4 连通会把一个块劈成两半
        labeled, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
        if n == 0:
            return []

        areas = np.bincount(labeled.ravel())
        areas[0] = 0  # 背景
        order = [i for i in np.argsort(-areas, kind="stable") if areas[i] >= self.min_area]
        order = order[: self.max_dets]
        if not order:
            return []

        slices = ndimage.find_objects(labeled)
        out: list[DetectionRecord] = []
        for rank, label_id in enumerate(order):
            sl = slices[label_id - 1]
            if sl is None:
                continue
            y0, y1 = sl[0].start, sl[0].stop
            x0, x1 = sl[1].start, sl[1].stop
            cid = (
                self.class_id if self.class_ids is None
                else self.class_ids[rank % len(self.class_ids)]
            )
            out.append(
                DetectionRecord(
                    class_id=cid,
                    confidence=self._confidence(int(areas[label_id]), h, w),
                    bbox=(float(x0), float(y0), float(x1), float(y1)),
                    label=f"blob{cid}",
                )
            )
        return out

    def _mask(self, img: np.ndarray) -> np.ndarray:
        """颜色容差掩码。灰度图与三通道都支持。"""
        if img.ndim == 2:
            arr = img[:, :, None].astype(np.int16)
            tgt = self.target[:1] if self.target.size else np.array([255], dtype=np.int16)
        else:
            arr = img[:, :, :3].astype(np.int16)
            tgt = self.target[:3] if self.target.size >= 3 else np.repeat(self.target[:1], 3)
        return np.all(np.abs(arr - tgt) <= self.tol, axis=-1)

    @staticmethod
    def _confidence(area: int, h: int, w: int) -> float:
        """由相对面积给出置信度：块越大越可信，且**严格单调**。

        单调很重要——NMS 用它排序，若面积不同的两块分数相同，抑制顺序就
        取决于输入顺序，回放两次可能得到不同结果。
        """
        frac = area / float(max(h * w, 1))
        return float(np.clip(0.05 + 19.0 * frac, 0.05, 0.99))
