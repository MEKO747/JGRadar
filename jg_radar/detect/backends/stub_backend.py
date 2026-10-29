"""确定性假检测器。

存在的理由：本项目的第一优先级是测试，而测试不该被"权重还没下好"阻塞。
Stub 让整条流水线（检测→跟踪→融合→输出）在**没有任何模型**的情况下可跑、
可回放、可比对金标准。

确定性来自**图像内容哈希**而非调用计数：同一帧永远得到同一组检测，
回放两次的 JSONL 必须逐字节相同。
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from jg_radar.detect.protocol import DetectionRecord


class StubBackend:
    """按图像内容生成伪检测。

    参数
    ----
    num_classes: 类别数上界（含 0）。默认 6，对应数字 1/2/3/4/S/Q 的常见编码。
    dets_per_image: 每图生成多少个框。0 表示永远返回空——用于测降级路径。
    jitter: 框尺寸的随机抖动比例。
    seed: 哈希种子，改变它可得到不同的确定性序列。
    """

    def __init__(
        self,
        *,
        num_classes: int = 6,
        dets_per_image: int = 3,
        jitter: float = 0.2,
        seed: int = 0,
        name: str = "stub",
        input_size: int = 640,
    ) -> None:
        self.name = name
        self.input_size = input_size
        self.num_classes = num_classes
        self.dets_per_image = dets_per_image
        self.jitter = jitter
        self.seed = seed
        self._loaded = False

    # -- DetectorBackend --------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def warmup(self, n: int = 20) -> None:
        if not self._loaded:
            self.load()
        dummy = [np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)] * max(1, n)
        self.infer(dummy)

    def infer(self, images: Sequence[np.ndarray]) -> list[list[DetectionRecord]]:
        if not self._loaded:
            raise RuntimeError("StubBackend.load() 未调用")
        return [self._detect_one(img) for img in images]

    # -- 内部 -------------------------------------------------------------
    def _detect_one(self, img: np.ndarray) -> list[DetectionRecord]:
        if self.dets_per_image <= 0 or img.size == 0:
            return []
        h, w = img.shape[:2]
        rng = np.random.default_rng(self._content_seed(img))

        out: list[DetectionRecord] = []
        for _ in range(self.dets_per_image):
            bw = float(rng.uniform(0.03, 0.08) * w)
            bh = float(rng.uniform(0.02, 0.06) * h)
            cx = float(rng.uniform(0.0, max(w - bw, 1.0)))
            cy = float(rng.uniform(0.0, max(h - bh, 1.0)))
            out.append(
                DetectionRecord(
                    class_id=int(rng.integers(0, self.num_classes)),
                    confidence=float(rng.uniform(0.3, 0.95)),
                    bbox=(cx, cy, cx + bw, cy + bh),
                    label=f"stub{int(rng.integers(0, self.num_classes))}",
                )
            )
        return out

    def _content_seed(self, img: np.ndarray) -> int:
        """由图像内容 + 配置决定的确定性种子。"""
        h = hashlib.blake2b(digest_size=8)
        h.update(np.ascontiguousarray(img).tobytes())
        h.update(f"{self.seed}:{self.num_classes}:{self.dets_per_image}".encode())
        return int.from_bytes(h.digest(), "little", signed=False)


class EmptyBackend(StubBackend):
    """永远返回空检测。用于测 ``MEASURED → HOLD → BLIND → MISSING`` 降级链。"""

    def __init__(self, **kwargs) -> None:
        kwargs.pop("dets_per_image", None)
        super().__init__(dets_per_image=0, name="empty", **kwargs)
