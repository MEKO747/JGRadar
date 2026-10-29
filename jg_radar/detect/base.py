"""级联各阶段共用的检测器骨架。

抽出来只有一个理由：**"哪些检测算数"必须两级一致**。整车和装甲板都走
"letterbox → 推理 → 映射回原图 → 过滤"，如果过滤规则各写各的，那么整车
框能过而装甲板框不能过（或者反过来），级联就会在中间断掉，而且断得很
难查——每一级单独测都通过。

``_accept`` 里各步骤的顺序是有讲究的，见其中注释。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from jg_radar.detect.labels import ClassMap
from jg_radar.detect.nms import nms_records
from jg_radar.detect.protocol import DetectionRecord, DetectorBackend
from jg_radar.detect.roi_batch import LetterboxTransform, letterbox

__all__ = ["BaseDetector", "DetectorSettings"]


@dataclass(frozen=True)
class DetectorSettings:
    """一级检测器的后处理参数。"""

    conf_thresh: float = 0.25
    iou_thresh: float = 0.45
    max_dets: int = 32
    min_box_px: float = 4.0
    """映射回原图后短边小于它的框丢弃。太小的框定位误差比它自身还大。"""


class BaseDetector:
    """单张输入 + 一个后端的检测器。子类负责"输入图怎么来"。"""

    stage: str = "base"

    def __init__(
        self,
        backend: DetectorBackend,
        *,
        class_map: ClassMap | None = None,
        settings: DetectorSettings | None = None,
    ) -> None:
        self.backend = backend
        # ``None`` = 不做类别过滤（后端输出什么就信什么）。给了一张表则
        # **只保留表里的类别**——这正是适配层存在的意义：网络不认识"哪些
        # 类是我要的"，配置告诉它。用一个空字典当成"不过滤"是行不通的，
        # 空表会把所有检测都滤掉，而且表现为"检测器突然什么都不输出了"。
        self.class_map: ClassMap | None = class_map
        self.settings = settings if settings is not None else DetectorSettings()
        self.inference_errors = 0
        self.last_error = ""

    # -- 状态 -------------------------------------------------------------
    @property
    def name(self) -> str:
        return f"{self.stage}:{getattr(self.backend, 'name', 'unknown')}"

    @property
    def input_size(self) -> int:
        return int(self.backend.input_size)

    @property
    def is_ready(self) -> bool:
        return bool(self.backend.is_ready)

    def _wanted(self, class_id: int) -> bool:
        return self.class_map is None or self.class_map.has(class_id)

    def _label(self, class_id: int, fallback: str = "") -> str:
        if self.class_map is not None and self.class_map.has(class_id):
            return self.class_map.label(class_id)
        return fallback

    def load(self) -> None:
        self.backend.load()

    def warmup(self, n: int = 20) -> None:
        self.backend.warmup(n)

    # -- 后处理 -----------------------------------------------------------
    def _accept(
        self,
        records: Sequence[DetectionRecord],
        transform: LetterboxTransform,
        *,
        src_w: int,
        src_h: int,
        valid_region: tuple[float, float, float, float] | None = None,
    ) -> list[DetectionRecord]:
        """把画布坐标的原始检测变成原图坐标的可信检测。

        顺序（每一步都为下一步排除一类错误）：

        1. **置信度**——最便宜的一刀，先砍掉大部分垃圾；
        2. **类别表**——不在配置声明范围内的类别一律不要。网络输出的整数
           只有配着训练时的类别表才有意义，表外的 id 是噪声；
        3. **是否落在有效画面内**——letterbox 的灰边占了画布的一大块，网络
           在那上面会输出完全随机的框。这些框映射回原图后坐标"看起来正常"，
           但指向的地方根本没有东西，是最阴险的一类假检测。

           ``valid_region`` 必须由调用方给：一级是整帧的
           ``transform.content_region()``，二级是瓦片的 ``valid_box``。
           两者**不能**都用 ``transform.content_region()``——瓦片的那个
           区域描述的是"整张原图映射到画布哪里"，因为 ROI 常被放大数倍，
           它会盖住几乎整张画布，检查形同虚设；
        4. **映射回原图并裁剪**；
        5. **最小尺寸**——位置精度不如自身尺寸的框没有信息量；
        6. **NMS**——放最后，因为抑制要在最终坐标上做才自洽；
        7. **截断到 max_dets**。
        """
        if not records:
            return []

        st = self.settings
        kept = [
            r for r in records
            if r.confidence >= st.conf_thresh and self._wanted(r.class_id)
        ]
        if not kept:
            return []

        x1v, y1v, x2v, y2v = (
            transform.content_region() if valid_region is None else valid_region
        )
        centers = np.array([r.center for r in kept], dtype=np.float64)
        # NaN/Inf 显式挡掉。下面的区间比较"恰好"也会把它们判为不在区域内
        # （NaN 的比较恒为 False），但那是 IEEE 754 的副作用而不是意图——
        # 依赖它意味着以后有人把这里换成别的写法时会静默放行 NaN 框，
        # 而 NaN 一旦进入跟踪层，协方差会被污染得无法恢复。
        finite = np.isfinite(centers).all(axis=1)
        inside = (
            finite
            & (centers[:, 0] >= x1v) & (centers[:, 0] <= x2v)
            & (centers[:, 1] >= y1v) & (centers[:, 1] <= y2v)
        )
        kept = [r for r, ok in zip(kept, inside) if ok]
        if not kept:
            return []

        out: list[DetectionRecord] = []
        for r in kept:
            x1, y1, x2, y2 = transform.inverse_box(r.bbox)
            x1 = min(max(x1, 0.0), float(src_w))
            x2 = min(max(x2, 0.0), float(src_w))
            y1 = min(max(y1, 0.0), float(src_h))
            y2 = min(max(y2, 0.0), float(src_h))
            if min(x2 - x1, y2 - y1) < st.min_box_px:
                continue
            out.append(
                DetectionRecord(
                    class_id=int(r.class_id),
                    confidence=float(r.confidence),
                    bbox=(x1, y1, x2, y2),
                    label=self._label(r.class_id, r.label),
                )
            )

        out = nms_records(out, st.iou_thresh)
        return out[: st.max_dets]

    def _infer_canvas(self, image: np.ndarray) -> tuple[np.ndarray, LetterboxTransform]:
        """letterbox 到后端输入尺寸，返回画布与变换。"""
        return letterbox(
            image, self.input_size, self.input_size,
            return_transform=True,
        )

    def _safe_infer(self, images: Sequence[np.ndarray]) -> list[list[DetectionRecord]]:
        """推理失败时返回等长的空结果，而不是把异常抛给流水线。

        检测层挂掉不该让整个雷达站停摆：宁可这一帧没有检测（跟踪层会走
        ``occlusion_hold``），也不要让一个模型问题变成一次未捕获异常。
        """
        if not images:
            return []
        try:
            out = self.backend.infer(list(images))
        except Exception as exc:  # noqa: BLE001 —— 降级仍需保留可诊断的原因
            self.inference_errors += 1
            self.last_error = str(exc)
            return [[] for _ in images]
        if len(out) != len(images):
            self.inference_errors += 1
            self.last_error = "后端结果数与输入图像数不一致"
            return [[] for _ in images]
        return out
