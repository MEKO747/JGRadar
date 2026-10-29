"""非极大值抑制。纯 numpy，几行就够，不值得引依赖。

**按类别分别做**：数字 1 和数字 3 的框完全重叠是两个不同的结论（一次
身份投票的歧义），不该被当成同一个目标的重复框压掉一个。压掉之后跟踪层
只会看到一个干净的结果，永远不知道这里本来有歧义。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from jg_radar.detect.protocol import DetectionRecord

__all__ = ["iou_matrix", "nms", "nms_records"]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两组框 ``(N,4)``/``(M,4)`` 的两两 IoU，返回 ``(N,M)``。"""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0.0, None) * np.clip(iy2 - iy1, 0.0, None)

    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0.0, inter / np.maximum(union, 1e-12), 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45) -> np.ndarray:
    """贪心 NMS。返回保留下来的下标，按分数降序。"""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.intp)

    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thresh]
    return np.asarray(keep, dtype=np.intp)


def nms_records(
    records: Sequence[DetectionRecord], iou_thresh: float = 0.45, *, per_class: bool = True
) -> list[DetectionRecord]:
    """对检测记录做 NMS。``per_class=True`` 时各类别互不抑制。"""
    recs = list(records)
    if len(recs) <= 1:
        return recs

    groups: dict[int, list[int]] = {}
    for i, r in enumerate(recs):
        groups.setdefault(r.class_id if per_class else -1, []).append(i)

    keep: list[int] = []
    for idxs in groups.values():
        boxes = np.array([recs[i].bbox for i in idxs], dtype=np.float64)
        scores = np.array([recs[i].confidence for i in idxs], dtype=np.float64)
        for k in nms(boxes, scores, iou_thresh):
            keep.append(idxs[int(k)])

    keep.sort(key=lambda i: -recs[i].confidence)
    return [recs[i] for i in keep]
