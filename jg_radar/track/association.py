"""数据关联：**双闸门 → 代价矩阵 → 匈牙利**。

方案 §六 的顺序不能换。先门后配的理由是：匈牙利求的是全局最优，只要一对
"看起来还行"的错误配对在总代价上划算，它就会为了整体最优而接受局部荒谬。
门限的作用是把荒谬的配对**在求最优之前**从可选集里删掉——匈牙利只能在剩下的
合法配对里谈最优。

## 双闸门（取自 shark ``main.py:770-864``）

``raw_jump`` **原始观测跳变门**
    新检测与轨迹**上一次真正观测到**的位置之间的距离。它不看滤波器怎么想，
    所以能拦住一类攻击：滤波器已经被缓慢带偏，预测位置正好落在假检测上。
    预测残差门这时候是满意的，原始跳变门不满意。

``chi2`` **预测残差门**
    观测相对预测分布的马氏距离平方，阈值 ``χ²(0.95, 4) = 9.49``。它管的是
    "这个检测符合轨迹的运动模型吗"，对缓慢漂移敏感，对单帧大跳变不敏感——
    正好和 raw_jump 互补。**缺一个都会漏一类错误。**

两个阈值都必须来自 ``tools/measure_noise.py`` 的实测（≥3σ）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from jg_radar.detect.nms import iou_matrix

__all__ = [
    "FORBIDDEN_COST",
    "Associations",
    "associate",
    "build_cost_matrix",
    "gate_mask",
    "raw_jump_matrix",
]

FORBIDDEN_COST = 1.0e6
"""被门限拒掉的配对在代价矩阵里的取值。

**用大有限值而不是 ``inf``**：scipy 的 ``linear_sum_assignment`` 在某一行
全为 ``inf`` 时会直接报"矩阵不可行"，而"这一帧所有检测都被某条轨迹的闸门
拒了"是完全正常的现象（目标被挡了）。给个大有限值，匈牙利照常跑完，最后
我们再按 ``max_cost`` 把它们筛掉——语义一样，但不会让正常情况抛异常。
"""


@dataclass(frozen=True)
class Associations:
    """一次关联的结果。三个集合互不相交，并集是全部下标。"""

    pairs: tuple[tuple[int, int], ...]
    """``(轨迹下标, 检测下标)``，按轨迹下标升序。"""
    unmatched_tracks: tuple[int, ...]
    unmatched_detections: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def empty(self) -> bool:
        return not self.pairs


def raw_jump_matrix(
    last_centers: np.ndarray, det_centers: np.ndarray
) -> np.ndarray:
    """原始观测跳变：两次观测中心之间的欧氏距离（像素），``(T, D)``。

    ``last_centers`` 里为 NaN 的轨迹（还没观测过）给 0——新轨迹没有"上一次
    观测"，跳变门对它无从谈起，该由残差门和代价去管。
    """
    a = np.asarray(last_centers, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(det_centers, dtype=np.float64).reshape(-1, 2)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return np.where(np.isfinite(dist), dist, 0.0)


def gate_mask(
    *,
    raw_jump: np.ndarray | None = None,
    raw_jump_max: float = float("inf"),
    chi2: np.ndarray | None = None,
    chi2_max: float = float("inf"),
    same_label: np.ndarray | None = None,
) -> np.ndarray:
    """三道闸门取交集，返回 ``(T, D)`` 布尔掩码。

    ``same_label`` 是类别相符矩阵（``None`` = 不看类别）。**类别不符是硬拒，
    不是代价**：把 B1 的检测绑到 R1 的轨迹上，后面整条身份投票链都会跟着错，
    而投票是慢变量，错了要好几秒才能纠回来。这不是一个可以用"其他项便宜"
    来补偿的权衡。
    """
    shape = None
    for m in (raw_jump, chi2, same_label):
        if m is not None:
            shape = np.asarray(m).shape
            break
    if shape is None:
        raise ValueError("gate_mask 至少要给一个矩阵")

    ok = np.ones(shape, dtype=bool)
    if raw_jump is not None:
        ok &= np.asarray(raw_jump) <= float(raw_jump_max)
    if chi2 is not None:
        # NaN 表示"算不出来"，一律拒（不能假装它合格）
        c = np.asarray(chi2, dtype=np.float64)
        ok &= np.isfinite(c) & (c <= float(chi2_max))
    if same_label is not None:
        ok &= np.asarray(same_label, dtype=bool)
    return ok


def same_label_matrix(
    track_labels: Sequence[str] | None, det_labels: Sequence[str] | None
) -> np.ndarray | None:
    """类别相符矩阵。任一侧没给标签就返回 ``None``（不按类别过滤）。

    **空字符串是"不知道"，不是"一个叫空的类别"**：检测器没读出类别时
    强行按类别过滤会把所有配对都拒掉。不知道就不拦，交给别的项。
    """
    if track_labels is None or det_labels is None:
        return None
    t = [str(x) for x in track_labels]
    d = [str(x) for x in det_labels]
    if not t or not d:
        return None
    if all(x == "" for x in t) or all(x == "" for x in d):
        return None
    tt = np.array(t, dtype=object)[:, None]
    dd = np.array(d, dtype=object)[None, :]
    unknown = (tt == "") | (dd == "")
    return unknown | (tt == dd)


def build_cost_matrix(
    track_boxes: np.ndarray,
    det_boxes: np.ndarray,
    *,
    allowed: np.ndarray | None = None,
    track_world: np.ndarray | None = None,
    det_world: np.ndarray | None = None,
    w_iou: float = 1.0,
    w_center: float = 0.7,
    w_world: float = 0.5,
    center_scale_px: float = 120.0,
    world_scale_cm: float = 300.0,
) -> np.ndarray:
    """代价矩阵 ``(T, D)``，越小越像。

    三项加权和，每一项都先归一化到 ``[0, 1]``，所以权重是可以横向比较的：

    - ``1 − IoU``：形状与位置的整体相似度。像素域关联的主项。
    - ``1 − exp(−d/scale)``：中心距。IoU 对**远处小目标**几乎不敏感——一个
      10×10 的框平移 20px，IoU 直接归零，代价饱和到 1，再也分不出"差一点"
      和"差很远"。指数项在近距离有分辨力，正好补上这段。
    - ``1 − exp(−d_world/scale)``：场地距离。只在两侧都有世界坐标时算。
      它管的是 IoU 和中心距都失灵的情况——两辆车在画面上重叠（远近关系被
      投影压扁），此时场地坐标还分得开。

    ``allowed`` 为假的位置填 ``FORBIDDEN_COST``。
    """
    tb = np.asarray(track_boxes, dtype=np.float64).reshape(-1, 4)
    db = np.asarray(det_boxes, dtype=np.float64).reshape(-1, 4)
    T, D = len(tb), len(db)
    if T == 0 or D == 0:
        return np.zeros((T, D), dtype=np.float64)

    cost = w_iou * (1.0 - iou_matrix(tb, db))

    tc = np.stack([(tb[:, 0] + tb[:, 2]) * 0.5, (tb[:, 1] + tb[:, 3]) * 0.5], axis=1)
    dc = np.stack([(db[:, 0] + db[:, 2]) * 0.5, (db[:, 1] + db[:, 3]) * 0.5], axis=1)
    dist_px = np.linalg.norm(tc[:, None, :] - dc[None, :, :], axis=-1)
    cost += w_center * (1.0 - np.exp(-dist_px / max(center_scale_px, 1e-6)))

    if track_world is not None and det_world is not None:
        tw = np.asarray(track_world, dtype=np.float64).reshape(-1, 2)
        dw = np.asarray(det_world, dtype=np.float64).reshape(-1, 2)
        if len(tw) == T and len(dw) == D:
            dist_cm = np.linalg.norm(tw[:, None, :] - dw[None, :, :], axis=-1)
            # 有 NaN（投影失败）的那一对不加这一项，而不是当 0 距离——
            # 当 0 会把"不知道"变成"完全重合"，凭空制造一次假匹配
            valid_pair = np.isfinite(dist_cm)
            term = w_world * (1.0 - np.exp(-np.where(valid_pair, dist_cm, 0.0)
                                           / max(world_scale_cm, 1e-6)))
            cost += np.where(valid_pair, term, 0.0)

    if allowed is not None:
        cost = np.where(np.asarray(allowed, dtype=bool), cost, FORBIDDEN_COST)
    return cost


def associate(cost: np.ndarray, *, max_cost: float = 1.2) -> Associations:
    """匈牙利求全局最优配对，再按 ``max_cost`` 筛掉不合格的。

    **先求最优后筛代价**，而不是先把超代价的位置置成禁止：置成禁止会改变
    最优解——一条轨迹本可以"将就"配一个贵一点的检测，禁止之后它变成未匹配，
    而那个检测转手配给了另一条轨迹，连锁改变整张分配。先让匈牙利在完整矩阵上
    求最优，再拒绝超限的配对，是在"尽量多配对"和"配对要可信"之间明确的取舍：
    **宁可漏配（走遮挡保持），不可错配（污染轨迹）。**
    """
    from scipy.optimize import linear_sum_assignment

    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError(f"代价矩阵必须是二维，得到 {c.shape}")
    T, D = c.shape
    if T == 0 or D == 0:
        return Associations((), tuple(range(T)), tuple(range(D)))

    rows, cols = linear_sum_assignment(c)
    pairs = tuple(
        sorted(
            (int(r), int(det))
            for r, det in zip(rows, cols)
            if c[r, det] <= float(max_cost)
        )
    )
    matched_t = {r for r, _ in pairs}
    matched_d = {d for _, d in pairs}
    return Associations(
        pairs=pairs,
        unmatched_tracks=tuple(i for i in range(T) if i not in matched_t),
        unmatched_detections=tuple(i for i in range(D) if i not in matched_d),
    )
