"""数据关联：代价矩阵与匈牙利。

这一层最容易出的不是崩溃，而是**配错**——把 A 车的检测配给 B 车的轨迹。
配错之后位置还是"看起来正常"的，错的只是身份；而身份是慢变量，要好几秒
才纠得回来。所以这里测的重点是门限有没有真的挡住不可能的组合。
"""

from __future__ import annotations

import numpy as np
import pytest

from jg_radar.track import (
    associate,
    build_cost_matrix,
    gate_mask,
    same_label_matrix,
)
from jg_radar.track.association import FORBIDDEN_COST


def box(cx: float, cy: float, w: float = 80.0, h: float = 40.0):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


# ======================================================================
# 代价矩阵
# ======================================================================
def test_an_identical_box_costs_nothing():
    c = build_cost_matrix(np.array([box(100, 100)]), np.array([box(100, 100)]))
    assert c[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_cost_grows_with_separation():
    tracks = np.array([box(100, 100)])
    dets = np.array([box(100, 100), box(110, 100), box(160, 100), box(400, 100)])
    c = build_cost_matrix(tracks, dets)[0]
    assert c[0] < c[1] < c[2] < c[3], f"代价不是单调的：{c}"


def test_cost_is_normalised_to_a_comparable_range():
    """三项都归一到 ``[0,1]``，权重才能横向比较。最大值是权重之和。"""
    tracks = np.array([box(0, 0)])
    dets = np.array([box(5000, 5000)])
    c = build_cost_matrix(tracks, dets, w_iou=1.0, w_center=0.7, w_world=0.5)
    assert 0.0 <= c[0, 0] <= 2.2 + 1e-9


def test_world_distance_breaks_a_tie_in_the_image():
    """两辆车在画面上重叠（远近被投影压扁）时，场地坐标还分得开。"""
    tracks = np.array([box(100, 100)])
    dets = np.array([box(100, 100)])  # 图像上完全一样

    near = build_cost_matrix(
        tracks, dets, track_world=np.array([[500.0, 750.0]]),
        det_world=np.array([[510.0, 750.0]]),
    )
    far = build_cost_matrix(
        tracks, dets, track_world=np.array([[500.0, 750.0]]),
        det_world=np.array([[1500.0, 750.0]]),
    )
    assert far[0, 0] > near[0, 0], "场地距离没有进入代价"


def test_a_missing_world_coordinate_does_not_create_a_false_match():
    """投影失败（NaN）那一对该**不加**距离项，而不是当成距离 0。

    当成 0 就是把"不知道"变成"完全重合"，凭空制造一次假匹配。
    """
    tracks = np.array([box(100, 100)])
    dets = np.array([box(100, 100)])
    c = build_cost_matrix(
        tracks, dets,
        track_world=np.array([[np.nan, np.nan]]),
        det_world=np.array([[900.0, 900.0]]),
    )
    # 只剩 IoU(0) + 中心距(0) 两项
    assert c[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_forbidden_pairs_get_the_sentinel():
    tracks = np.array([box(100, 100)])
    dets = np.array([box(100, 100), box(200, 200)])
    allowed = np.array([[True, False]])
    c = build_cost_matrix(tracks, dets, allowed=allowed)
    assert c[0, 0] < FORBIDDEN_COST
    assert c[0, 1] == pytest.approx(FORBIDDEN_COST)


def test_empty_inputs_give_an_empty_matrix():
    c = build_cost_matrix(np.zeros((0, 4)), np.array([box(1, 1)]))
    assert c.shape == (0, 1)


# ======================================================================
# 类别门
# ======================================================================
def test_a_label_mismatch_is_a_hard_gate_not_a_cost():
    """把 B1 的检测绑到 R1 的轨迹上，后面整条身份投票链都会跟着错。
    这不是一个能用"其他项便宜"补偿的权衡。"""
    m = same_label_matrix(["B1"], ["R1"])
    assert m.tolist() == [[False]]


def test_a_blank_label_never_blocks():
    """空串是"没读出来"，不是"一个叫空的类别"。

    当成类别去比，会把所有配对都拒掉——检测器没读出类别的那一帧，
    整条轨迹直接断掉。**混合的一帧才是有意义的情形**：一部分读出来了、
    一部分没有，没读出来的那些不能因此被排除。
    """
    m = same_label_matrix(["B1"], ["", "B1", "R1"])
    assert m.tolist() == [[True, True, False]], "空标签把配对拦掉了"
    # 轨迹侧全都没读出来 → 整项不做门
    assert same_label_matrix([""], ["R1"]) is None


def test_no_labels_at_all_means_no_class_gate():
    assert same_label_matrix(None, ["B1"]) is None
    assert same_label_matrix(["B1"], None) is None
    assert same_label_matrix(["", ""], ["", ""]) is None


def test_the_class_gate_combines_with_the_others():
    allowed = gate_mask(
        raw_jump=np.array([[10.0, 10.0]]),
        raw_jump_max=120.0,
        same_label=np.array([[True, False]]),
    )
    assert allowed.tolist() == [[True, False]]


# ======================================================================
# 匈牙利
# ======================================================================
def test_hungarian_beats_greedy():
    """构造一个贪心会输的局面，锁住"全局最优"这件事。

    贪心先拿最便宜的那一对 ``A–1``（0.1），剩给 B 的只有 ``B–2``（10.0），
    总代价 10.1。全局最优是 ``A–2`` + ``B–1`` = 0.35。
    """
    cost = np.array([[0.1, 0.2], [0.15, 10.0]])
    result = associate(cost, max_cost=20.0)
    assert result.pairs == ((0, 1), (1, 0)), f"退化成贪心了：{result.pairs}"


def test_max_cost_rejects_expensive_pairs():
    """宁可漏配（走遮挡保持），不可错配（污染轨迹）。"""
    cost = np.array([[0.1, 5.0]])
    result = associate(cost, max_cost=1.2)
    assert result.pairs == ((0, 0),)
    assert result.unmatched_detections == (1,)


def test_a_fully_forbidden_row_does_not_raise():
    """某条轨迹的所有配对都被门限拒掉是完全正常的（目标被挡住了）。

    如果禁止用 ``inf`` 表示，scipy 会直接报"矩阵不可行"，一帧正常遮挡就
    把整条流水线打断。
    """
    cost = np.array([[FORBIDDEN_COST, FORBIDDEN_COST], [0.1, 0.2]])
    result = associate(cost, max_cost=1.2)
    assert result.pairs == ((1, 0),)
    assert result.unmatched_tracks == (0,)


def test_pairs_are_sorted_and_indices_are_disjoint():
    rng = np.random.default_rng(7)
    cost = rng.random((6, 6))
    result = associate(cost, max_cost=1.2)
    assert list(result.pairs) == sorted(result.pairs)
    rows = [r for r, _ in result.pairs]
    cols = [c for _, c in result.pairs]
    assert len(set(rows)) == len(rows) and len(set(cols)) == len(cols)
    assert set(rows).isdisjoint(result.unmatched_tracks)
    assert set(cols).isdisjoint(result.unmatched_detections)


def test_association_is_deterministic():
    rng = np.random.default_rng(11)
    cost = rng.random((5, 5))
    assert associate(cost) == associate(cost)


def test_empty_inputs():
    r = associate(np.zeros((0, 3)))
    assert r.pairs == () and r.unmatched_tracks == () and r.unmatched_detections == (0, 1, 2)
    r = associate(np.zeros((2, 0)))
    assert r.pairs == () and r.unmatched_tracks == (0, 1)


def test_a_non_2d_cost_matrix_is_rejected():
    with pytest.raises(ValueError, match="二维"):
        associate(np.zeros(3))


# ======================================================================
# 同槽位冲突：谁代表这个槽位
# ======================================================================
def _track(track_id: int, *, hits: int, matched: bool):
    from jg_radar.track.types import Track

    t = Track(track_id=track_id, created_at=0.0)
    t.hits = hits
    t.matched = matched
    return t


def test_a_track_that_is_being_observed_beats_a_starving_ghost():
    """**核心契约。** 正在被喂观测的轨迹，胜过一条断了粮的幽灵。

    只比累计命中数会得出相反的结论：幽灵跟得久、命中数天然更大，于是它会
    永远压住真正在被跟的那条。对外表现是持续报 ``OCCLUSION_HOLD`` /
    ``BLIND_PREDICTION``，而目标**每一帧都看得见**——不是"把预测当成了测量"，
    而是"把测量当成了丢失"，一样致命。

    这个失效不是假想的：端到端回放里 B3 连续可见，却在第 48 帧
    （χ² 门拒掉了自己那条有效观测）断轨、另起一条。当时幽灵以 48:1 的
    命中数优势赢下了槽位，B3 此后一路报预测态。
    """
    from jg_radar.track.tracker import MultiTracker

    ghost = _track(2, hits=48, matched=False)
    fresh = _track(4, hits=1, matched=True)
    assert MultiTracker._outranks(fresh, ghost) is True
    assert MultiTracker._outranks(ghost, fresh) is False


def test_hit_count_still_decides_when_neither_is_being_observed():
    """两条都没在观测时，退回比命中数——不能因为新加的规则把这条盖掉。"""
    from jg_radar.track.tracker import MultiTracker

    old = _track(1, hits=30, matched=False)
    new = _track(9, hits=3, matched=False)
    assert MultiTracker._outranks(old, new) is True


def test_the_earlier_track_wins_a_total_tie():
    """命中数与观测状态都打平时比 id，保证结果**可复现**。

    依赖 ``self._tracks`` 的顺序会让回放两次给出不同的人。
    """
    from jg_radar.track.tracker import MultiTracker

    first = _track(1, hits=5, matched=True)
    second = _track(2, hits=5, matched=True)
    assert MultiTracker._outranks(first, second) is True
    assert MultiTracker._outranks(second, first) is False
