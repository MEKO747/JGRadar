"""身份投票：车类别定人，数字只能印证。

方案 §十 的关键再框定——雷达不需要每帧读出数字，身份只需建立一次。
所以这里最要紧的两条是：**错的证据不能太快翻盘**，以及**横跳的读数不能
变成横跳的身份**。
"""

from __future__ import annotations

import pytest

from jg_radar.config import load_config
from jg_radar.track import IdentityVote, MultiTracker, TrackObservation, corroborated_votes

DT = 0.05


def obs(x, y, world, *, cls="B1", digit="", w=80.0, h=40.0):
    return TrackObservation(
        bbox=(x - w / 2, y - h / 2, x + w / 2, y + h / 2),
        world_cm=world, confidence=0.9, car_class=cls, digit=digit,
    )


def _tracker():
    return MultiTracker(load_config("config/default.yaml").track)


# ======================================================================
# 一次观测折成几票
# ======================================================================
def test_car_class_alone_is_a_full_vote():
    votes = corroborated_votes("B1", "")
    assert len(votes) == 1
    assert votes[0].label == "B1"
    assert votes[0].weight == pytest.approx(1.0)


def test_matching_digit_strengthens_the_vote():
    """数字与车类别一致 = 两份独立证据。"""
    plain = corroborated_votes("B1", "")[0]
    agree = corroborated_votes("B1", "1")[0]
    assert agree.label == "B1"
    assert agree.weight > plain.weight


def test_conflicting_digit_weakens_but_does_not_overrule():
    """数字在 20–30m 外只有 6–8 像素（方案 §十三 风险 2），不该翻掉车类别。"""
    votes = corroborated_votes("B1", "3")
    by_label = {v.label: v.weight for v in votes}
    assert by_label["B1"] < 1.0, "矛盾时车类别的票没被削弱"
    assert by_label["B1"] > by_label["B3"], "数字反客为主了"
    assert by_label["B3"] > 0.0, "数字该留一票，反复出现时才有翻盘的机会"


def test_a_digit_alone_cannot_name_a_slot():
    """数字里没有阵营信息：看到"1"不知道是 R1 还是 B1。

    硬猜一半概率是错的，而身份错一次要好几秒才纠得回来——宁可不投。
    """
    assert corroborated_votes("", "1") == []
    assert corroborated_votes(None, "5") == []


def test_no_evidence_no_votes():
    assert corroborated_votes("", "") == []


# ======================================================================
# 票箱
# ======================================================================
def test_votes_decay_by_half_life():
    v = IdentityVote(half_life_s=2.0)
    v.add("B1", 1.0, now=0.0)
    v.decay_to(2.0)
    assert v.snapshot()["B1"] == pytest.approx(0.5)
    v.decay_to(4.0)
    assert v.snapshot()["B1"] == pytest.approx(0.25)


def test_decay_needs_time_to_pass():
    """同一时刻重复调用不该衰减——否则一帧里投几票就把旧票清空了。"""
    v = IdentityVote(half_life_s=2.0)
    v.add("B1", 1.0, now=0.0)
    v.decay_to(0.0)
    v.decay_to(0.0)
    assert v.snapshot()["B1"] == pytest.approx(1.0)


def test_a_clock_that_goes_backwards_does_not_advance_the_baseline():
    """时间戳倒退是上游的 bug。推进基准时刻会让后面的 dt 变小、衰减偏慢，
    等于让坏时间戳悄悄削弱衰减的作用。"""
    v = IdentityVote(half_life_s=2.0)
    v.add("B1", 1.0, now=1.0)
    v.decay_to(0.5)  # 倒退
    assert v.snapshot()["B1"] == pytest.approx(1.0)
    v.decay_to(3.0)  # 从 1.0 算起，正好一个半衰期
    assert v.snapshot()["B1"] == pytest.approx(0.5)


def test_nonpositive_weights_are_dropped():
    v = IdentityVote()
    v.add("B1", 0.0, now=0.0)
    v.add("B2", -5.0, now=0.0)
    assert len(v) == 0 and v.best is None


def test_share_is_a_fraction_of_total_weight():
    v = IdentityVote()
    v.add("B1", 3.0, now=0.0)
    v.add("B2", 1.0, now=0.0)
    assert v.share("B1") == pytest.approx(0.75)
    assert v.share("B9") == pytest.approx(0.0)


def test_an_empty_box_has_no_best():
    assert IdentityVote().best is None


def test_the_first_label_wins_a_tie():
    """平局按首次出现顺序裁决：先看到的先赢，且完全确定。"""
    v = IdentityVote()
    v.add("B2", 1.0, now=0.0)
    v.add("B1", 1.0, now=0.0)
    assert v.leader[0] == "B2"


# ======================================================================
# 粘滞：防止身份横跳
# ======================================================================
def test_alternating_reads_do_not_flip_the_label_every_frame():
    """检测器在 B1/B2 之间反复横跳时，对外身份不许跟着跳。

    没有粘滞的话票权会变成 ``B1=2.90, B2=1.93`` → 下一帧 ``B1=2.85, B2=2.90``，
    领跑者每帧都换。对外表现是雷达站反复改口，比一开始就不确定还糟。
    """
    v = IdentityVote(half_life_s=2.0, switch_margin=0.15)
    labels = []
    for i in range(60):
        v.add("B1" if i % 2 == 0 else "B2", 1.0, now=i * DT)
        labels.append(v.best[0])
    assert set(labels) == {"B1"}, f"身份横跳：{sorted(set(labels))}"


def test_a_genuine_identity_change_still_gets_through():
    """粘滞不能变成锁死：真的换了人，票权持续累积后必须能改过来。"""
    v = IdentityVote(half_life_s=10.0, switch_margin=0.15)
    for i in range(5):
        v.add("B1", 1.0, now=i * DT)
    assert v.best[0] == "B1"

    for i in range(20):  # 之后一路都是 B2
        v.add("B2", 1.0, now=(5 + i) * DT)
    assert v.best[0] == "B2"


def test_the_first_identity_is_adopted_without_a_margin():
    """首次定身份不需要越过幅度，否则新轨迹要等好几帧才敢说自己是谁。"""
    v = IdentityVote(switch_margin=5.0)
    v.add("B1", 1.0, now=0.0)
    assert v.best == ("B1", pytest.approx(1.0))


# ======================================================================
# 链路上
# ======================================================================
def _establish(tracker, *, cls="B1", digit="", n=4, start=0.0):
    now = start
    for _ in range(n):
        tracker.step([obs(500.0, 600.0, (500.0, 750.0), cls=cls, digit=digit)], now=now)
        now += DT
    return now


def test_an_unknown_identity_is_never_published():
    """认不出是谁就不报——**位置准而不知道是谁，等于没有用**，
    报错人比不报更糟。"""
    tracker = _tracker()
    _establish(tracker, cls="")  # 车类别没读出来
    out = tracker.step([obs(500.0, 600.0, (500.0, 750.0), cls="")], now=4 * DT)
    assert out == {}
    assert len(tracker.tracks) == 1, "轨迹本身应该在，只是没有对外输出"


def test_a_label_outside_the_slot_list_is_not_published():
    """类别表里出现了一个不是槽位名的标签（比如权重换了、类别表没更新）。"""
    tracker = _tracker()
    _establish(tracker, cls="banana")
    out = tracker.step([obs(500.0, 600.0, (500.0, 750.0), cls="banana")], now=4 * DT)
    assert out == {}


def test_digits_corroborate_without_renaming():
    """数字一直和车类别一致时，身份不变，只是更稳。"""
    tracker = _tracker()
    now = _establish(tracker, cls="B2", digit="2", n=10)
    for _ in range(20):
        out = tracker.step([obs(500.0, 600.0, (500.0, 750.0), cls="B2", digit="2")], now=now)
        now += DT
    assert set(out) == {"B2"}


def test_a_persistently_conflicting_digit_does_not_steal_the_slot():
    """整场都在矛盾（远处把 2 读成 3）：车类别必须守住槽位。

    数字的暗示票权重小得多，累计也不会翻盘——这正是"数字只用来印证"的意思。
    """
    tracker = _tracker()
    now = _establish(tracker, cls="B1", digit="3", n=2)
    for _ in range(60):
        out = tracker.step([obs(500.0, 600.0, (500.0, 750.0), cls="B1", digit="3")], now=now)
        now += DT
    assert set(out) == {"B1"}, f"数字把身份偷走了：{set(out)}"


def test_identity_survives_an_occlusion():
    """方案 §十：身份只需建立一次，之后靠跟踪维持。遮挡不该让它忘了自己是谁。"""
    tracker = _tracker()
    now = _establish(tracker, cls="R4", n=10)
    for _ in range(20):  # 1 秒遮挡
        tracker.step([], now=now)
        now += DT
    out = tracker.step([obs(500.0, 600.0, (500.0, 750.0), cls="R4")], now=now)
    assert set(out) == {"R4"}
