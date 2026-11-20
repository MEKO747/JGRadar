"""P 递推的验收线：规则手册 §5.7.4 那段算例必须逐拍一致。

规则手册 112 页写得很细，等于免费给了我们一组测试向量：

- 5Hz、100% 准确 15 拍 → ``1, 3, 6, 10, 15 …``，第 3.0 秒到 **120**
- 随后 0% 准确 → ``119.2, 117.6, 115.2, 112, 108``
- 再连续 1.0 秒不发 → ``103.2, 97.6``（"额外连续两次错误"）
- 再两次准确 → ``98.6, 100.6``

任何"看起来更合理"的修改都会把这串数字改坏，所以这些值全部钉死。
"""

from __future__ import annotations

import math

import pytest

from jg_radar.eval.marking_progress import (
    ACCURATE_MAX_M,
    HALF_MAX_M,
    INTERRUPT_PERIOD_S,
    P_MAX,
    Judgement,
    MarkingTracker,
    ProgressState,
    classify_distance,
)

# ---------------------------------------------------------------------------
# 分级
# ---------------------------------------------------------------------------
def test_distance_thresholds_follow_the_rules():
    assert classify_distance(0.0) is Judgement.ACCURATE
    assert classify_distance(0.79) is Judgement.ACCURATE
    # 0.8 是半准确的闭下界：规则写 0.8m ≤ d < 1.6m
    assert classify_distance(0.8) is Judgement.HALF
    assert classify_distance(1.59) is Judgement.HALF
    assert classify_distance(1.6) is Judgement.ERROR
    assert classify_distance(99.0) is Judgement.ERROR


def test_thresholds_are_meters():
    assert (ACCURATE_MAX_M, HALF_MAX_M) == (0.8, 1.6)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_classify_rejects_impossible_distances(bad):
    with pytest.raises(ValueError):
        classify_distance(bad)


# ---------------------------------------------------------------------------
# x 的更新逻辑（表 5-29）
# ---------------------------------------------------------------------------
def test_accurate_accumulates_while_it_stays_accurate():
    state = ProgressState()
    assert [state.apply(Judgement.ACCURATE) for _ in range(5)] == [1.0, 3.0, 6.0, 10.0, 15.0]
    assert state.x == 5.0


def test_half_accurate_adds_half_and_keeps_accumulating():
    state = ProgressState()
    assert state.apply(Judgement.ACCURATE) == 1.0
    assert state.apply(Judgement.HALF) == 2.5  # x 1.0→1.5
    assert state.apply(Judgement.HALF) == 4.5  # x 1.5→2.0
    assert state.x == 2.0


def test_error_after_accurate_restarts_at_minus_0_8():
    state = ProgressState()
    for _ in range(5):
        state.apply(Judgement.ACCURATE)
    assert state.apply(Judgement.ERROR) == pytest.approx(14.2)  # 15 − 0.8
    assert state.apply(Judgement.ERROR) == pytest.approx(12.6)  # x 继续 −0.8
    assert state.x == pytest.approx(-1.6)


def test_accurate_after_error_restarts_x_at_one():
    """规则：「若上一次标记为准确或者半准确…否则 x=1.0」。"""
    state = ProgressState()
    assert state.apply(Judgement.ERROR) == 0.0  # 起点 0，限幅到 0
    assert state.apply(Judgement.ACCURATE) == 1.0
    # 上一次是准确 → 半准确走"＋0.5 累积"：x 1.0→1.5，P 1.0→2.5
    assert state.apply(Judgement.HALF) == 2.5


def test_p_is_clamped_to_the_rule_range():
    state = ProgressState()
    for _ in range(200):
        state.apply(Judgement.ACCURATE)
    assert state.p == P_MAX
    state = ProgressState(p=1.0, x=0.0, last=Judgement.ERROR)
    for _ in range(20):
        state.apply(Judgement.ERROR)
    assert state.p == 0.0


def test_interrupt_uses_the_same_branch_as_error():
    state = ProgressState()
    for _ in range(3):
        state.apply(Judgement.ACCURATE)
    assert state.apply(Judgement.INTERRUPT) == pytest.approx(5.2)  # x 3.0→−0.8
    assert state.apply(Judgement.ERROR) == pytest.approx(3.6)  # 上一次是"错误/中断"→ 继续 −0.8
    assert state.x == pytest.approx(-1.6)


# ---------------------------------------------------------------------------
# 规则手册里的那段算例，逐拍复现
# ---------------------------------------------------------------------------
def test_handbook_worked_example():
    """5Hz：15 拍准确 → 5 拍错误 → 断流 1 秒 → 2 拍准确。"""
    tracker = MarkingTracker()
    t = 0.0
    accurate = []
    for _ in range(15):
        t += 0.2
        tracker.data(t, 0.3)
    accurate = [sample.p for sample in tracker.samples]
    assert accurate[:5] == [1.0, 3.0, 6.0, 10.0, 15.0]
    assert accurate[-1] == pytest.approx(120.0)
    assert tracker.samples[-1].t == pytest.approx(3.0)

    wrong = []
    for _ in range(5):
        t += 0.2
        tracker.data(t, 3.0)  # d ≥ 1.6m → 错误
        wrong.append(tracker.samples[-1].p)
    assert wrong == [pytest.approx(119.2), pytest.approx(117.6), pytest.approx(115.2),
                     pytest.approx(112.0), pytest.approx(108.0)]

    # 连续 1.0 秒不发：每累计 0.5 秒算一次错误 → 103.2、97.6
    tracker.silence_until(5.0)
    silent = [sample.p for sample in tracker.samples if sample.judgement is Judgement.INTERRUPT]
    assert silent == [pytest.approx(103.2), pytest.approx(97.6)]

    # 再两次准确：x 从 1.0 重新累积
    t = 5.2
    tracker.data(t, 0.3)
    assert tracker.samples[-1].p == pytest.approx(98.6)
    t = 5.4
    tracker.data(t, 0.3)
    assert tracker.samples[-1].p == pytest.approx(100.6)


def test_interrupt_period_is_half_a_second():
    tracker = MarkingTracker()
    tracker.data(1.0, 0.2)
    # 1.4 秒时还没有满一个周期
    tracker.silence_until(1.4)
    assert [s.judgement for s in tracker.samples] == [Judgement.ACCURATE]
    # 1.5 秒正好一个周期
    tracker.silence_until(1.5)
    assert [s.judgement for s in tracker.samples][-1] is Judgement.INTERRUPT
    # 2.6 秒时已补到 2.5（1.5/2.0/2.5），2.6 秒不会再补
    tracker.silence_until(2.6)
    interrupts = [s for s in tracker.samples if s.judgement is Judgement.INTERRUPT]
    assert [s.t for s in interrupts] == [1.5, 2.0, 2.5]
    assert INTERRUPT_PERIOD_S == 0.5


def test_interrupt_timer_restarts_from_the_last_data():
    """收到数据就把中断计时清零，否则正常 4.7Hz 收发也会被补错误。"""
    tracker = MarkingTracker()
    for index in range(20):
        tracker.data(index * 0.2128, 0.2)
    assert not any(sample.judgement is Judgement.INTERRUPT for sample in tracker.samples)


def test_short_gap_under_half_a_second_costs_nothing():
    """0.4 秒的抖动不发中断判定——规则的门槛是累计 0.5 秒。"""
    tracker = MarkingTracker()
    tracker.data(1.0, 0.2)
    tracker.data(1.4, 0.2)
    assert not any(sample.judgement is Judgement.INTERRUPT for sample in tracker.samples)


def test_first_time_at_reports_the_crossing():
    tracker = MarkingTracker()
    t = 0.0
    for _ in range(15):
        t += 0.2
        tracker.data(t, 0.2)
    assert math.isclose(tracker.first_time_at(100.0) or -1, 2.8, abs_tol=1e-6)
    assert math.isclose(tracker.first_time_at(120.0) or -1, 3.0, abs_tol=1e-6)
    assert tracker.first_time_at(151.0) is None
