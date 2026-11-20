"""仲裁器的新鲜度标记：``age_s`` / ``stale`` / ``source``。

2027 规则是「每累计 0.5 秒收不到数据记一次错误」，所以「有数据但在变旧」
和「数据彻底没了」必须是两件看得见的事，不能一刀切。这一组测试钉死三档：

- 0.1~0.2 秒：照常上报，``stale=False``
- 0.5 秒以上：**照常上报**，但 ``stale=True`` 且 ``age_s`` 要如实反映年龄
- 1.0 秒以上：精度上界越过 ``max_position_error_cm``，仍被否决（不变）

最后一条是回归线：改标记不能顺手把否决改了。
"""

from __future__ import annotations

import pytest

from jg_radar.config import FuseConfig
from jg_radar.fuse import Arbiter, PublishLayer, StampedSample
from jg_radar.types import RobotState, SourceKind, SourceSample

CFG = FuseConfig()


def radio(observed_at: float, *, x: float = 1000.0, y: float = 700.0) -> SourceSample:
    return SourceSample(
        source=SourceKind.RADIO,
        name="R1",
        position_cm=(int(x), int(y)),
        state=RobotState.MEASURED,
        confidence=1.0,
        observed_at=observed_at,
        revision=1,
        is_predicted=False,
    )


def decide_at(arb: Arbiter, now: float, observed_at: float):
    """在 ``now`` 时刻，缓冲里躺着一条 ``observed_at`` 采到的报文。"""
    stamped = StampedSample(sample=radio(observed_at), received_at=observed_at)
    return arb.decide([stamped], now=now)["R1"]


def test_fresh_sample_is_not_stale():
    arb = Arbiter(CFG)
    decision = decide_at(arb, now=1.0, observed_at=1.0)
    assert decision.tier == "radio_fresh"
    assert decision.stale is False
    assert decision.age_s == pytest.approx(0.0)
    assert decision.source is SourceKind.RADIO


def test_half_second_of_silence_is_marked_stale_but_still_published():
    """0.6 秒断流：降档到 radio_stale，**照发**，但要标 stale 并报出真实年龄。"""
    arb = Arbiter(CFG)
    decide_at(arb, now=1.0, observed_at=1.0)

    decision = decide_at(arb, now=1.6, observed_at=1.0)
    assert decision.tier == "radio_stale", "0.6 秒后应该已经掉出 fresh 窗"
    assert decision.sample is not None, "标 stale 不等于否决，坐标必须照常发出去"
    assert decision.stale is True
    assert decision.age_s is not None and decision.age_s >= 0.5
    assert decision.age_s == pytest.approx(0.6)


def test_stale_mark_does_not_replace_the_one_second_veto():
    """1.2 秒断流仍然被精度上界否决——标记归标记，否决归否决。"""
    arb = Arbiter(CFG)
    decide_at(arb, now=1.0, observed_at=1.0)

    decision = decide_at(arb, now=2.2, observed_at=1.0)
    assert decision.sample is None, "1.2 秒 × 500cm/s = 600cm 超过 500cm，必须否掉"
    assert decision.tier is None
    assert decision.age_s is None
    assert arb.vetoed >= 1, "被精度上界否掉的候选要计进 vetoed，否则外部看不见"


def test_mark_point_is_configurable():
    """标记点来自配置，不是写死的常量。"""
    arb = Arbiter(FuseConfig(stale_mark_s=0.3))
    decide_at(arb, now=1.0, observed_at=1.0)
    assert decide_at(arb, now=1.35, observed_at=1.0).stale is True


def test_publish_layer_surfaces_age_and_stale():
    """发布层把年龄和 stale 摊进 diagnostics，0x0305 发送端不必自己再算。"""
    arb = Arbiter(CFG)
    layer = PublishLayer(CFG, slot_names=("R1",))
    assert layer is not None

    decide_at(arb, now=1.0, observed_at=1.0)
    stamped = StampedSample(sample=radio(1.0), received_at=1.0)
    decisions = arb.decide([stamped], now=1.6)
    frame = layer.build(
        decisions, frame_seq=0, captured_at=1.6, published_at=1.6
    )
    assert frame.diagnostics["R1_age_s"] == pytest.approx(0.6)
    assert frame.diagnostics["R1_stale"] == 1.0
    assert frame.diagnostics["stale_slots"] == 1.0
    assert frame.robots["R1"].source is SourceKind.RADIO


def test_publish_layer_marks_stale_slots_zero_when_fresh():
    arb = Arbiter(CFG)
    layer = PublishLayer(CFG, slot_names=("R1",))
    stamped = StampedSample(sample=radio(1.0), received_at=1.0)
    decisions = arb.decide([stamped], now=1.0)
    frame = layer.build(decisions, frame_seq=0, captured_at=1.0, published_at=1.0)
    assert frame.diagnostics["stale_slots"] == 0.0
    assert "R1_stale" not in frame.diagnostics
