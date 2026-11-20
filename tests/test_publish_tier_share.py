"""发布层的档位统计：一次运行自己回答"预测到底占多少"。

admin 要看 ``vision_blind`` 档占总槽位帧的比例，用来决定航点预测还值不值得
继续投入。这个数字以前只能靠人翻帧，现在由发布层在成帧时顺手累加——不额外
扫一遍数据，也不会和实际输出对不上。
"""

from __future__ import annotations

import pytest

from jg_radar.config import FuseConfig
from jg_radar.fuse import Arbiter, PublishLayer, StampedSample
from jg_radar.fuse.arbiter import TIER_RADIO_FRESH, TIER_VISION_BLIND, SlotDecision
from jg_radar.types import RobotState, SourceKind, SourceSample

CFG = FuseConfig()


def radio(observed_at: float, *, x: float = 1000.0, y: float = 700.0) -> SourceSample:
    return SourceSample(
        name="R1",
        position_cm=(x, y),
        observed_at=observed_at,
        source=SourceKind.RADIO,
        state=RobotState.MEASURED,
        confidence=1.0,
        revision=1,
        is_predicted=False,
    )


def vision(observed_at: float, *, x: float = 1000.0, y: float = 700.0) -> SourceSample:
    return SourceSample(
        name="R1",
        position_cm=(x, y),
        observed_at=observed_at,
        source=SourceKind.VISION,
        state=RobotState.MEASURED,
        confidence=1.0,
        revision=1,
        is_predicted=False,
    )


def build_frame(layer: PublishLayer, decision, *, seq: int = 0):
    return layer.build(
        {"R1": decision} if decision is not None else {},
        frame_seq=seq,
        captured_at=1.0,
        published_at=1.0,
    )


def test_share_counts_the_adopted_tier():
    layer = PublishLayer(CFG, slot_names=("R1",))
    arb = Arbiter(CFG)
    decisions = arb.decide([StampedSample(sample=radio(1.0), received_at=1.0)], now=1.0)
    layer.build(decisions, frame_seq=0, captured_at=1.0, published_at=1.0)
    share = layer.tier_share()
    assert share == {TIER_RADIO_FRESH: 1.0}


def test_missing_slots_are_counted_as_missing():
    """没有候选的槽位必须出现在分母里，否则比例会虚高。"""
    layer = PublishLayer(CFG, slot_names=("R1",))
    build_frame(layer, None)
    assert layer.tier_share() == {"missing": 1.0}


def test_share_mixes_tiers_over_several_frames():
    layer = PublishLayer(CFG, slot_names=("R1",))
    fresh = SlotDecision(name="R1", sample=radio(1.0), tier=TIER_RADIO_FRESH, switched=False)
    blind = SlotDecision(name="R1", sample=vision(1.0), tier=TIER_VISION_BLIND, switched=True)
    for index, decision in enumerate((fresh, fresh, blind, blind)):
        build_frame(layer, decision, seq=index)
    share = layer.tier_share()
    assert share[TIER_VISION_BLIND] == pytest.approx(0.5)
    assert share[TIER_RADIO_FRESH] == pytest.approx(0.5)
    assert sum(share.values()) == pytest.approx(1.0)


def test_share_is_empty_before_any_frame():
    assert PublishLayer(CFG, slot_names=("R1",)).tier_share() == {}


def test_reset_clears_the_counters():
    layer = PublishLayer(CFG, slot_names=("R1",))
    build_frame(layer, None)
    layer.reset()
    assert layer.tier_share() == {}
