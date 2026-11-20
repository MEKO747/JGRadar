"""0x0305 发送端：48 字节布局、发送节拍、过期策略、统计。

协议原文（2027 通信协议 V1.0.0 表 1-36）三条硬约束：
数据段 48 字节、频率上限 5 Hz、``x=y=0`` 等于「未发送此机器人坐标」。
2027 每方只有重装/3 号/4 号/哨兵四个有效位，工程和空中是保留位。

验收口径（肥鱼大总管定）：
- 连续 10 秒无信号 → 帧率仍 ≥4.8Hz，且每车写 (0,0)
- 断流 0.6 秒 → ``drop`` 策略下发 (0,0)，``send_stale`` 下发旧值
"""

from __future__ import annotations

import struct

import pytest

from jg_radar.config import FuseConfig
from jg_radar.fuse import Arbiter, PublishLayer, StampedSample
from jg_radar.integration.referee_tx import (
    DEFAULT_RATE_HZ,
    ENEMY_OFFSETS,
    MAX_RATE_HZ,
    OWN_OFFSETS,
    PAYLOAD_LEN,
    RADAR_ROLES,
    RESERVED_RANGES,
    RobotPositionSender,
    build_radar_payload,
    other_side,
    parse_radar_payload,
    slots_within_field,
)
from jg_radar.types import RobotState, SourceKind, SourceSample, default_slot_names

CFG = FuseConfig()
SLOTS = default_slot_names()


# ---------------------------------------------------------------------------
# 辅助：用真实仲裁链产出 FrameOutput
# ---------------------------------------------------------------------------
class Rig:
    """一条最小的真实链路：仲裁器 → 发布层 → FrameOutput。"""

    def __init__(self) -> None:
        self.arb = Arbiter(CFG)
        self.layer = PublishLayer(CFG, slot_names=SLOTS)
        self.seq = 0

    def frame(self, now: float, samples=()):
        decisions = self.arb.decide(samples, now=now)
        frame = self.layer.build(
            decisions, frame_seq=self.seq, captured_at=now, published_at=now
        )
        self.seq += 1
        return frame


def radio_sample(name: str, x: float, y: float, *, observed_at: float) -> StampedSample:
    sample = SourceSample(
        source=SourceKind.RADIO,
        name=name,
        position_cm=(int(x), int(y)),
        state=RobotState.MEASURED,
        confidence=1.0,
        observed_at=observed_at,
        revision=1,
        is_predicted=False,
    )
    return StampedSample(sample=sample, received_at=observed_at)


def collect_sender(**kwargs):
    """装一个假时钟，返回 (sender, 发出的载荷列表, 时钟控制器)。"""
    sent: list[bytes] = []
    now = {"t": 0.0}

    def clock() -> float:
        return now["t"]

    sender = RobotPositionSender(sent.append, clock=clock, **kwargs)
    return sender, sent, now


# ---------------------------------------------------------------------------
# 48 字节布局
# ---------------------------------------------------------------------------
def test_layout_matches_the_2027_offsets():
    slots = {f"B{role}": (100 + index, 200 + index) for index, role in enumerate(RADAR_ROLES)}
    slots.update({f"R{role}": (300 + index, 400 + index) for index, role in enumerate(RADAR_ROLES)})
    payload = build_radar_payload(slots, own_team="R")
    assert len(payload) == PAYLOAD_LEN

    for index, role in enumerate(RADAR_ROLES):
        assert struct.unpack_from("<HH", payload, ENEMY_OFFSETS[role]) == (100 + index, 200 + index)
        assert struct.unpack_from("<HH", payload, OWN_OFFSETS[role]) == (300 + index, 400 + index)


def test_own_team_flips_which_side_goes_first():
    slots = {"R1": (11, 22), "B1": (77, 88)}
    red = parse_radar_payload(build_radar_payload(slots, own_team="red"), own_team="red")
    assert red["B1"] == (77, 88) and red["R1"] == (11, 22)
    blue = parse_radar_payload(build_radar_payload(slots, own_team="blue"), own_team="blue")
    assert blue["R1"] == (11, 22) and blue["B1"] == (77, 88)


def test_reserved_bytes_are_zero():
    payload = build_radar_payload({f"R{r}": (1, 1) for r in RADAR_ROLES}, own_team="R")
    for start, end in RESERVED_RANGES:
        assert payload[start:end] == b"\x00" * (end - start), f"保留位 {start}-{end} 不全是 0"


def test_missing_robot_is_written_as_zero_zero():
    payload = build_radar_payload({"R1": (1000, 700)}, own_team="R")
    decoded = parse_radar_payload(payload, own_team="R")
    assert decoded["R1"] == (1000, 700)
    assert decoded["R3"] == (0, 0)
    assert decoded["B1"] == (0, 0), "对方没有坐标时也必须是 (0,0)，那是「未发送」"


def test_unknown_role_and_side_are_ignored():
    payload = build_radar_payload({"R2": (1, 2), "R6": (3, 4), "X1": (5, 6)}, own_team="R")
    assert payload == bytes(PAYLOAD_LEN), "工程/空中在 2027 是保留位，不该占任何字节"


def test_coordinates_are_clamped_into_uint16():
    payload = build_radar_payload({"R1": (-5, 70000)}, own_team="R")
    assert parse_radar_payload(payload, own_team="R")["R1"] == (0, 65535)


def test_other_side_understands_both_spellings():
    assert other_side("R") == "B"
    assert other_side("red") == "B"
    assert other_side("blue") == "R"
    with pytest.raises(ValueError):
        other_side("green")


def test_out_of_field_coordinates_are_treated_as_missing():
    cleaned = slots_within_field({"R1": (9999, 700), "R3": (1000, 700), "R4": None})
    assert cleaned == {"R1": None, "R3": (1000.0, 700.0), "R4": None}


# ---------------------------------------------------------------------------
# 发送节拍
# ---------------------------------------------------------------------------
def test_rate_is_clamped_to_the_protocol_ceiling():
    sender, _sent, _now = collect_sender(rate_hz=8.0)
    assert sender.rate_hz == MAX_RATE_HZ
    assert sender.clamped is True
    assert DEFAULT_RATE_HZ < MAX_RATE_HZ, "默认值要留余量，别贴上限"


def test_silence_keeps_sending_at_the_nominal_rate_with_all_zeros():
    """验收 1：连续 10 秒没有任何输入，帧率仍 ≥4.8Hz，每车 (0,0)。"""
    sender, sent, now = collect_sender(rate_hz=4.8, own_team="R")
    rig = Rig()

    # 主循环 20Hz：每 50ms 更新一次（永远是全 MISSING 的帧）
    for step in range(int(10.0 / 0.05)):
        now["t"] = step * 0.05
        sender.update(rig.frame(now["t"]))
        sender.tick()

    rate = sender.stats.effective_rate_hz(now["t"])
    assert rate >= 4.8, f"10 秒静默后实际发送 {rate:.2f} Hz，低于 4.8"
    assert sender.stats.frames_sent >= 48, f"10 秒只发了 {sender.stats.frames_sent} 帧"
    for payload in sent:
        assert payload == bytes(PAYLOAD_LEN), "没有坐标时整帧都必须是 (0,0)"


def test_jittery_loop_still_hits_the_average_rate():
    """主循环抖动到 70ms 一拍时，平均频率也不能掉下来。"""
    sender, _sent, now = collect_sender(rate_hz=4.8)
    rig = Rig()
    step = 0.0
    while step < 10.0:
        now["t"] = step
        sender.update(rig.frame(step))
        sender.tick()
        step += 0.07
    assert sender.stats.effective_rate_hz() >= 4.6


def test_stalled_loop_does_not_burst_catch_up():
    """主循环卡了 1 秒：醒来只补一帧，不许把欠的帧一次性喷出去。"""
    sender, sent, now = collect_sender(rate_hz=4.8)
    rig = Rig()
    now["t"] = 0.0
    sender.update(rig.frame(0.0))
    assert sender.tick() is True
    now["t"] = 1.0
    assert sender.tick() is True
    now["t"] = 1.01
    assert sender.tick() is False
    assert len(sent) == 2


def test_tick_before_due_does_nothing():
    sender, sent, now = collect_sender(rate_hz=4.8)
    rig = Rig()
    sender.update(rig.frame(0.0))
    assert sender.tick(0.0) is True
    assert sender.tick(0.1) is False
    assert len(sent) == 1


# ---------------------------------------------------------------------------
# 过期策略
# ---------------------------------------------------------------------------
def _stale_scenario(policy: str):
    """无线电在 t=0 报了一次位置，之后断流到 t=0.6。"""
    rig = Rig()
    sender, sent, _now = collect_sender(rate_hz=4.8, own_team="R",
                                        stale_policy=policy, stale_after_s=0.5)
    sender.update(rig.frame(0.0, [radio_sample("B1", 1000, 700, observed_at=0.0)]))
    # 断流 = 缓冲里还是那条旧样本，每帧都被重新交付一次（这正是现场的形态）
    frame = rig.frame(0.6, [radio_sample("B1", 1000, 700, observed_at=0.0)])
    assert frame.robots["B1"].position_cm is not None, "0.6 秒的坐标还没被仲裁器否掉"
    sender.update(frame)
    assert sender.tick(0.6) is True
    return sender, parse_radar_payload(sent[-1], own_team="R")


def test_drop_policy_zeroes_a_stale_coordinate():
    """验收 2a：断流 0.6s，drop 策略下该车写 (0,0)。"""
    sender, decoded = _stale_scenario("drop")
    assert decoded["B1"] == (0, 0)
    assert sender.stats.dropped.get("B1") == 1
    assert sender.stats.sent_stale == {}


def test_send_stale_policy_keeps_the_old_coordinate_and_counts_it():
    """验收 2b：同一个 0.6s 断流，send_stale 下照发旧值并记账。"""
    sender, decoded = _stale_scenario("send_stale")
    assert decoded["B1"] == (1000, 700)
    assert sender.stats.sent_stale.get("B1") == 1
    assert sender.stats.last_max_age_s == pytest.approx(0.6)


def test_fresh_coordinate_is_never_dropped():
    rig = Rig()
    sender, _sent, _now = collect_sender(rate_hz=4.8, own_team="R")
    sender.update(rig.frame(0.3, [radio_sample("B1", 1200, 800, observed_at=0.3)]))
    assert sender.tick(0.35) is True
    decoded = parse_radar_payload(sender.last_payload, own_team="R")
    assert decoded["B1"] == (1200, 800)
    assert sender.stats.dropped.get("B1") is None


def test_position_ages_out_even_if_updates_keep_arriving():
    """发布停不下来但数据不更新时，年龄要自己长——不能拿旧快照一直发。"""
    rig = Rig()
    sender, _sent, _now = collect_sender(rate_hz=4.8, own_team="R", stale_after_s=0.5)
    old = radio_sample("B1", 1000, 700, observed_at=0.0)
    sender.update(rig.frame(0.0, [old]))
    for step in range(1, 20):
        now = step * 0.1
        sender.update(rig.frame(now, [old]))   # 同样的旧样本被反复交付
        sender.tick(now)
    decoded = parse_radar_payload(sender.last_payload, own_team="R")
    assert decoded["B1"] == (0, 0), "1.9 秒前的坐标必须已经被丢掉"


def test_unknown_stale_policy_is_rejected():
    with pytest.raises(ValueError, match="stale_policy"):
        RobotPositionSender(lambda _p: None, stale_policy="keep_forever")


def test_stats_snapshot_is_json_friendly():
    sender, _sent, now = collect_sender(rate_hz=4.8)
    rig = Rig()
    now["t"] = 0.0
    sender.update(rig.frame(0.0))
    sender.tick()
    now["t"] = 0.25
    sender.tick()
    snapshot = sender.snapshot()
    assert snapshot["frames_sent"] == 2
    assert snapshot["rate_hz"] == 4.8
    assert snapshot["stale_policy"] == "drop"
    assert isinstance(snapshot["dropped"], dict)
