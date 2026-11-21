"""0x0305 发送端：48 字节布局、发送节拍、过期策略、统计。

协议原文（2027 通信协议 V1.0.0 表 1-36）三条硬约束：
数据段 48 字节、频率上限 5 Hz、``x=y=0`` 等于「未发送此机器人坐标」。
2027 每方只有重装/3 号/4 号/哨兵四个有效位，工程和空中是保留位。

验收口径（肥鱼大总管定）：
- 连续 10 秒无信号 → 帧率仍 ≥4.7Hz（默认值 4.8 Hz，比判据和 5Hz 上限都留余量），且每车写 (0,0)
- 断流 0.6 秒 → ``drop`` 策略下发 (0,0)，``send_stale`` 下发旧值
"""

from __future__ import annotations

import json
import struct

import pytest

from jg_radar.config import FuseConfig
from jg_radar.fuse import Arbiter, PublishLayer, StampedSample
from jg_radar.integration.referee_tx import (
    DEFAULT_RATE_HZ,
    ENEMY_OFFSETS,
    MAX_RATE_HZ,
    MAX_FRAMES_PER_WINDOW,
    RATE_SETTLE_S,
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

    def send(payload: bytes) -> bool:
        """假串口：记下来并**明确回报写成功**（发送端只认 True）。"""
        sent.append(payload)
        return True

    sender = RobotPositionSender(send, clock=clock, **kwargs)
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
    """验收 1：连续 10 秒没有任何输入，帧率仍 ≥4.7Hz（按默认值跑），每车 (0,0)。"""
    sender, sent, now = collect_sender(rate_hz=DEFAULT_RATE_HZ, own_team="R")
    rig = Rig()

    # 主循环 20Hz：每 50ms 更新一次（永远是全 MISSING 的帧）
    for step in range(int(10.0 / 0.05)):
        now["t"] = step * 0.05
        sender.update(rig.frame(now["t"]))
        sender.tick()

    rate = sender.stats.effective_rate_hz(now["t"])
    assert rate >= 4.7, f"10 秒静默后实际发送 {rate:.2f} Hz，低于 4.7"
    assert sender.stats.frames_sent >= 47, f"10 秒只发了 {sender.stats.frames_sent} 帧"
    assert sender.stats.max_frames_in_window <= int(MAX_RATE_HZ), "1 秒窗口内不得超过 5 帧"
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


def test_default_rate_leaves_headroom_below_the_protocol_cap():
    """默认频率必须低于上限：贴线打的话某些 1 秒窗口正好 5 帧，零余量。"""
    sender, _sent, _now = collect_sender()
    assert sender.rate_hz == DEFAULT_RATE_HZ
    assert DEFAULT_RATE_HZ < MAX_RATE_HZ


def test_window_peak_is_reported_for_field_checks():
    """统计里要有「任意 1 秒窗口最多发了几帧」，现场一眼看出有没有贴线。"""
    sender, _sent, now = collect_sender(rate_hz=4.7)
    rig = Rig()
    for step in range(int(10.0 / 0.05)):
        now["t"] = step * 0.05
        sender.update(rig.frame(now["t"]))
        sender.tick()
    peak = sender.stats.max_frames_in_window
    assert 4 <= peak <= 5, f"1 秒窗口峰值 {peak} 帧，按 4.7Hz 应该在 4~5"
    assert peak <= MAX_RATE_HZ, "任何 1 秒窗口都不能超过 5 帧"
    assert sender.snapshot()["max_frames_1s"] == peak


# ---------------------------------------------------------------------------
# 故障注入：写失败必须被看见，不能算成「发送」
# ---------------------------------------------------------------------------
def test_write_failure_is_never_counted_as_sent():
    """串口抛异常：组帧照样进行，但 frames_sent 必须是 0。"""
    now = {"t": 0.0}

    def broken(_payload: bytes) -> bool:
        raise OSError("device unplugged")

    sender = RobotPositionSender(broken, rate_hz=4.8, clock=lambda: now["t"])
    rig = Rig()
    for step in range(20):
        now["t"] = step * 0.05
        sender.update(rig.frame(now["t"]))
        assert sender.tick() in (True, False)
    assert sender.stats.frames_composed > 0
    assert sender.stats.frames_sent == 0
    assert sender.stats.frames_failed == sender.stats.frames_composed
    assert sender.last_write_ok is False
    assert "device unplugged" in (sender.last_write_error or "")
    assert sender.stats.effective_rate_hz(now["t"]) == 0.0


def test_a_send_callback_that_returns_nothing_is_not_evidence():
    """返回 None = 没回报，不是成功。旧代码正是在这里把失败当成发送。"""
    now = {"t": 0.0}
    sender = RobotPositionSender(lambda _payload: None, rate_hz=4.8,
                                 clock=lambda: now["t"])
    rig = Rig()
    for step in range(10):
        now["t"] = step * 0.05
        sender.update(rig.frame(now["t"]))
        sender.tick()
    assert sender.stats.frames_composed > 0
    assert sender.stats.frames_sent == 0
    assert "没有返回成功标记" in (sender.last_write_error or "")


def test_one_second_window_is_a_hard_gate_not_just_a_statistic():
    """故障注入：主循环晚到一拍后紧接着又来一拍，两帧只隔几十毫秒。

    旧节拍器只排程不拦截，这种挤兑能在 1 秒里塞进 6 帧（用户复现的那条）。
    现在发之前查窗口，满 5 帧就丢这一拍。
    """
    # 4.7Hz：due 0.8512 时主循环在 1.05 才醒（晚了 0.199 < 一个周期），
    # 于是 1.08 那一拍几乎贴着发（两帧只隔 30ms）——典型的挤兑。
    # 旧节拍器在这个序列里 1.05~1.93 的 0.88 秒内会发 6 帧。
    ticks = [0.0, 0.22, 0.43, 0.64, 1.05, 1.08, 1.29, 1.50, 1.72, 1.93, 2.14]
    sent_times = []
    now = {"t": 0.0}

    def send(_payload: bytes) -> bool:
        sent_times.append(now["t"])
        return True

    sender = RobotPositionSender(send, rate_hz=4.7, clock=lambda: now["t"])
    rig = Rig()
    for stamp in ticks:
        now["t"] = stamp
        sender.update(rig.frame(stamp))
        sender.tick()

    assert sender.stats.max_frames_in_window <= MAX_FRAMES_PER_WINDOW, sent_times
    assert sender.stats.throttled >= 1, "挤兑时应当有被硬闸门挡掉的拍"
    # 用验收器同款算法复核一遍：任意 1 秒窗口不超过 5 帧
    for index, start in enumerate(sent_times):
        inside = sum(1 for t in sent_times[index:] if t - start < 1.0)
        assert inside <= MAX_FRAMES_PER_WINDOW, f"从 {start}s 起 1 秒内有 {inside} 帧"


# 现场对账的日志行：必须自带「写成功了没有」，否则验收器只能看到组帧时间戳
def test_referee_log_line_carries_write_ok_and_the_unobservable_third_state(tmp_path):
    from tools.run_radio_vision import _log_referee_frame

    sender, _sent, now = collect_sender(rate_hz=4.8)
    rig = Rig()
    now["t"] = 0.0
    sender.update(rig.frame(0.0))
    assert sender.tick() is True

    log = tmp_path / "referee.jsonl"
    with log.open("w", encoding="utf-8") as handle:
        _log_referee_frame(handle, sender, writer=object())   # 有串口
        _log_referee_frame(handle, sender, writer=None)       # 干跑
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["write_ok"] is True
    assert rows[0]["write_mode"] == "serial"
    assert rows[1]["write_mode"] == "dry_run"
    # 第三态「裁判服务器收到」本地观测不到，必须显式写 None 而不是假装有数
    assert all(row["delivered"] is None for row in rows)


def test_referee_log_line_records_the_failure_reason(tmp_path):
    from tools.run_radio_vision import _log_referee_frame

    now = {"t": 0.0}

    def broken(_payload: bytes) -> bool:
        raise OSError("device unplugged")

    sender = RobotPositionSender(broken, rate_hz=4.8, clock=lambda: now["t"])
    rig = Rig()
    sender.update(rig.frame(0.0))
    sender.tick()

    log = tmp_path / "dead.jsonl"
    with log.open("w", encoding="utf-8") as handle:
        _log_referee_frame(handle, sender, writer=object())
    row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert row["write_ok"] is False
    assert "device unplugged" in row["write_error"]


def test_send_failure_carries_the_transport_reason():
    """回调自己知道为什么没写出去时，统计里要有那句话，不能只剩「失败了」。"""
    class Broken:
        last_error = "SerialException: device reports readiness but returned no data"

        def __call__(self, _payload: bytes) -> bool:
            return False

    now = {"t": 0.0}
    broken = Broken()
    sender = RobotPositionSender(broken, rate_hz=4.8, clock=lambda: now["t"])
    rig = Rig()
    sender.update(rig.frame(0.0))
    sender.tick()
    assert sender.stats.frames_sent == 0
    assert "SerialException" in (sender.last_write_error or "")


def test_throttled_tick_reschedules_to_when_the_window_opens():
    """闸门命中后不该白等一个周期：空洞要跟循环卡顿解耦。

    等一个周期时，主循环卡 0.25s + 一个周期 = 0.46s，贴着 0.5s 断流判据；
    改成等「最老那帧满 1 秒」，闸门造成的空洞硬性 ≤ 1.0 - 窗口跨度。
    """
    now = {"t": 0.0}
    sent = []

    def send(_payload: bytes) -> bool:
        sent.append(now["t"])
        return True

    sender = RobotPositionSender(send, rate_hz=4.8, clock=lambda: now["t"])
    # 真实会触发闸门的排程：主循环卡一下 + 紧跟一拍，5 帧挤在 0.85s 里；
    # 到 1.7s 那一拍，窗口里还有 5 帧没满 1 秒，于是被挡掉。
    for stamp in (0.0, 0.05, 0.25, 0.3, 0.35, 0.55, 0.75, 0.8, 0.85,
                  1.05, 1.1, 1.4, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 2.1, 2.4):
        now["t"] = stamp
        sender.tick()
    assert sender.stats.throttled >= 1, "这个排程本该触发闸门"
    gaps = [b - a for a, b in zip(sent, sent[1:])]
    # 闸门造成的空洞硬性 ≤ 1.0 - 窗口跨度，4.8Hz 下约 0.38s
    assert max(gaps) <= 0.40, f"闸门后空洞 {max(gaps):.3f}s 太大"
    for index, start in enumerate(sent):
        inside = sum(1 for t in sent[index:] if t - start < 1.0)
        assert inside <= MAX_FRAMES_PER_WINDOW


def test_short_run_rate_is_flagged_as_provisional():
    """2 秒跑出 5.0+ 的「频率」只是分母太小，自报必须标成仅供粗看。"""
    sender, _sent, now = collect_sender(rate_hz=4.8)
    rig = Rig()
    for step in range(40):          # 2 秒、20Hz 循环
        now["t"] = step * 0.05
        sender.update(rig.frame(now["t"]))
        sender.tick()
    snapshot = sender.stats.as_dict(now=now["t"])
    assert snapshot["rate_window_s"] < RATE_SETTLE_S
    assert snapshot["rate_settled"] is False

    for step in range(40, 700):     # 跑到 35 秒
        now["t"] = step * 0.05
        sender.update(rig.frame(now["t"]))
        sender.tick()
    snapshot = sender.stats.as_dict(now=now["t"])
    assert snapshot["rate_settled"] is True
    assert 4.6 < snapshot["effective_rate_hz"] < 4.9
