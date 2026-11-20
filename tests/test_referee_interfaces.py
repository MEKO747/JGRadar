"""2027 雷达侧下行接口：0x020C（标记进度）与 0x020E（自主决策同步）。

字段全部按协议原文的位定义钉死——这类位图错一位不会报错，只会安静地
把「对方重装易伤」读成「己方 3 号被标记」，而论文里的 P(t) 真值就来自它。

0x020C 的判定门槛**不对称**：对方是标记进度 ≥100 置 1，己方是 ≥50。
0x020E 的 byte0 bit3-4 就是现有文档里 `(radar_info >> 3) & 0x03` 取的
己方加密等级，2027 仍然成立。
"""

from __future__ import annotations

import pytest

from rmuc.coder import SerialFrameDecoder
from rmuc.referee import (
    CMD_MARKING_PROGRESS,
    CMD_RADAR_INFO,
    MARKING_RESERVED_BITS,
    decode_marking,
    decode_radar_info,
)
from rmuc.referee_frame import build_frame
from rmuc.referee_worker import RefereeWorker


# ---------------------------------------------------------------------------
# 0x020C
# ---------------------------------------------------------------------------
def test_marking_bits_match_the_protocol_table():
    """bit0/2/3/5 是对方易伤，bit6/8/9/11 是己方特殊标识。"""
    payload = (1 << 0) | (1 << 2) | (1 << 3) | (1 << 5) | (1 << 6) | (1 << 8) | (1 << 9) | (1 << 11)
    decoded = decode_marking(payload.to_bytes(2, "little"), own_team="R")
    assert decoded.enemy_vulnerable == ("B1", "B3", "B4", "B7")
    assert decoded.own_marked == ("R1", "R3", "R4", "R7")
    assert decoded.any_marked is True


def test_marking_is_empty_when_all_bits_are_zero():
    decoded = decode_marking(b"\x00\x00")
    assert decoded.enemy_vulnerable == ()
    assert decoded.own_marked == ()
    assert decoded.any_marked is False


def test_marking_reserved_bits_are_ignored():
    payload = sum(1 << bit for bit in MARKING_RESERVED_BITS).to_bytes(2, "little")
    decoded = decode_marking(payload)
    assert decoded.enemy_vulnerable == ()
    assert decoded.own_marked == ()


def test_marking_flips_with_own_team():
    payload = ((1 << 0) | (1 << 6)).to_bytes(2, "little")   # 对方重装 + 己方重装
    red = decode_marking(payload, own_team="R")
    blue = decode_marking(payload, own_team="blue")
    assert red.enemy_vulnerable == ("B1",) and red.own_marked == ("R1",)
    assert blue.enemy_vulnerable == ("R1",) and blue.own_marked == ("B1",)


def test_marking_rejects_wrong_length():
    with pytest.raises(ValueError, match="2 字节"):
        decode_marking(b"\x00")


# ---------------------------------------------------------------------------
# 0x020E
# ---------------------------------------------------------------------------
def _radar_info_bytes(**fields) -> bytes:
    first = second = 0
    first |= int(fields.get("chances", 0)) & 0x03
    first |= (1 if fields.get("enemy_dual", False) else 0) << 2
    first |= (int(fields.get("level", 1)) & 0x03) << 3
    first |= (1 if fields.get("key_modifiable", False) else 0) << 5
    second |= 1 if fields.get("armor_active", False) else 0
    second |= (int(fields.get("armor_chances", 0)) & 0x07) << 1
    import struct
    return bytes([first, second]) + struct.pack(
        "<HHH",
        int(fields.get("x", 0)),
        int(fields.get("y", 0)),
        int(fields.get("radius", 0)),
    )


def test_radar_info_decodes_every_field():
    payload = _radar_info_bytes(chances=2, enemy_dual=True, level=3, key_modifiable=True,
                                armor_active=True, armor_chances=5, x=1234, y=5678, radius=900)
    info = decode_radar_info(payload)
    assert info.dual_vulnerability_chances == 2
    assert info.enemy_dual_vulnerable is True
    assert info.encryption_level == 3
    assert info.key_modifiable is True
    assert info.armor_active is True
    assert info.armor_chances_left == 5
    assert (info.armor_center_x_mm, info.armor_center_y_mm, info.armor_radius_mm) == (1234, 5678, 900)


def test_encryption_level_uses_the_same_two_bits_as_before():
    """现有文档那句 `(radar_info >> 3) & 0x03` 在 2027 依然成立。"""
    for level in (1, 2, 3):
        raw_first = _radar_info_bytes(level=level)[0]
        assert (raw_first >> 3) & 0x03 == level


def test_radar_info_without_armor_region():
    info = decode_radar_info(_radar_info_bytes(level=1))
    assert info.armor_active is False
    assert info.armor_chances_left == 0
    assert info.armor_radius_mm == 0


def test_radar_info_rejects_wrong_length():
    with pytest.raises(ValueError, match="8 字节"):
        decode_radar_info(b"\x00" * 7)


# ---------------------------------------------------------------------------
# 串口读取线程的分派
# ---------------------------------------------------------------------------
def _frame(cmd_id: int, payload: bytes, seq: int = 0) -> bytes:
    return build_frame(cmd_id, payload, seq=seq)


def test_worker_dispatches_both_commands_and_ignores_others():
    markings, infos = [], []
    worker = RefereeWorker(
        read=lambda: b"",
        on_marking=markings.append,
        on_radar_info=infos.append,
    )
    stream = (
        _frame(CMD_RADAR_INFO, _radar_info_bytes(level=2, chances=1))
        + _frame(0x0001, b"\x00" * 10)          # 和我们无关的比赛状态包
        + _frame(CMD_MARKING_PROGRESS, (1 << 2).to_bytes(2, "little"))
    )
    worker.feed(stream)
    assert len(infos) == 1 and infos[0].encryption_level == 2
    assert len(markings) == 1 and markings[0].enemy_vulnerable == ("B3",)
    assert (worker.marking_frames, worker.radar_info_frames) == (1, 1)


def test_worker_survives_a_frame_split_across_reads():
    """串口一次读到的字节数是随机的，帧会被切开——拆帧器必须能等。"""
    markings = []
    worker = RefereeWorker(read=lambda: b"", on_marking=markings.append)
    frame = _frame(CMD_MARKING_PROGRESS, (1 << 0).to_bytes(2, "little"))
    for index in range(0, len(frame), 3):
        worker.feed(frame[index:index + 3])
    assert len(markings) == 1
    assert markings[0].enemy_vulnerable == ("B1",)


def test_worker_counts_and_warns_on_bad_payload_length():
    warnings = []
    worker = RefereeWorker(read=lambda: b"", warn=warnings.append)
    worker.feed(_frame(CMD_RADAR_INFO, b"\x00" * 3))   # 长度不符但 CRC 正确
    assert worker.rejected_frames == 1
    assert worker.radar_info_frames == 0
    assert warnings, "解不出来的帧要报到日志"


def test_worker_uses_the_current_own_team():
    markings = []
    team = {"side": "R"}
    worker = RefereeWorker(
        read=lambda: b"",
        get_own_team=lambda: team["side"],
        on_marking=markings.append,
    )
    payload = ((1 << 0) | (1 << 6)).to_bytes(2, "little")
    worker.feed(_frame(CMD_MARKING_PROGRESS, payload))
    team["side"] = "B"
    worker.feed(_frame(CMD_MARKING_PROGRESS, payload))
    assert markings[0].enemy_vulnerable == ("B1",)
    assert markings[1].enemy_vulnerable == ("R1",)


def test_worker_round_trips_through_the_receiver_frame_decoder():
    """发送方向的封装器（referee_frame）和读取方向用的是同一套帧格式。"""
    payload = _radar_info_bytes(level=3)
    frames = SerialFrameDecoder().feed(_frame(CMD_RADAR_INFO, payload, seq=9))
    assert len(frames) == 1 and frames[0].cmd_id == CMD_RADAR_INFO
