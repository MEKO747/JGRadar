"""radio_probe 诊断工具：解码链计数与判定结论。

这个工具的价值全在「结论必须对」上。它曾经差点骗人——2 次同步就被判成
「有帧但解码全失败」，于是被当成解码器坏了。下面把每一级断点的结论都钉死，
包括「偶发同步」这种最容易误报的边界。

合成帧按接收包自己的常量拼：Access Code + AIR_HEADER + 15 字节载荷，
载荷本身是一整条串口帧（SOF + 长度 + 序号 + CRC8 + 命令字 + 数据 + CRC16）。
"""

from __future__ import annotations

import random
from types import SimpleNamespace

from tools.radio_probe import ChannelStats, classify, format_message

from rmuc.coder import CMD_INTERFERENCE_KEY, SOF, crc8, crc16
from rmuc.constants import ACCESS_CODES, AIR_HEADER, AIR_PAYLOAD_LEN, WaveType


def _serial_frame(cmd_id: int, payload: bytes) -> bytes:
    """拼一条通过 CRC8 + CRC16 的串口帧。"""
    head = bytes([SOF]) + len(payload).to_bytes(2, "little") + b"\x00"
    head += bytes([crc8(head)])
    body = cmd_id.to_bytes(2, "little") + payload
    frame = head + body
    return frame + crc16(frame).to_bytes(2, "little")


def _air_bits(wave: WaveType, payload: bytes) -> bytes:
    """把载荷包成空口帧，按 MSB 优先拆成 unpacked bit bytes。"""
    assert len(payload) == AIR_PAYLOAD_LEN
    frame = ACCESS_CODES[wave] + AIR_HEADER + payload
    return bytes((byte >> bit) & 1 for byte in frame for bit in range(7, -1, -1))


def _key_frame(key: bytes = b"ABC123", wave: WaveType = WaveType.INFO) -> bytes:
    """拼一条 15 字节（正好一个空口帧）的干扰波密钥包。"""
    payload = _serial_frame(CMD_INTERFERENCE_KEY, key)
    assert len(payload) == AIR_PAYLOAD_LEN
    return _air_bits(wave, payload)


def test_valid_frame_passes_all_three_stages():
    stats = ChannelStats(WaveType.INFO)
    stats.feed(_key_frame())
    assert (stats.bits, stats.sync, stats.frames, stats.decoded) == (27 * 8, 1, 1, 1)
    assert "正常" in classify(stats, 10.0)


def test_inverted_polarity_still_decodes():
    """空口允许整帧反相，工具计数不能因为极性翻转就归零。"""
    stats = ChannelStats(WaveType.INFO)
    stats.feed(bytes(0 if bit else 1 for bit in _key_frame()))
    assert stats.decoded == 1


def test_broken_crc16_stops_at_serial_stage():
    """同步率正常但 CRC16 全坏时，结论要指向帧层而不是「信号弱」。"""
    bad = bytearray(_key_frame())
    # 只翻转串口帧最后一字节（CRC16 高位）对应的一个 bit。
    bad[-8] ^= 1
    stats = ChannelStats(WaveType.INFO)
    for _ in range(20):
        stats.feed(bytes(bad))
    assert stats.sync == 20
    assert stats.frames == 0
    assert "序列帧全失败" in classify(stats, 1.0)


def test_unknown_command_stops_at_business_stage():
    """CRC 全过但命令字不认识时，结论必须指向业务层而不是帧层。"""
    payload = _serial_frame(0x0AFF, b"\x00" * 6)
    stats = ChannelStats(WaveType.INFO)
    stats.feed(_air_bits(WaveType.INFO, payload))
    assert stats.frames == 1
    assert stats.decoded == 0
    assert "业务层解不出" in classify(stats, 10.0)


def test_pure_noise_reports_no_sync():
    rng = random.Random(20261120)
    noise = bytes(rng.getrandbits(1) for _ in range(20000))
    stats = ChannelStats(WaveType.INFO)
    stats.feed(noise)
    assert stats.sync == 0
    assert "没有空口帧同步" in classify(stats, 10.0)


def test_sparse_sync_is_not_reported_as_decode_failure():
    """回归：偶发同步不能被判成「有同步但解码全失败」。"""
    stats = ChannelStats(WaveType.INTERFERE, bits=250_000, sync=2, frames=0, decoded=0)
    verdict = classify(stats, 12.0)
    assert "偶发同步" in verdict
    assert "序列帧全失败" not in verdict


def test_silent_channel_reports_no_bits():
    verdict = classify(ChannelStats(WaveType.INFO), 10.0)
    assert "没收到任何比特" in verdict


def test_partial_business_failure_is_reported_as_partial():
    stats = ChannelStats(WaveType.INFO, bits=1000, sync=10, frames=10, decoded=7)
    assert "部分失败" in classify(stats, 10.0)


def test_position_message_formatting_marks_invalid_slots():
    message = SimpleNamespace(
        hero_x=2422, hero_y=1483,
        engineer_x=0, engineer_y=0,
        infantry3_x=604, infantry3_y=21,
        infantry4_x=1204, infantry4_y=35,
        aerial_x=2205, aerial_y=825,
        sentry_x=1978, sentry_y=1354,
    )
    text = format_message("0x0A01", message)
    assert "英雄1(24.22m,14.83m)" in text
    assert "工程2(--,--)" in text
