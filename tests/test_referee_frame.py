"""0x0305 的线格式：0xA5 帧头 + CRC8 + CRC16 封装。

发射方向的帧必须能被接收方向同一套解码器认回来——这是唯一能离线验证
「CRC 算对了没有」的办法，也顺手证明了收发用的是同一份 CRC 表。
"""

from __future__ import annotations

import pytest

from rmuc.coder import SOF, SerialFrameDecoder
from rmuc.referee_frame import (
    CMD_RADAR_POSITION,
    RadarPositionWriter,
    build_frame,
    build_radar_frame,
)

PAYLOAD = bytes(range(48))


def test_radar_frame_round_trips_through_the_receiver_decoder():
    frame = build_radar_frame(PAYLOAD, seq=7)
    frames = SerialFrameDecoder().feed(frame)
    assert len(frames) == 1, "帧头/CRC8/CRC16 任何一处不对，这里都会是 0 条"
    assert frames[0].cmd_id == CMD_RADAR_POSITION
    assert frames[0].payload == PAYLOAD


def test_frame_header_layout():
    frame = build_radar_frame(PAYLOAD, seq=3)
    assert frame[0] == SOF
    assert int.from_bytes(frame[1:3], "little") == len(PAYLOAD)
    assert frame[3] == 3
    assert int.from_bytes(frame[5:7], "little") == CMD_RADAR_POSITION
    assert len(frame) == 5 + 2 + len(PAYLOAD) + 2


def test_sequence_wraps_in_a_byte():
    assert build_radar_frame(PAYLOAD, seq=256)[3] == 0
    assert build_radar_frame(PAYLOAD, seq=257)[3] == 1


def test_wrong_payload_length_is_rejected():
    with pytest.raises(ValueError, match="48 字节"):
        build_radar_frame(b"\x00" * 47)


def test_cmd_id_must_fit_uint16():
    with pytest.raises(ValueError, match="cmd_id"):
        build_frame(0x10000, b"")


def test_writer_numbers_frames_and_counts_them():
    written: list[bytes] = []
    writer = RadarPositionWriter(write=written.append, seq_start=250)
    for _ in range(4):
        writer.write(PAYLOAD)
    assert writer.frames_written == 4
    assert [frame[3] for frame in written] == [250, 251, 252, 253]
    assert all(SerialFrameDecoder().feed(frame) for frame in written)


def test_writer_stops_after_a_transport_error():
    """串口拔掉之后不能再每拍都抛异常——记一次失败就停，让统计里看得见。"""
    calls = {"count": 0}

    def boom(_frame: bytes) -> None:
        calls["count"] += 1
        raise OSError("device unplugged")

    messages: list[str] = []
    writer = RadarPositionWriter(write=boom, logger=messages.append)
    assert writer.write(PAYLOAD) is None
    assert writer.failed is True
    assert "device unplugged" in (writer.last_error or "")
    assert messages, "失败要能报到日志，不能静默"

    assert writer.write(PAYLOAD) is None
    assert calls["count"] == 1, "失败之后不该再反复碰设备"
    assert writer.frames_written == 0


def test_writer_is_callable_as_the_sender_callback():
    written: list[bytes] = []
    writer = RadarPositionWriter(write=written.append)
    assert writer(PAYLOAD) is True, "写成功必须回报 True，否则发送端会把成功记成失败"
    assert len(written) == 1


def test_writer_callback_reports_failure_instead_of_silently_swallowing_it():
    """故障注入：串口写失败时回调必须返回 False，发送端据此不计 frames_sent。"""
    def boom(_frame: bytes) -> None:
        raise OSError("device unplugged")

    writer = RadarPositionWriter(write=boom)
    assert writer(PAYLOAD) is False
    assert writer.frames_written == 0
    assert writer.failed is True
