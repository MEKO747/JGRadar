"""业务包协议版本：2026 / 2027 两套布局。

2027 通信协议相对 2026 是字段级重构，而且**字节布局并不完全一样**：

| 命令 | 2026 | 2027 |
| --- | --- | --- |
| `0x0A01` | 24B，重装/工程/3/4/空中/哨兵 | 24B，工程与空中变保留位 |
| `0x0A02` | 12B，6 个 uint16 | 12B，工程变保留位 |
| `0x0A03` | 10B，含空中 | 10B，空中变保留位 |
| `0x0A04` | **8B** | **12B**，末尾多一个地形跨越状态 uint32 |
| `0x0A05` | **36 / 41B**，5 台 | **47B**，4 台 + 自定义装甲模块 mm 坐标 |

长度是唯一可靠的版本判据，所以这里逐字节钉死每套布局的偏移——偏移错一格
不会报错，只会安静地读错字段。
"""

from __future__ import annotations

import struct

import pytest

from rmuc.coder import (
    BUFF_OFFSETS,
    DEFAULT_PROTOCOL,
    PAYLOAD_LENGTHS,
    PROTOCOL_2026,
    PROTOCOL_2027,
    PROTOCOL_AUTO,
    BusinessDecoder,
    SerialFrame,
    resolve_protocol,
)


def decode(cmd_id: int, payload: bytes, **kwargs):
    return BusinessDecoder.decode(SerialFrame(cmd_id, payload), **kwargs)


# ---------------------------------------------------------------------------
# 版本判定
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("cmd_id", "length", "expected"),
    [
        (0x0A04, 8, PROTOCOL_2026),
        (0x0A04, 12, PROTOCOL_2027),
        (0x0A05, 36, PROTOCOL_2026),
        (0x0A05, 41, PROTOCOL_2026),
        (0x0A05, 47, PROTOCOL_2027),
    ],
)
def test_auto_detection_uses_payload_length(cmd_id: int, length: int, expected: str):
    assert resolve_protocol(cmd_id, length) == expected


def test_same_length_commands_fall_back_to_the_default_season():
    """0x0A01 两代都是 24 字节，长度分不出来，按 DEFAULT_PROTOCOL 解释。"""
    assert resolve_protocol(0x0A01, 24, PROTOCOL_AUTO) == DEFAULT_PROTOCOL
    assert DEFAULT_PROTOCOL == PROTOCOL_2027, "目标赛季是 2027"


def test_ambiguous_lengths_are_read_as_the_target_season_not_as_2026():
    """0x0A01 两代同长，auto 下按 2027 解释：工程和空中作为保留位丢掉。

    这是有意的取舍。2027 的保留位在协议里没定义内容，按 2026 读会把可能是
    垃圾值的两槽当成真实坐标发出去；宁可丢掉这两个槽，回放 2026 录制时
    显式给 ``--protocol 2026``。
    """
    payload = struct.pack("<12h", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    decoded = decode(0x0A01, payload)
    assert decoded["protocol"] == PROTOCOL_2027
    assert set(decoded["positions"]) == {"hero", "inf3", "inf4", "sentry"}


def test_pinned_protocol_rejects_the_other_seasons_length():
    assert resolve_protocol(0x0A04, 12, PROTOCOL_2026) is None
    assert resolve_protocol(0x0A04, 8, PROTOCOL_2027) is None
    assert resolve_protocol(0x0A04, 12, PROTOCOL_2027) == PROTOCOL_2027


def test_unknown_length_is_not_a_candidate():
    assert resolve_protocol(0x0A04, 99) is None
    assert decode(0x0A04, b"\x00" * 99) is None


def test_unknown_protocol_name_fails_loudly():
    with pytest.raises(ValueError, match="未知协议版本"):
        resolve_protocol(0x0A04, 12, "2028")


def test_unknown_command_is_rejected_even_with_a_valid_length():
    assert decode(0x0AFF, b"\x00" * 6) is None


# ---------------------------------------------------------------------------
# 2026 回归：既有行为不许变
# ---------------------------------------------------------------------------
def test_2026_position_has_all_six_slots():
    payload = struct.pack("<12h", *range(12))
    decoded = decode(0x0A01, payload, protocol=PROTOCOL_2026)
    assert decoded["protocol"] == PROTOCOL_2026
    assert set(decoded["positions"]) == {"hero", "eng", "inf3", "inf4", "aerial", "sentry"}
    assert decoded["positions"]["eng"] == {"x": 2, "y": 3}
    assert decoded["positions"]["sentry"] == {"x": 10, "y": 11}


def test_2026_hp_and_ammo_map_the_same_slots_as_before():
    hp = decode(0x0A02, struct.pack("<6H", 1, 2, 3, 4, 5, 6), protocol=PROTOCOL_2026)
    assert hp["hp"] == {"hero": 1, "eng": 2, "inf3": 3, "inf4": 4, "sentry": 6}

    ammo = decode(0x0A03, struct.pack("<5H", 7, 8, 9, 10, 11), protocol=PROTOCOL_2026)
    assert ammo["ammo"] == {"hero": 7, "inf3": 8, "inf4": 9, "aerial": 10, "sentry": 11}


def test_2026_field_status_is_eight_bytes():
    decoded = decode(0x0A04, struct.pack("<HHI", 12, 340, 0x00AB_00CD), protocol=PROTOCOL_2026)
    assert decoded["protocol"] == PROTOCOL_2026
    assert decoded["field"] == {"remain_coins": 12, "total_coins": 340,
                                "status_flags": 0x00AB_00CD}


def test_2026_buff_has_five_robots_at_seven_byte_stride():
    payload = bytearray(41)
    for index, (_name, offset) in enumerate(BUFF_OFFSETS[PROTOCOL_2026]):
        struct.pack_into("<BHBBH", payload, offset, index, 100 + index, 2, 3, 200 + index)
    payload[35] = 4
    for offset, value in zip(range(36, 41), (0, 1, 2, 3, 4)):
        payload[offset] = value

    decoded = decode(0x0A05, bytes(payload), protocol=PROTOCOL_2026)
    assert decoded["protocol"] == PROTOCOL_2026
    assert set(decoded["buffs"]) == {"hero", "eng", "inf3", "inf4", "sentry"}
    assert decoded["buffs"]["eng"] == {"heal": 1, "cool": 101, "def": 2, "vuln": 3, "atk": 201}
    assert decoded["sentry_posture"] == 4
    assert decoded["robot_statuses"] == {"hero": 0, "eng": 1, "inf3": 2, "inf4": 3, "sentry": 4}
    assert "armor" not in decoded


def test_2026_short_buff_variant_has_no_status_bytes():
    payload = bytearray(36)
    payload[35] = 2
    decoded = decode(0x0A05, bytes(payload), protocol=PROTOCOL_2026)
    assert decoded["protocol"] == PROTOCOL_2026
    assert decoded["sentry_posture"] == 2
    assert set(decoded["robot_statuses"].values()) == {0}


# ---------------------------------------------------------------------------
# 2027：字段级重构
# ---------------------------------------------------------------------------
def test_2027_position_drops_engineer_and_aerial():
    payload = struct.pack("<12h", *range(12))
    decoded = decode(0x0A01, payload, protocol=PROTOCOL_2027)
    assert decoded["protocol"] == PROTOCOL_2027
    assert set(decoded["positions"]) == {"hero", "inf3", "inf4", "sentry"}
    assert decoded["positions"]["inf3"] == {"x": 4, "y": 5}
    assert decoded["positions"]["sentry"] == {"x": 10, "y": 11}


def test_2027_hp_and_ammo_drop_the_reserved_slots():
    hp = decode(0x0A02, struct.pack("<6H", 1, 999, 3, 4, 999, 6), protocol=PROTOCOL_2027)
    assert hp["hp"] == {"hero": 1, "inf3": 3, "inf4": 4, "sentry": 6}

    ammo = decode(0x0A03, struct.pack("<5H", 7, 8, 9, 999, 11), protocol=PROTOCOL_2027)
    assert ammo["ammo"] == {"hero": 7, "inf3": 8, "inf4": 9, "sentry": 11}


def test_2027_field_status_carries_terrain_flags():
    decoded = decode(0x0A04, struct.pack("<HHII", 5, 900, 0x0000_0ABC, 0x0000_03FF))
    assert decoded["protocol"] == PROTOCOL_2027
    assert decoded["field"]["remain_coins"] == 5
    assert decoded["field"]["status_flags"] == 0xABC
    assert decoded["field"]["terrain_flags"] == 0x3FF


def test_2027_buff_layout_is_byte_exact():
    """逐个偏移写值再读回来：47 字节布局错一格就会安静地读错字段。"""
    payload = bytearray(47)
    # 重装 0、3 号步兵 14、4 号步兵 21、哨兵 28，每块 <BHBBH>
    for index, (_name, offset) in enumerate(BUFF_OFFSETS[PROTOCOL_2027]):
        struct.pack_into("<BHBBH", payload, offset, 10 + index, 100 + index,
                         20 + index, 30 + index, 200 + index)
    payload[35] = 3          # 哨兵姿态：移动姿态
    payload[36] = 1          # 重装状态：战亡
    payload[37] = 0xEE       # 保留位，必须被忽略
    payload[38] = 2          # 3 号步兵：无敌但不虚弱
    payload[39] = 3          # 4 号步兵
    payload[40] = 0          # 哨兵：存活
    struct.pack_into("<HHH", payload, 41, 1234, 5678, 90)   # 加固区域 x/y/半径 mm

    decoded = decode(0x0A05, bytes(payload))
    assert decoded["protocol"] == PROTOCOL_2027
    assert set(decoded["buffs"]) == {"hero", "inf3", "inf4", "sentry"}
    assert decoded["buffs"]["hero"] == {"heal": 10, "cool": 100, "def": 20, "vuln": 30, "atk": 200}
    assert decoded["buffs"]["inf3"] == {"heal": 11, "cool": 101, "def": 21, "vuln": 31, "atk": 201}
    assert decoded["buffs"]["sentry"] == {"heal": 13, "cool": 103, "def": 23, "vuln": 33, "atk": 203}
    assert decoded["sentry_posture"] == 3
    assert decoded["robot_statuses"] == {"hero": 1, "inf3": 2, "inf4": 3, "sentry": 0}
    assert decoded["armor"] == {"x_mm": 1234, "y_mm": 5678, "radius_mm": 90}


def test_2027_armor_all_zero_means_no_intel():
    payload = bytearray(47)
    decoded = decode(0x0A05, bytes(payload))
    assert decoded["armor"] == {"x_mm": 0, "y_mm": 0, "radius_mm": 0}


def test_interference_key_is_version_agnostic():
    decoded = decode(0x0A06, b"ABC123")
    assert decoded["key"] == "ABC123"
    assert decoded["protocol"] in (PROTOCOL_2026, PROTOCOL_2027)


# ---------------------------------------------------------------------------
# 长度表要自洽
# ---------------------------------------------------------------------------
def test_every_command_has_a_length_table_for_both_seasons():
    for version in (PROTOCOL_2026, PROTOCOL_2027):
        for cmd_id in (0x0A01, 0x0A02, 0x0A03, 0x0A04, 0x0A05, 0x0A06):
            assert cmd_id in PAYLOAD_LENGTHS[version]
            assert PAYLOAD_LENGTHS[version][cmd_id], f"{version} 的 {cmd_id:#06x} 没有合法长度"


def test_slot_tables_have_the_right_width():
    from rmuc.coder import AMMO_SLOTS, HP_SLOTS, POSITION_SLOTS

    for version in (PROTOCOL_2026, PROTOCOL_2027):
        assert len(POSITION_SLOTS[version]) == 6
        assert len(HP_SLOTS[version]) == 6
        assert len(AMMO_SLOTS[version]) == 5
    assert POSITION_SLOTS[PROTOCOL_2026].count(None) == 0
    assert POSITION_SLOTS[PROTOCOL_2027].count(None) == 2
