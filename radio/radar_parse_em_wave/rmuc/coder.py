# -*- coding: utf-8 -*-
"""RMUC 2026 空口帧、串口帧和业务包解码。"""

from __future__ import annotations

from dataclasses import dataclass
import struct

from .constants import (
    ACCESS_CODES,
    AIR_FRAME_LEN,
    AIR_HEADER,
    AIR_PAYLOAD_LEN,
    CMD_FIELD_STATUS,
    CMD_INTERFERENCE_KEY,
    CMD_ROBOT_AMMO,
    CMD_ROBOT_BUFF,
    CMD_ROBOT_HP,
    CMD_ROBOT_POSITION,
    CRC8_INIT,
    CRC8_TABLE,
    CRC16_INIT,
    CRC16_TABLE,
    WaveType,
)


# ==================== CRC ====================

def crc8(data: bytes, init: int = CRC8_INIT) -> int:
    """计算 DJI/RMUC 风格帧头 CRC8。

    覆盖范围：SOF + data_length + seq，即串口帧头前 4 字节。
    """
    crc = init & 0xFF
    for byte in data:
        crc = CRC8_TABLE[(crc ^ byte) & 0xFF]
    return crc


def crc16(data: bytes, init: int = CRC16_INIT) -> int:
    """计算 DJI/RMUC 风格整帧 CRC16。

    覆盖范围：从 SOF 到 payload 结束的全部内容，不包含帧尾 CRC16 自身。
    """
    crc = init & 0xFFFF
    for byte in data:
        crc = ((crc >> 8) ^ CRC16_TABLE[(crc ^ byte) & 0xFF]) & 0xFFFF
    return crc


# ==================== 串口帧数据模型 ====================

SOF = 0xA5
"""RMUC 串口协议帧 SOF 字节。"""


@dataclass(frozen=True)
class SerialFrame:
    """已经通过 CRC 校验的串口协议帧。"""

    cmd_id: int
    payload: bytes


class SerialFrameDecoder:
    """跨空口 payload 恢复连续串口帧，并校验 CRC8 和 CRC16。"""

    def __init__(self, max_payload_len: int = 200):
        self.buffer = bytearray()
        self.max_payload_len = max_payload_len

    def feed(self, data: bytes) -> list[SerialFrame]:
        self.buffer.extend(data)
        frames = []
        while len(self.buffer) >= 5:
            if self.buffer[0] != SOF or crc8(self.buffer[:4]) != self.buffer[4]:
                del self.buffer[0]
                continue

            data_len = struct.unpack_from("<H", self.buffer, 1)[0]
            if data_len > self.max_payload_len:
                del self.buffer[0]
                continue

            frame_len = 5 + 2 + data_len + 2
            if len(self.buffer) < frame_len:
                break

            expected_crc16 = struct.unpack_from("<H", self.buffer, frame_len - 2)[0]
            if crc16(self.buffer[:frame_len - 2]) != expected_crc16:
                del self.buffer[0]
                continue

            cmd_id = struct.unpack_from("<H", self.buffer, 5)[0]
            frames.append(SerialFrame(cmd_id, bytes(self.buffer[7:frame_len - 2])))
            del self.buffer[:frame_len]
        return frames


# ==================== 协议版本 ====================

PROTOCOL_2026 = "2026"
PROTOCOL_2027 = "2027"
PROTOCOL_AUTO = "auto"

SUPPORTED_PROTOCOLS = (PROTOCOL_2026, PROTOCOL_2027)

#: 自动识别时，同一长度在两代都合法就按这个赛季解释。
#: 2027 是目标赛季（规则与通信协议都以 2027 为准），2026 只用于回放旧录制。
DEFAULT_PROTOCOL = PROTOCOL_2027

#: 各赛季每个业务命令的合法载荷长度。
#:
#: 长度是唯一可靠的版本仲裁依据：0x0A04 在 2026 是 8 字节、2027 是 12 字节，
#: 0x0A05 在 2026 是 36/41 字节、2027 是 47 字节，两代不重叠。0x0A01/0x0A02/
#: 0x0A03 长度相同，只能按 ``DEFAULT_PROTOCOL`` 解释——它们两代的字节布局
#: 一样，差别只在哪些槽位是保留位。
PAYLOAD_LENGTHS: dict[str, dict[int, tuple[int, ...]]] = {
    PROTOCOL_2026: {
        CMD_ROBOT_POSITION: (24,),
        CMD_ROBOT_HP: (12,),
        CMD_ROBOT_AMMO: (10,),
        CMD_FIELD_STATUS: (8,),
        CMD_ROBOT_BUFF: (36, 41),
        CMD_INTERFERENCE_KEY: (6,),
    },
    PROTOCOL_2027: {
        CMD_ROBOT_POSITION: (24,),
        CMD_ROBOT_HP: (12,),
        CMD_ROBOT_AMMO: (10,),
        CMD_FIELD_STATUS: (12,),
        CMD_ROBOT_BUFF: (47,),
        CMD_INTERFERENCE_KEY: (6,),
    },
}

#: 位置包 12 个 int16 的槽位顺序（每槽 x、y 两个）。``None`` = 保留位。
#: 2027 只下发重装、3 号步兵、4 号步兵、哨兵，工程和空中是保留位。
POSITION_SLOTS: dict[str, tuple[str | None, ...]] = {
    PROTOCOL_2026: ("hero", "eng", "inf3", "inf4", "aerial", "sentry"),
    PROTOCOL_2027: ("hero", None, "inf3", "inf4", None, "sentry"),
}

#: 血量包 6 个 uint16 的槽位顺序。2026 的 2 号槽是工程、4 号槽未命名。
HP_SLOTS: dict[str, tuple[str | None, ...]] = {
    PROTOCOL_2026: ("hero", "eng", "inf3", "inf4", None, "sentry"),
    PROTOCOL_2027: ("hero", None, "inf3", "inf4", None, "sentry"),
}

#: 弹量包 5 个 uint16 的槽位顺序。
AMMO_SLOTS: dict[str, tuple[str | None, ...]] = {
    PROTOCOL_2026: ("hero", "inf3", "inf4", "aerial", "sentry"),
    PROTOCOL_2027: ("hero", "inf3", "inf4", None, "sentry"),
}

#: 0x0A05 每个机器人增益块的起始偏移，块结构固定为 ``<BHBBH``：
#: 回血(1B) 冷却(2B) 防御(1B) 负防御(1B) 攻击(2B)，共 7 字节。
#: 2027 在 7~13 留了保留位，所以重装之后直接跳到 3 号步兵。
BUFF_OFFSETS: dict[str, tuple[tuple[str, int], ...]] = {
    PROTOCOL_2026: (("hero", 0), ("eng", 7), ("inf3", 14), ("inf4", 21), ("sentry", 28)),
    PROTOCOL_2027: (("hero", 0), ("inf3", 14), ("inf4", 21), ("sentry", 28)),
}

#: 0x0A05 主要状态字节的偏移。2027 的 37 号字节是保留位。
STATUS_OFFSETS: dict[str, tuple[tuple[str, int], ...]] = {
    PROTOCOL_2026: (("hero", 36), ("eng", 37), ("inf3", 38), ("inf4", 39), ("sentry", 40)),
    PROTOCOL_2027: (("hero", 36), ("inf3", 38), ("inf4", 39), ("sentry", 40)),
}

#: 2027 新增：自定义装甲模块加固区域圆心 x、y 和半径，单位 mm（0 = 暂无情报）。
ARMOR_OFFSET = (41, 43, 45)

#: 0x0A05 里哨兵姿态的字节偏移，两代相同。
POSTURE_OFFSET = 35

#: 2026 的 36 字节版本没有主要状态字节。
_MIN_STATUS_LENGTH = 41


def resolve_protocol(
    cmd_id: int, payload_len: int, requested: str = PROTOCOL_AUTO
) -> str | None:
    """判断这份载荷属于哪个赛季；长度不认识就返回 ``None``。

    ``requested`` 指定了赛季时长度必须与它相符——把 2026 的 8 字节 0x0A04
    当 2027 读会读到别的字段，宁可判不合法也不要猜。
    """
    if requested != PROTOCOL_AUTO:
        if requested not in PAYLOAD_LENGTHS:
            raise ValueError(
                f"未知协议版本 {requested!r}；可用 = {list(SUPPORTED_PROTOCOLS)} 或 {PROTOCOL_AUTO!r}"
            )
        return requested if payload_len in PAYLOAD_LENGTHS[requested].get(cmd_id, ()) else None

    matches = [
        version
        for version, table in PAYLOAD_LENGTHS.items()
        if payload_len in table.get(cmd_id, ())
    ]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    return DEFAULT_PROTOCOL


# ==================== 业务层解析 ====================


def _decode_position(payload: bytes, version: str) -> dict:
    values = struct.unpack("<12h", payload)
    positions = {
        name: {"x": values[index * 2], "y": values[index * 2 + 1]}
        for index, name in enumerate(POSITION_SLOTS[version])
        if name
    }
    return {"cmd": "0x0A01", "protocol": version, "positions": positions}


def _decode_hp(payload: bytes, version: str) -> dict:
    values = struct.unpack("<6H", payload)
    hp = {
        name: values[index]
        for index, name in enumerate(HP_SLOTS[version])
        if name
    }
    return {"cmd": "0x0A02", "protocol": version, "hp": hp}


def _decode_ammo(payload: bytes, version: str) -> dict:
    values = struct.unpack("<5H", payload)
    ammo = {
        name: values[index]
        for index, name in enumerate(AMMO_SLOTS[version])
        if name
    }
    return {"cmd": "0x0A03", "protocol": version, "ammo": ammo}


def _decode_field(payload: bytes, version: str) -> dict:
    if version == PROTOCOL_2027:
        remain, total, status, terrain = struct.unpack("<HHII", payload)
        field = {
            "remain_coins": remain,
            "total_coins": total,
            "status_flags": status,
            "terrain_flags": terrain,
        }
    else:
        remain, total, status = struct.unpack("<HHI", payload)
        field = {"remain_coins": remain, "total_coins": total, "status_flags": status}
    return {"cmd": "0x0A04", "protocol": version, "field": field}


def _decode_buff(payload: bytes, version: str) -> dict:
    buffs = {}
    for name, offset in BUFF_OFFSETS[version]:
        heal, cool, defense, vuln, attack = struct.unpack_from("<BHBBH", payload, offset)
        buffs[name] = {
            "heal": heal,
            "cool": cool,
            "def": defense,
            "vuln": vuln,
            "atk": attack,
        }

    if len(payload) >= _MIN_STATUS_LENGTH:
        statuses = {name: payload[offset] for name, offset in STATUS_OFFSETS[version]}
    else:
        # 2026 的 36 字节旧版没有状态字节，补零保持下游形状一致。
        statuses = {"hero": 0, "eng": 0, "inf3": 0, "inf4": 0, "sentry": 0}

    decoded = {
        "cmd": "0x0A05",
        "protocol": version,
        "buffs": buffs,
        "sentry_posture": payload[POSTURE_OFFSET],
        "robot_statuses": statuses,
    }
    if version == PROTOCOL_2027:
        x, y, radius = struct.unpack_from("<HHH", payload, ARMOR_OFFSET[0])
        decoded["armor"] = {"x_mm": x, "y_mm": y, "radius_mm": radius}
    return decoded


_HANDLERS = {
    CMD_ROBOT_POSITION: _decode_position,
    CMD_ROBOT_HP: _decode_hp,
    CMD_ROBOT_AMMO: _decode_ammo,
    CMD_FIELD_STATUS: _decode_field,
    CMD_ROBOT_BUFF: _decode_buff,
}


class BusinessDecoder:
    """解析 RMUC 业务包，支持 2026 / 2027 两套协议。

    两代协议的字节布局并不完全一致：``0x0A04`` 从 8 字节变 12 字节（末尾多了
    一个地形跨越状态 uint32），``0x0A05`` 从 41 字节变 47 字节（工程机器人变成
    保留位、末尾多了自定义装甲模块加固区域的圆心与半径）。``0x0A01``/
    ``0x0A02``/``0x0A03`` 长度没变，只是保留位换了语义。

    版本用 ``protocol`` 参数控制：``"auto"`` 按载荷长度自动识别（两代长度
    不重叠的命令可靠），长度相同的命令按 ``DEFAULT_PROTOCOL`` 解释；
    ``"2026"`` / ``"2027"`` 则要求长度必须与该赛季相符，不符判不合法。
    """

    @staticmethod
    def decode(frame: SerialFrame, *, protocol: str = PROTOCOL_AUTO) -> dict | None:
        """按命令字和长度解析有效载荷。"""
        version = resolve_protocol(frame.cmd_id, len(frame.payload), protocol)
        if version is None:
            return None

        handler = _HANDLERS.get(frame.cmd_id)
        if handler is not None:
            return handler(frame.payload, version)

        if frame.cmd_id == CMD_INTERFERENCE_KEY:
            return {
                "cmd": "0x0A06",
                "protocol": version,
                "key": frame.payload.decode("ascii", errors="replace"),
            }
        return None


class AirFrameDecoder:
    """空口帧解码器。

    输入格式为 GNU Radio 解调输出的 unpacked bytes：每个 byte 表示一个判决 bit。

    解码规则：
    - 在 bit 流中滑窗寻找 Access Code；
    - Access Code 允许 max_access_errors 个 bit 误码；
    - 支持整帧极性反相检测与校正；
    - Access Code 匹配后，Header 必须与 AIR_HEADER 完全一致；
    - Header 不一致时只丢弃 1 bit 继续滑窗，避免错误同步导致整帧跳过。
    """

    def __init__(self, wave_type: WaveType, max_access_errors: int = 5):
        self.wave_type = wave_type
        self.access_code = ACCESS_CODES[wave_type]
        self.max_access_errors = int(max_access_errors)

        self.bit_buffer: list[int] = []
        self.access_bits = self._bytes_to_msb_bits(self.access_code)
        self.inverted_access_bits = [0 if bit else 1 for bit in self.access_bits]
        self.frame_bits_len = AIR_FRAME_LEN * 8

    # ==================== 对外接口 ====================

    def feed_unpacked_bytes(self, unpacked_bytes: bytes) -> list[bytes]:
        """喂入 unpacked bytes，返回恢复出的空口帧 payload。"""
        if unpacked_bytes:
            self.bit_buffer.extend(byte & 0x01 for byte in unpacked_bytes)

        payloads: list[bytes] = []

        while len(self.bit_buffer) >= self.frame_bits_len:
            match = self._find_access_match()
            if match is None:
                # 当前缓存中不存在一个足够长、可组成完整帧的同步位置。
                # 保留最后 frame_bits_len - 1 bit，防止 Access Code 已经出现但完整帧尚未到齐。
                keep_bits = self.frame_bits_len - 1
                if len(self.bit_buffer) > keep_bits:
                    del self.bit_buffer[:-keep_bits]
                break

            start_index, inverted, _ = match
            if start_index:
                del self.bit_buffer[:start_index]

            if len(self.bit_buffer) < self.frame_bits_len:
                break

            frame_bits = self.bit_buffer[: self.frame_bits_len]
            if inverted:
                frame_bits = [0 if bit else 1 for bit in frame_bits]

            frame = self._bits_to_bytes_msb(frame_bits)

            # Access Code 可以容错，但 Header 必须逐字节完全一致。
            if frame[8:12] != AIR_HEADER:
                del self.bit_buffer[0]
                continue

            payloads.append(frame[12:12 + AIR_PAYLOAD_LEN])
            del self.bit_buffer[: self.frame_bits_len]

        return payloads

    # ==================== 内部工具 ====================

    def _find_access_match(self) -> tuple[int, bool, int] | None:
        """寻找 Access Code，允许少量误码并支持整帧反相。"""
        search_limit = len(self.bit_buffer) - self.frame_bits_len + 1
        if search_limit <= 0:
            return None

        best_match: tuple[int, bool, int] | None = None

        for index in range(search_limit):
            window = self.bit_buffer[index : index + len(self.access_bits)]

            normal_errors = self._hamming_distance(window, self.access_bits)
            if normal_errors <= self.max_access_errors:
                candidate = (index, False, normal_errors)
                best_match = self._select_better_match(best_match, candidate)
                if normal_errors == 0:
                    return candidate

            inverted_errors = self._hamming_distance(window, self.inverted_access_bits)
            if inverted_errors <= self.max_access_errors:
                candidate = (index, True, inverted_errors)
                best_match = self._select_better_match(best_match, candidate)
                if inverted_errors == 0:
                    return candidate

        return best_match

    @staticmethod
    def _select_better_match(
        current: tuple[int, bool, int] | None,
        candidate: tuple[int, bool, int],
    ) -> tuple[int, bool, int]:
        """在多个 Access Code 候选中选择更可靠者。"""
        if current is None:
            return candidate

        current_index, _current_inverted, current_errors = current
        candidate_index, _candidate_inverted, candidate_errors = candidate

        if candidate_errors < current_errors:
            return candidate
        if candidate_errors == current_errors and candidate_index < current_index:
            return candidate
        return current

    @staticmethod
    def _hamming_distance(left: list[int], right: list[int]) -> int:
        """计算两个等长 bit 序列的汉明距离。"""
        return sum(1 for left_bit, right_bit in zip(left, right) if left_bit != right_bit)

    @staticmethod
    def _bytes_to_msb_bits(data: bytes) -> list[int]:
        """将 packed bytes 按 bit7 -> bit0 顺序展开为 bit 列表。"""
        return [
            (byte >> bit_index) & 0x01
            for byte in data
            for bit_index in range(7, -1, -1)
        ]

    @staticmethod
    def _bits_to_bytes_msb(bits: list[int]) -> bytes:
        """将 MSB-first bit 列表重新合成为 packed bytes。"""
        if len(bits) % 8 != 0:
            raise ValueError("bit length must be a multiple of 8")

        output = bytearray()
        for offset in range(0, len(bits), 8):
            value = 0
            for bit in bits[offset : offset + 8]:
                value = (value << 1) | (1 if bit else 0)
            output.append(value)
        return bytes(output)
