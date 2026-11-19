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


# ==================== 业务层解析 ====================

class BusinessDecoder:
    """解析支持的 RMUC 2026 无线电业务包。"""

    @staticmethod
    def decode(frame: SerialFrame) -> dict | None:
        """按命令字和长度解析有效载荷。"""
        payload = frame.payload
        payload_len = len(payload)

        if frame.cmd_id == CMD_ROBOT_POSITION and payload_len == 24:
            values = struct.unpack("<12h", payload)
            names = ["hero", "eng", "inf3", "inf4", "aerial", "sentry"]
            return {
                "cmd": "0x0A01",
                "positions": {
                    name: {"x": values[index * 2], "y": values[index * 2 + 1]}
                    for index, name in enumerate(names)
                },
            }

        if frame.cmd_id == CMD_ROBOT_HP and payload_len == 12:
            values = struct.unpack("<6H", payload)
            return {
                "cmd": "0x0A02",
                "hp": {
                    "hero": values[0],
                    "eng": values[1],
                    "inf3": values[2],
                    "inf4": values[3],
                    "sentry": values[5],
                },
            }

        if frame.cmd_id == CMD_ROBOT_AMMO and payload_len == 10:
            values = struct.unpack("<5H", payload)
            return {
                "cmd": "0x0A03",
                "ammo": {
                    "hero": values[0],
                    "inf3": values[1],
                    "inf4": values[2],
                    "aerial": values[3],
                    "sentry": values[4],
                },
            }

        if frame.cmd_id == CMD_FIELD_STATUS and payload_len == 8:
            remain_coins, total_coins, status_flags = struct.unpack("<HHI", payload)
            return {
                "cmd": "0x0A04",
                "field": {
                    "remain_coins": remain_coins,
                    "total_coins": total_coins,
                    "status_flags": status_flags,
                },
            }

        if frame.cmd_id == CMD_ROBOT_BUFF and payload_len in (36, 41):
            # 旧版 36 B 不含机器人主要状态；新版 41 B 在末尾追加 5 个状态字节。
            if payload_len == 36:
                values = struct.unpack("<" + ("BHBBH" * 4) + "BHBBHB", payload)
                robot_statuses = [0] * 5
            else:
                values = struct.unpack("<" + ("BHBBH" * 5) + "B" + ("B" * 5), payload)
                robot_statuses = values[26:31]

            names = ["hero", "eng", "inf3", "inf4", "sentry"]
            buffs = {}
            for index, name in enumerate(names):
                base = index * 5
                buffs[name] = {
                    "heal": values[base],
                    "cool": values[base + 1],
                    "def": values[base + 2],
                    "vuln": values[base + 3],
                    "atk": values[base + 4],
                }
            return {
                "cmd": "0x0A05",
                "buffs": buffs,
                "sentry_posture": values[25],
                "robot_statuses": dict(zip(names, robot_statuses)),
            }

        if frame.cmd_id == CMD_INTERFERENCE_KEY and payload_len == 6:
            return {
                "cmd": "0x0A06",
                "key": payload.decode("ascii", errors="replace"),
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
