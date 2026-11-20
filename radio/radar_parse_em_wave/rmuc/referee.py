# -*- coding: utf-8 -*-
"""2027 雷达侧下行接口：``0x020C``（标记进度）与 ``0x020E``（自主决策同步）。

两条命令都是**服务器发给雷达机器人**的（常规链路，1Hz），字段按
《2027 高校系列赛通信协议 V1.0.0》表格逐位抄录：

``0x020C``（2 字节位图）——对方机器人的「易伤情况」与己方机器人的
「特殊标识」。判定门槛**不对称**：对方是标记进度 ≥100 置 1，己方是 ≥50。
它是裁判系统回给我们的真值，论文里的 P(t) 曲线可以拿它交叉验证，
不用只靠自己推的模型。

``0x020E``（8 字节）——雷达自主决策用到的服务器状态：双倍易伤机会数、
对方是否正在双倍易伤、**己方加密等级**（即对方干扰波难度等级）、现在能不能
改密钥，以及装甲加固区域的生效状态、剩余次数和圆心半径（mm）。

``0x02xx`` 的长度和无线链路的 ``0x0Axx`` 完全不同，解码器按命令字分派，
不会互相干扰。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

#: 标记进度（易伤/特殊标识）位图，服务器 → 雷达。
CMD_MARKING_PROGRESS = 0x020C

#: 雷达自主决策信息同步，服务器 → 雷达。
CMD_RADAR_INFO = 0x020E

MARKING_LEN = 2
RADAR_INFO_LEN = 8

#: 0x020C 里每一位对应的槽位名。值是被置 1 时该机器人的状态类型。
#: 对方是「易伤」（标记进度 ≥100），己方是「特殊标识」（≥50）。
MARKING_BITS: tuple[tuple[int, str, str], ...] = (
    (0, "B1", "enemy_vulnerable"),
    (2, "B3", "enemy_vulnerable"),
    (3, "B4", "enemy_vulnerable"),
    (5, "B7", "enemy_vulnerable"),
    (6, "R1", "own_marked"),
    (8, "R3", "own_marked"),
    (9, "R4", "own_marked"),
    (11, "R7", "own_marked"),
)

#: 协议里保留的位。解码时按位保留原样，测试用它证明没有串位。
MARKING_RESERVED_BITS: tuple[int, ...] = (1, 4, 7, 10, 12, 13, 14, 15)


@dataclass(frozen=True)
class MarkingProgress:
    """``0x020C`` 解出来的标记状态。

    以红方视角为例：``B1`` 等是对方（蓝方）机器人，``R1`` 等是己方机器人。
    阵营互换时按 ``own_team`` 解释，见 :func:`decode_marking`。
    """

    raw: int
    #: 对方机器人里正在易伤（标记进度 ≥100）的槽位。
    enemy_vulnerable: tuple[str, ...]
    #: 己方机器人里带特殊标识（标记进度 ≥50）的槽位。
    own_marked: tuple[str, ...]

    @property
    def any_marked(self) -> bool:
        return bool(self.enemy_vulnerable or self.own_marked)

    def as_dict(self) -> dict:
        return {
            "raw": self.raw,
            "enemy_vulnerable": list(self.enemy_vulnerable),
            "own_marked": list(self.own_marked),
        }


@dataclass(frozen=True)
class RadarInfo:
    """``0x020E`` 解出来的自主决策状态。"""

    raw: bytes
    #: 雷达还能触发几次双倍易伤（0~2，开局 0）。
    dual_vulnerability_chances: int
    #: 对方此刻是否正被触发双倍易伤。
    enemy_dual_vulnerable: bool
    #: 己方加密等级 = 对方干扰波难度等级（1~3，开局 1）。
    encryption_level: int
    #: 现在能不能改密钥。
    key_modifiable: bool
    #: 是否存在生效中的装甲加固区域。
    armor_active: bool
    #: 还剩几次装甲加固机会。
    armor_chances_left: int
    #: 加固区域圆心与半径，单位 mm；没有生效区域时服务器可能仍填上一次的值。
    armor_center_x_mm: int
    armor_center_y_mm: int
    armor_radius_mm: int

    def as_dict(self) -> dict:
        return {
            "dual_vulnerability_chances": self.dual_vulnerability_chances,
            "enemy_dual_vulnerable": self.enemy_dual_vulnerable,
            "encryption_level": self.encryption_level,
            "key_modifiable": self.key_modifiable,
            "armor_active": self.armor_active,
            "armor_chances_left": self.armor_chances_left,
            "armor_center_x_mm": self.armor_center_x_mm,
            "armor_center_y_mm": self.armor_center_y_mm,
            "armor_radius_mm": self.armor_radius_mm,
        }


def decode_marking(payload: bytes, *, own_team: str = "R") -> MarkingProgress:
    """解 ``0x020C``。``own_team`` 是 R/B（也认 red/blue）。

    协议表里按「对方 / 己方」写位，槽位名按红方视角记成 ``B*`` / ``R*``；
    己方是蓝方时把前缀对调。
    """
    if len(payload) != MARKING_LEN:
        raise ValueError(f"0x020C 数据段必须是 {MARKING_LEN} 字节，收到 {len(payload)}")
    side = own_team.upper()[:1]
    if side not in ("R", "B"):
        raise ValueError(f"未知阵营 {own_team!r}，应为 R/B 或 red/blue")
    enemy = "B" if side == "R" else "R"
    raw = int.from_bytes(payload, "little")

    def slot_name(slot: str) -> str:
        """表里的 B* 是对方、R* 是己方；换成当前视角的阵营前缀。"""
        return f"{enemy if slot.startswith('B') else side}{slot[1:]}"

    return MarkingProgress(
        raw=raw,
        enemy_vulnerable=tuple(
            slot_name(slot)
            for bit, slot, kind in MARKING_BITS
            if kind == "enemy_vulnerable" and raw >> bit & 1
        ),
        own_marked=tuple(
            slot_name(slot)
            for bit, slot, kind in MARKING_BITS
            if kind == "own_marked" and raw >> bit & 1
        ),
    )


def decode_radar_info(payload: bytes) -> RadarInfo:
    """解 ``0x020E``。"""
    if len(payload) != RADAR_INFO_LEN:
        raise ValueError(f"0x020E 数据段必须是 {RADAR_INFO_LEN} 字节，收到 {len(payload)}")
    first, second = payload[0], payload[1]
    center_x, center_y, radius = struct.unpack_from("<HHH", payload, 2)
    level = (first >> 3) & 0x03
    return RadarInfo(
        raw=bytes(payload),
        dual_vulnerability_chances=first & 0x03,
        enemy_dual_vulnerable=bool(first >> 2 & 1),
        # 协议写「开局为 1，最高为 3」。读到 0 说明服务器还没下发过，
        # 这里原样返回 0，由调用方决定要不要当成"未知"。
        encryption_level=level,
        key_modifiable=bool(first >> 5 & 1),
        armor_active=bool(second & 1),
        armor_chances_left=(second >> 1) & 0x07,
        armor_center_x_mm=center_x,
        armor_center_y_mm=center_y,
        armor_radius_mm=radius,
    )


#: 命令字 → 解码器。给串口读取线程分派用。
DECODERS = {
    CMD_MARKING_PROGRESS: decode_marking,
    CMD_RADAR_INFO: decode_radar_info,
}
