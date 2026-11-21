# -*- coding: utf-8 -*-
"""将接收的业务包发布到 ROS 2，并接收雷达主程序的解调配置。"""

from __future__ import annotations

from rclpy.node import Node

from radar_parse_em_wave.msg import (
    RadarParseEmWave0A01RobotPosition,
    RadarParseEmWave0A02RobotHp,
    RadarParseEmWave0A03RobotAmmo,
    RadarParseEmWave0A04FieldStatus,
    RadarParseEmWave0A05RobotBuff,
    RadarParseEmWave0A06InterferenceKey,
    RadarParseEmWaveDemodConfig,
)

from dataclasses import replace

from .constants import Team
from .referee import MarkingProgress, RadarInfo
from .referee_topics import DEMOD_CONFIG_TOPIC, REFEREE_TOPICS, RefereeTopics
from .state import DemodConfig


# ==================== ROS2 话题配置 ====================

RX_PACKET_TOPICS = {
    "0x0A01": "/rmuc/rx/packet_0x0a01",
    "0x0A02": "/rmuc/rx/packet_0x0a02",
    "0x0A03": "/rmuc/rx/packet_0x0a03",
    "0x0A04": "/rmuc/rx/packet_0x0a04",
    "0x0A05": "/rmuc/rx/packet_0x0a05",
    "0x0A06": "/rmuc/rx/packet_0x0a06",
}
"""业务包发布话题。topic 片段不能以数字开头，因此加 packet_ 前缀。"""

class RadarParseEmWaveRos2Bridge(Node):
    def __init__(self):
        super().__init__("radar_parse_em_wave_node")
        self.demod_config = DemodConfig()

        self._demod_sub = self.create_subscription(
            RadarParseEmWaveDemodConfig,
            DEMOD_CONFIG_TOPIC,
            self._handle_demod_config,
            10,
        )

        self._packet_publishers = {
            "0x0A01": self.create_publisher(
                RadarParseEmWave0A01RobotPosition,
                RX_PACKET_TOPICS["0x0A01"],
                10,
            ),
            "0x0A02": self.create_publisher(
                RadarParseEmWave0A02RobotHp,
                RX_PACKET_TOPICS["0x0A02"],
                10,
            ),
            "0x0A03": self.create_publisher(
                RadarParseEmWave0A03RobotAmmo,
                RX_PACKET_TOPICS["0x0A03"],
                10,
            ),
            "0x0A04": self.create_publisher(
                RadarParseEmWave0A04FieldStatus,
                RX_PACKET_TOPICS["0x0A04"],
                10,
            ),
            "0x0A05": self.create_publisher(
                RadarParseEmWave0A05RobotBuff,
                RX_PACKET_TOPICS["0x0A05"],
                10,
            ),
            "0x0A06": self.create_publisher(
                RadarParseEmWave0A06InterferenceKey,
                RX_PACKET_TOPICS["0x0A06"],
                10,
            ),
        }
        # 0x020C / 0x020E / demod_config 的发布与联合 runner 共用一份实现
        self.topics = RefereeTopics(self)

    def _handle_demod_config(self, msg: RadarParseEmWaveDemodConfig) -> None:
        """用主程序下发的阵营和等级更新接收配置。"""
        team_text = msg.team.strip().lower()
        if team_text in ("blue", "b", "1"):
            team = Team.BLUE
        elif team_text in ("red", "r", "0"):
            team = Team.RED
        else:
            self.get_logger().warning(f"忽略未知的接收阵营 {msg.team!r}", throttle_duration_sec=5)
            return

        level = int(msg.interference_level)
        if level not in (1, 2, 3):
            self.get_logger().warning(f"忽略非法的干扰等级 {level}", throttle_duration_sec=5)
            return

        config = DemodConfig(team, level)
        if config != self.demod_config:
            self.demod_config = config
            self.get_logger().info(f"收到解调配置 {team.value}-{level}")

    def set_interference_level(self, level: int) -> bool:
        """用裁判系统下发的加密等级替换解调等级，保留阵营。

        返回是否真的变了——调用方据此决定要不要重下发 SDR 参数。
        """
        level = int(level)
        if level not in (1, 2, 3):
            return False
        if self.demod_config.interference_level == level:
            return False
        self.demod_config = replace(self.demod_config, interference_level=level)
        self.get_logger().info(f"裁判系统下发加密等级 {level}，已切换解调等级")
        return True

    def publish_marking(self, marking: MarkingProgress, *, own_team: str = "R") -> None:
        """发布 0x020C（对方易伤 / 己方特殊标识）。"""
        self.topics.publish_marking(marking, own_team=own_team)

    def publish_radar_info(self, info: RadarInfo) -> None:
        """发布 0x020E（自主决策信息同步）。"""
        self.topics.publish_radar_info(info)

    def publish_demod_config(self) -> None:
        """把当前解调配置发一次。

        单独运行接收节点、由 ``--referee-level`` 驱动时用：等级变了必须让融合侧
        知道，不能只改节点内部值（旧实现就是这样，融合端一直用旧等级）。
        """
        config = self.demod_config
        self.topics.publish_demod_config(config.team.value, config.interference_level)

    def publish_decoded_packet(self, decoded: dict) -> None:
        """发布通过 CRC 和业务长度校验的数据。"""
        command = decoded["cmd"]
        self._packet_publishers[command].publish(self._build_packet_message(command, decoded))

    def _build_packet_message(self, command: str, decoded: dict):
        """根据业务包 ID 构造对应的 ROS2 消息。"""
        if command == "0x0A01":
            msg = RadarParseEmWave0A01RobotPosition()
            self._fill_header(msg, "rmuc_0a01")
            positions = decoded.get("positions", {})
            self._fill_position_msg(msg, positions)
            return msg

        if command == "0x0A02":
            msg = RadarParseEmWave0A02RobotHp()
            self._fill_header(msg, "rmuc_0a02")
            hp = decoded.get("hp", {})
            msg.hero_hp = self._uint16(hp.get("hero"))
            msg.engineer_hp = self._uint16(hp.get("eng"))
            msg.infantry3_hp = self._uint16(hp.get("inf3"))
            msg.infantry4_hp = self._uint16(hp.get("inf4"))
            msg.sentry_hp = self._uint16(hp.get("sentry"))
            return msg

        if command == "0x0A03":
            msg = RadarParseEmWave0A03RobotAmmo()
            self._fill_header(msg, "rmuc_0a03")
            ammo = decoded.get("ammo", {})
            msg.hero_ammo = self._uint16(ammo.get("hero"))
            msg.infantry3_ammo = self._uint16(ammo.get("inf3"))
            msg.infantry4_ammo = self._uint16(ammo.get("inf4"))
            msg.aerial_ammo = self._uint16(ammo.get("aerial"))
            msg.sentry_ammo = self._uint16(ammo.get("sentry"))
            return msg

        if command == "0x0A04":
            msg = RadarParseEmWave0A04FieldStatus()
            self._fill_header(msg, "rmuc_0a04")
            field = decoded.get("field", {})
            msg.remain_coins = self._uint16(field.get("remain_coins"))
            msg.total_coins = self._uint16(field.get("total_coins"))
            msg.status_flags = self._uint32(field.get("status_flags"))
            return msg

        if command == "0x0A05":
            msg = RadarParseEmWave0A05RobotBuff()
            self._fill_header(msg, "rmuc_0a05")
            self._fill_buff_msg(msg, decoded)
            return msg

        if command == "0x0A06":
            msg = RadarParseEmWave0A06InterferenceKey()
            self._fill_header(msg, "rmuc_0a06")
            msg.key = str(decoded.get("key", ""))
            return msg

        return None

    def _fill_header(self, msg, frame_id: str) -> None:
        """填充消息头时间戳和 frame_id。"""
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

    @staticmethod
    def _fill_position_msg(msg: RadarParseEmWave0A01RobotPosition, positions: dict) -> None:
        """填充 0x0A01 机器人位置消息。"""
        mapping = {
            "hero": ("hero_x", "hero_y"),
            "eng": ("engineer_x", "engineer_y"),
            "inf3": ("infantry3_x", "infantry3_y"),
            "inf4": ("infantry4_x", "infantry4_y"),
            "aerial": ("aerial_x", "aerial_y"),
            "sentry": ("sentry_x", "sentry_y"),
        }
        for robot_name, (x_attr, y_attr) in mapping.items():
            position = positions.get(robot_name, {})
            setattr(msg, x_attr, int(position.get("x", 0)))
            setattr(msg, y_attr, int(position.get("y", 0)))

    def _fill_buff_msg(self, msg: RadarParseEmWave0A05RobotBuff, decoded: dict) -> None:
        """填充 V2 版 0x0A05 增益、哨兵姿态和主要状态消息。"""
        buffs = decoded.get("buffs", {})
        prefix_map = {
            "hero": "hero",
            "eng": "engineer",
            "inf3": "infantry3",
            "inf4": "infantry4",
            "sentry": "sentry",
        }
        for robot_name, prefix in prefix_map.items():
            buff = buffs.get(robot_name, {})
            setattr(msg, f"{prefix}_heal", self._uint8(buff.get("heal")))
            setattr(msg, f"{prefix}_cool", self._uint16(buff.get("cool")))
            setattr(msg, f"{prefix}_def", self._uint8(buff.get("def")))
            setattr(msg, f"{prefix}_vuln", self._uint8(buff.get("vuln")))
            setattr(msg, f"{prefix}_atk", self._uint16(buff.get("atk")))

        statuses = decoded.get("robot_statuses", {})
        msg.sentry_posture = self._uint8(decoded.get("sentry_posture"))
        msg.hero_status = self._uint8(statuses.get("hero"))
        msg.engineer_status = self._uint8(statuses.get("eng"))
        msg.infantry3_status = self._uint8(statuses.get("inf3"))
        msg.infantry4_status = self._uint8(statuses.get("inf4"))
        msg.sentry_status = self._uint8(statuses.get("sentry"))

    @staticmethod
    def _uint8(value) -> int:
        """转换为 uint8 范围整数。"""
        return max(0, min(0xFF, int(value or 0)))

    @staticmethod
    def _uint16(value) -> int:
        """转换为 uint16 范围整数。"""
        return max(0, min(0xFFFF, int(value or 0)))

    @staticmethod
    def _uint32(value) -> int:
        """转换为 uint32 范围整数。"""
        return max(0, min(0xFFFFFFFF, int(value or 0)))
