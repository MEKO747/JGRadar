# -*- coding: utf-8 -*-
"""``0x020C`` / ``0x020E`` / 解调配置的话题发布——接收节点与联合 runner 共用。

以前这三处的发布各写一份：接收节点在自己类里，runner 在 ``RadioBridge`` 里，
字段映射一旦不一样就会出现「接收节点发的是对的、融合侧读到的却是别的」。
现在只有一个实现，两边都持有 :class:`RefereeTopics`。
"""

from __future__ import annotations

from radar_parse_em_wave.msg import (
    RadarParseEmWave020CMarking,
    RadarParseEmWave020ERadarInfo,
    RadarParseEmWaveDemodConfig,
)

from .referee import MarkingProgress, RadarInfo

DEMOD_CONFIG_TOPIC = "/rmuc/demod_config"
"""解调参数配置话题：谁读到 ``0x020E``，谁就在等级变化时发一次。"""

REFEREE_TOPICS = {
    "0x020C": "/rmuc/referee/packet_0x020c",
    "0x020E": "/rmuc/referee/packet_0x020e",
}
"""裁判系统下发给雷达的两条接口。和无线链路的 0x0Axx 分开一组话题。"""


class RefereeTopics:
    """在给定 rclpy 节点上发布三条雷达侧接口。

    ``node`` 只需要 ``create_publisher`` 和 ``get_clock``——
    接收节点的 bridge 和 runner 的 ``RadioBridge`` 传的都是自己的节点。
    """

    def __init__(self, node) -> None:
        self.node = node
        self._marking = node.create_publisher(
            RadarParseEmWave020CMarking, REFEREE_TOPICS["0x020C"], 10)
        self._radar_info = node.create_publisher(
            RadarParseEmWave020ERadarInfo, REFEREE_TOPICS["0x020E"], 10)
        self._demod = node.create_publisher(
            RadarParseEmWaveDemodConfig, DEMOD_CONFIG_TOPIC, 10)

    def _fill_header(self, msg, frame_id: str) -> None:
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

    def publish_marking(self, marking: MarkingProgress, *, own_team: str = "R") -> None:
        """发布 0x020C（对方易伤 / 己方特殊标识）。"""
        side = own_team.upper()[:1]
        enemy = "B" if side == "R" else "R"
        msg = RadarParseEmWave020CMarking()
        self._fill_header(msg, "rmuc_020c")
        msg.raw = int(marking.raw)
        for role, suffix in (("1", "hero"), ("3", "infantry3"),
                             ("4", "infantry4"), ("7", "sentry")):
            setattr(msg, f"enemy_{suffix}_vulnerable", f"{enemy}{role}" in marking.enemy_vulnerable)
            setattr(msg, f"own_{suffix}_marked", f"{side}{role}" in marking.own_marked)
        self._marking.publish(msg)

    def publish_radar_info(self, info: RadarInfo) -> None:
        """发布 0x020E（自主决策信息同步）。"""
        msg = RadarParseEmWave020ERadarInfo()
        self._fill_header(msg, "rmuc_020e")
        msg.dual_vulnerability_chances = int(info.dual_vulnerability_chances)
        msg.enemy_dual_vulnerable = bool(info.enemy_dual_vulnerable)
        msg.encryption_level = int(info.encryption_level)
        msg.key_modifiable = bool(info.key_modifiable)
        msg.armor_active = bool(info.armor_active)
        msg.armor_chances_left = int(info.armor_chances_left)
        msg.armor_center_x_mm = int(info.armor_center_x_mm)
        msg.armor_center_y_mm = int(info.armor_center_y_mm)
        msg.armor_radius_mm = int(info.armor_radius_mm)
        self._radar_info.publish(msg)

    def publish_demod_config(self, team: str, level: int) -> None:
        """发布解调配置。调用方负责「只在变化时调用」，这里不做去重。"""
        self._demod.publish(RadarParseEmWaveDemodConfig(team=str(team),
                                                        interference_level=int(level)))
