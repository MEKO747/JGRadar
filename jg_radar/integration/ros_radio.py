"""Connect the RMUC receiver and JGRadar over ROS 2."""
from __future__ import annotations

import json
import threading
import time

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from radar_parse_em_wave.msg import (
    RadarParseEmWave0A01RobotPosition,
    RadarParseEmWave0A02RobotHp,
    RadarParseEmWave0A03RobotAmmo,
    RadarParseEmWave0A04FieldStatus,
    RadarParseEmWave0A05RobotBuff,
    RadarParseEmWave0A06InterferenceKey,
    RadarParseEmWaveDemodConfig,
)
from std_msgs.msg import String

from jg_radar.integration.game_radio import POSITION_SLOTS, RadioAdapter
from jg_radar.integration.level_sync import LevelSync


DEMOD_CONFIG_TOPIC = "/rmuc/demod_config"
POSITION_TOPIC = "/rmuc/rx/packet_0x0a01"
STATUS_TOPICS = (
    ("/rmuc/rx/packet_0x0a02", "hp", RadarParseEmWave0A02RobotHp),
    ("/rmuc/rx/packet_0x0a03", "fire", RadarParseEmWave0A03RobotAmmo),
    ("/rmuc/rx/packet_0x0a04", "state", RadarParseEmWave0A04FieldStatus),
    ("/rmuc/rx/packet_0x0a05", "buff", RadarParseEmWave0A05RobotBuff),
    ("/rmuc/rx/packet_0x0a06", "password", RadarParseEmWave0A06InterferenceKey),
)
TEAM_NAMES = {"red": "red", "r": "red", "0": "red", "blue": "blue", "b": "blue", "1": "blue"}


class RadioBridge:
    def __init__(self, adapter: RadioAdapter, *, interference_level: int | None):
        self.adapter = adapter
        self._config: tuple[str, int] | None = None
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = Node("jg_radar_radio_bridge", context=self.context,
                         use_global_arguments=False)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.node.create_subscription(RadarParseEmWave0A01RobotPosition, POSITION_TOPIC, self._position, qos)
        for topic, kind, msg_type in STATUS_TOPICS:
            self.node.create_subscription(msg_type, topic, self._status_callback(kind), qos)
        self.node.create_subscription(
            RadarParseEmWaveDemodConfig, DEMOD_CONFIG_TOPIC, self._receive_config, 10)
        # 只在启动时发一次、之后**变化时才发**。以前这里每秒重发一次固定等级，
        # 把接收节点刚从 0x020E 切到的等级又顶回去——那是「等级不生效」的主因。
        self._config_pub = self.node.create_publisher(
            RadarParseEmWaveDemodConfig, DEMOD_CONFIG_TOPIC, 10)
        self._level_sync = LevelSync(self._publish_level, initial=interference_level)
        self._config = None
        if interference_level is not None:
            self._config = (adapter.team, int(interference_level))
            self._publish_level(int(interference_level))
        self._referee_topics = None
        self.fused_pub = self.node.create_publisher(String, "/jg_radar/fused", qos)
        self.status_pub = self.node.create_publisher(String, "/jg_radar/radio_status", qos)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self._error = None
        self._last_status = float("-inf")
        self._last_warning = float("-inf")
        self._thread = threading.Thread(target=self._spin, name="radio-ros", daemon=True)
        self._thread.start()

    @staticmethod
    def _timing(msg):
        now = time.monotonic()
        stamp = msg.header.stamp
        stamp_ns = stamp.sec * 10**9 + stamp.nanosec
        if stamp_ns <= 0:
            raise ValueError("Receiver packet is missing its publication timestamp")
        return now, (time.time_ns() - stamp_ns) / 1e9

    def _publish_level(self, level: int) -> None:
        """把配置发出去（调用方保证「只在变化时」）。"""
        self._config_pub.publish(
            RadarParseEmWaveDemodConfig(team=self.adapter.team, interference_level=int(level)))

    def set_level(self, level: int) -> bool:
        """``0x020E`` 驱动的等级变更：变了才发，返回是否发了。

        级别来自裁判系统，**比本地默认值优先**；同时更新 ``_config``，
        让 ``0x0A01`` 的受理条件立刻跟着变，不用等话题回声。
        """
        level = int(level)
        if level not in (1, 2, 3):
            return False
        if self._level_sync.level == level:
            return False
        self._config = (self.adapter.team, level)
        return self._level_sync.set(level)

    def level_snapshot(self) -> dict:
        """当前等级与「发了几次」——现场判断等级有没有跟着 0x020E 变。"""
        data = self._level_sync.snapshot()
        data["config"] = list(self._config) if self._config else None
        return data

    @property
    def referee_topics(self):
        """共用的话题发布器（0x020C / 0x020E / demod_config）。"""
        if self._referee_topics is None:
            from rmuc.referee_topics import RefereeTopics

            self._referee_topics = RefereeTopics(self.node)
        return self._referee_topics

    def publish_marking(self, marking, *, own_team: str = "R") -> None:
        self.referee_topics.publish_marking(marking, own_team=own_team)

    def publish_radar_info(self, info) -> None:
        self.referee_topics.publish_radar_info(info)

    def _receive_config(self, msg):
        team = TEAM_NAMES.get(msg.team.strip().lower())
        level = msg.interference_level
        if team is None or level not in (1, 2, 3):
            return
        config = team, level
        if config != self._config:
            self.adapter.clear()
            self._config = config

    def _warn(self, error):
        now = time.monotonic()
        if now - self._last_warning >= 2:
            self.node.get_logger().warning(str(error))
            self._last_warning = now

    def _position(self, msg):
        if self._config != (self.adapter.team, 3):
            return
        try:
            now, age = self._timing(msg)
            x = [getattr(msg, f"{field}_x") for _, field in POSITION_SLOTS]
            y = [getattr(msg, f"{field}_y") for _, field in POSITION_SLOTS]
            self.adapter.position(x, y, received_at=now, message_age_s=age)
        except ValueError as exc:
            self._warn(exc)

    def _status_callback(self, kind):
        def receive(msg):
            if self._config is None or self._config[0] != self.adapter.team:
                return
            if kind != "password" and self._config[1] != 3:
                return
            try:
                now, age = self._timing(msg)
                data = {} if kind == "password" else {
                    name: getattr(msg, name) for name in msg.get_fields_and_field_types()
                    if name != "header"
                }
                self.adapter.status(kind, data, received_at=now, message_age_s=age)
            except ValueError as exc:
                self._warn(exc)
        return receive

    def _spin(self):
        try:
            self.executor.spin()
        except BaseException as exc:
            if self.context.ok():
                self._error = exc

    def check(self):
        if self._error is not None:
            raise RuntimeError(f"ROS radio subscriber stopped: {self._error}") from self._error
        if not self.context.ok():
            raise RuntimeError("ROS context stopped")

    def publish(self, frame):
        self.check()
        self.fused_pub.publish(String(data=json.dumps(frame.to_dict(), ensure_ascii=False,
                                                      allow_nan=False)))
        now = time.monotonic()
        if now - self._last_status >= 0.2:
            snapshot = self.adapter.snapshot(now=now)
            self.status_pub.publish(String(data=json.dumps(snapshot, ensure_ascii=False,
                                                           allow_nan=False)))
            self._last_status = now

    def close(self):
        self.executor.shutdown(timeout_sec=2.0)
        self._thread.join(timeout=2.0)
        self.node.destroy_node()
        if self.context.ok():
            self.context.shutdown()
