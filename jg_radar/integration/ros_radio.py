"""ROS 2 side of the integration. Import this only in a sourced Jazzy runtime."""
from __future__ import annotations

import json
import threading
import time

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from radio_interface.msg import Position, Hp, Fire, State, Buff, Password
from std_msgs.msg import String

from jg_radar.integration.rm_radio import RadioAdapter


class RadioBridge:
    def __init__(self, adapter: RadioAdapter):
        self.adapter = adapter
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = Node("jg_radar_radio_bridge", context=self.context,
                         use_global_arguments=False)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self._subscriptions = [self.node.create_subscription(
            Position, "/robot_position", self._position, qos)]
        for topic, kind, msg_type in (
            ("/radio_hp", "hp", Hp), ("/radio_fire", "fire", Fire),
            ("/radio_state", "state", State), ("/radio_buff", "buff", Buff),
            ("/key_usart_sender", "password", Password)):
            self._subscriptions.append(self.node.create_subscription(
                msg_type, topic, self._status_callback(kind), qos))
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
    def _timing(info):
        now = time.monotonic()
        # source_timestamp is DDS publisher system time, not RF acquisition time.
        stamp = int(info["source_timestamp"] if isinstance(info, dict) else info.source_timestamp)
        if stamp <= 0:
            raise ValueError("DDS publisher timestamp unavailable")
        return now, (time.time_ns() - stamp) / 1e9

    def _warn(self, error):
        now = time.monotonic()
        if now - self._last_warning >= 2:
            self.node.get_logger().warning(str(error))
            self._last_warning = now

    def _position(self, msg, info):
        try:
            now, age = self._timing(info)
            self.adapter.position(msg.x, msg.y, received_at=now, transport_age_s=age)
        except ValueError as exc:
            self._warn(exc)

    def _status_callback(self, kind):
        def receive(msg, info):
            try:
                now, age = self._timing(info)
                fields = msg.get_fields_and_field_types()
                data = {name: [int(v) for v in value] if hasattr(value, "__iter__") else int(value)
                        for name in fields for value in (getattr(msg, name),)}
                self.adapter.status(kind, data, received_at=now, transport_age_s=age)
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
            # The original typed /key_usart_sender remains the key delivery interface.
            if "password" in snapshot["messages"]:
                snapshot["messages"]["password"] = {
                    key: value for key, value in snapshot["messages"]["password"].items()
                    if key != "data"}
            self.status_pub.publish(String(data=json.dumps(snapshot, ensure_ascii=False,
                                                           allow_nan=False)))
            self._last_status = now

    def close(self):
        self.executor.shutdown(timeout_sec=2.0)
        self._thread.join(timeout=2.0)
        self.node.destroy_node()
        if self.context.ok():
            self.context.shutdown()
