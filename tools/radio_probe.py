#!/usr/bin/env python3
"""JGRadar 无线电接收诊断：判断链路卡在帧同步、序列帧还是业务解码。

两种模式：

``bits``（默认）
    旁听 GNU Radio 推给接收节点的两路比特流（53103 信息波 / 53104 干扰波），
    直接喂进接收包自己的解码器，按级统计
    「比特 → 空口帧 → 序列帧 → 业务包」。这是判断
    「空口没信号」和「有信号但解不出」的唯一手段。

``watch``
    订阅 6 个 ``/rmuc/rx`` 话题，把解出的业务包按人可读格式打印，
    并每 5 秒汇总一次帧数。只回答「解出了什么」，不回答「为什么没解出」。

用法::

    source ~/.jg_env.zsh
    python3 tools/radio_probe.py                  # bits 模式，默认 10 秒
    python3 tools/radio_probe.py --seconds 30
    python3 tools/radio_probe.py --mode watch     # 看业务包内容

bits 模式不启动接收节点，也不占用设备；可以和其他实例同时跑。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RADIO_PACKAGE = ROOT / "radio" / "radar_parse_em_wave"
if str(RADIO_PACKAGE) not in sys.path:
    sys.path.insert(0, str(RADIO_PACKAGE))

from rmuc.coder import AirFrameDecoder, BusinessDecoder, SerialFrameDecoder  # noqa: E402
from rmuc.constants import (  # noqa: E402
    RX_INTERFERE_DEFAULT_PORT,
    RX_INFO_DEFAULT_PORT,
    WaveType,
)

WAVE_NAMES = {WaveType.INFO: "信息波", WaveType.INTERFERE: "干扰波"}

#: 低于这个同步率（次/秒）时，同步命中只按「偶发」处理，不足以说明成帧能力。
SPARSE_SYNC_RATE = 1.0


@dataclass
class ChannelStats:
    """单路比特流经过三级解码后的累计计数。"""

    wave: WaveType
    bits: int = 0
    sync: int = 0
    frames: int = 0
    decoded: int = 0
    air: AirFrameDecoder = field(init=False)
    serial: SerialFrameDecoder = field(init=False)

    def __post_init__(self) -> None:
        self.air = AirFrameDecoder(self.wave)
        self.serial = SerialFrameDecoder()

    def feed(self, unpacked_bytes: bytes) -> None:
        """喂入 GNU Radio 解调出的 unpacked bytes，并累加三级计数。"""
        self.bits += len(unpacked_bytes)
        for payload in self.air.feed_unpacked_bytes(unpacked_bytes):
            self.sync += 1
            for frame in self.serial.feed(payload):
                self.frames += 1
                if BusinessDecoder.decode(frame) is not None:
                    self.decoded += 1


def classify(stats: ChannelStats, seconds: float) -> str:
    """给出一路通道的诊断结论。

    判定顺序按解码链路的层级从上往下走，先报最早断掉的那一级，
    避免把「没有信号」误报成「解码失败」。
    """
    span = max(seconds, 1e-9)
    if stats.bits == 0:
        return "没收到任何比特：GNU Radio 流图没在跑，或端口不通"
    if stats.sync == 0:
        return "没有空口帧同步：这一路是噪声，没有可用信号"
    if stats.frames == 0:
        if stats.sync / span < SPARSE_SYNC_RATE:
            return f"只有偶发同步（{stats.sync} 次 / {span:.0f}s，达不到成帧频率）：信号很弱或误码过高"
        return "有同步但序列帧全失败：帧头、长度或 CRC8 不符"
    if stats.decoded == 0:
        return "序列帧通过 CRC16 但业务层解不出：命令字或载荷长度与当前协议不匹配"
    if stats.decoded < stats.frames:
        loss = stats.frames - stats.decoded
        return f"业务包部分失败：{stats.decoded}/{stats.frames} 解出，{loss} 个命令字或长度不匹配"
    return f"正常：每类约 {stats.decoded / span:.1f} 帧/秒，三级解码零丢弃"


def _describe(stats: ChannelStats, seconds: float) -> str:
    span = max(seconds, 1e-9)
    return (
        f"{WAVE_NAMES[stats.wave]:4s} "
        f"比特 {stats.bits / span:7.0f} B/s | "
        f"帧同步 {stats.sync:5d} | 序列帧 {stats.frames:5d} | 业务解码 {stats.decoded:5d}"
    )


def run_bits(args: argparse.Namespace) -> int:
    """旁听两路比特流，按级统计。"""
    import zmq

    ports = {WaveType.INFO: args.info_port, WaveType.INTERFERE: args.interfere_port}
    context = zmq.Context()
    stats = {wave: ChannelStats(wave) for wave in WAVE_NAMES}
    poller = zmq.Poller()
    sockets = {}
    for wave, port in ports.items():
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.connect(f"tcp://127.0.0.1:{port}")
        poller.register(socket, zmq.POLLIN)
        sockets[socket] = wave

    print(f"旁听 {args.seconds:.0f}s：信息波 {ports[WaveType.INFO]} / "
          f"干扰波 {ports[WaveType.INTERFERE]}（Ctrl+C 提前结束）", flush=True)
    started = last_report = time.monotonic()
    try:
        while time.monotonic() - started < args.seconds:
            for socket, _ in poller.poll(0.2):
                stats[sockets[socket]].feed(socket.recv())
            now = time.monotonic()
            if now - last_report >= args.interval:
                last_report = now
                elapsed = now - started
                for wave in WAVE_NAMES:
                    print(f"[{time.strftime('%H:%M:%S')}] {_describe(stats[wave], elapsed)}",
                          flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        for socket in sockets:
            socket.close()
        context.term()

    elapsed = max(time.monotonic() - started, 1e-9)
    print("--- 结论 ---", flush=True)
    for wave in WAVE_NAMES:
        print(f"{WAVE_NAMES[wave]}: {classify(stats[wave], elapsed)}", flush=True)
    return 0


WATCH_TOPICS = {
    "0x0A01": ("/rmuc/rx/packet_0x0a01", "机器人坐标"),
    "0x0A02": ("/rmuc/rx/packet_0x0a02", "血量"),
    "0x0A03": ("/rmuc/rx/packet_0x0a03", "弹量"),
    "0x0A04": ("/rmuc/rx/packet_0x0a04", "场地状态"),
    "0x0A05": ("/rmuc/rx/packet_0x0a05", "增益/哨兵"),
    "0x0A06": ("/rmuc/rx/packet_0x0a06", "干扰波密钥"),
}

POSITION_ROLES = (
    ("hero", "英雄1"),
    ("engineer", "工程2"),
    ("infantry3", "步兵3"),
    ("infantry4", "步兵4"),
    ("aerial", "空中6"),
    ("sentry", "哨兵7"),
)


def _centimetres(value: int) -> str:
    return "--" if value == 0 else f"{value / 100:.2f}m"


def format_message(code: str, msg) -> str:
    """把一条业务包格式化成一行人类可读文本。"""
    if code == "0x0A01":
        return " ".join(
            f"{label}({_centimetres(getattr(msg, f'{key}_x'))},{_centimetres(getattr(msg, f'{key}_y'))})"
            for key, label in POSITION_ROLES
        )
    if code == "0x0A02":
        return " ".join(
            f"{label}={getattr(msg, f'{key}_hp')}"
            for key, label in (("hero", "英雄"), ("engineer", "工程"), ("infantry3", "步兵3"),
                               ("infantry4", "步兵4"), ("sentry", "哨兵"))
        )
    if code == "0x0A03":
        return " ".join(
            f"{label}={getattr(msg, f'{key}_ammo')}"
            for key, label in (("hero", "英雄"), ("infantry3", "步兵3"), ("infantry4", "步兵4"),
                               ("aerial", "空中"), ("sentry", "哨兵"))
        )
    if code == "0x0A04":
        return f"金币 {msg.remain_coins}/{msg.total_coins} status_flags=0x{msg.status_flags:08x}"
    if code == "0x0A05":
        return f"哨兵姿态={msg.sentry_posture} 英雄(heal={msg.hero_heal} atk={msg.hero_atk}) " \
               f"工程(heal={msg.engineer_heal} atk={msg.engineer_atk})"
    return f'密钥"{msg.key}"'


def run_watch(args: argparse.Namespace) -> int:
    """订阅 6 个业务话题，打印解出的内容。"""
    import rclpy
    from rclpy.node import Node
    from radar_parse_em_wave import msg as radio_msgs

    message_types = {
        "0x0A01": radio_msgs.RadarParseEmWave0A01RobotPosition,
        "0x0A02": radio_msgs.RadarParseEmWave0A02RobotHp,
        "0x0A03": radio_msgs.RadarParseEmWave0A03RobotAmmo,
        "0x0A04": radio_msgs.RadarParseEmWave0A04FieldStatus,
        "0x0A05": radio_msgs.RadarParseEmWave0A05RobotBuff,
        "0x0A06": radio_msgs.RadarParseEmWave0A06InterferenceKey,
    }

    counts = {code: 0 for code in WATCH_TOPICS}
    started = time.monotonic()
    last_report = started

    class Watcher(Node):
        def __init__(self) -> None:
            super().__init__("jg_radio_probe_watch")
            for code, (topic, _label) in WATCH_TOPICS.items():
                self.create_subscription(
                    message_types[code], topic, lambda m, c=code: self.on_message(c, m), 10
                )

        def on_message(self, code: str, msg) -> None:
            counts[code] += 1
            stamp = getattr(msg.header.stamp, "sec", 0) + getattr(msg.header.stamp, "nanosec", 0) / 1e9
            age = time.time() - stamp
            print(f"[{time.strftime('%H:%M:%S')}] {code} {WATCH_TOPICS[code][1]} "
                  f"#{counts[code]} 延迟={age:.2f}s | {format_message(code, msg)}", flush=True)

    rclpy.init()
    node = Watcher()
    try:
        while rclpy.ok() and time.monotonic() - started < args.seconds:
            rclpy.spin_once(node, timeout_sec=0.2)
            now = time.monotonic()
            if now - last_report >= 5.0:
                last_report = now
                summary = " ".join(f"{code}={counts[code]}" for code in WATCH_TOPICS)
                print(f"[{time.strftime('%H:%M:%S')}] 汇总 {now - started:.0f}s | {summary}",
                      flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:  # 已经 shutdown 时 rclpy 会抛 RCLError
            pass
    if not any(counts.values()):
        print("整段时间没有任何业务包：先跑 --mode bits 确认卡在哪一级。", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("bits", "watch"), default="bits",
                        help="bits=诊断解码链路，watch=打印解出的业务包")
    parser.add_argument("--seconds", type=float, default=10.0, help="采样时长，默认 10 秒")
    parser.add_argument("--interval", type=float, default=2.0, help="bits 模式打印间隔，默认 2 秒")
    parser.add_argument("--info-port", type=int, default=RX_INFO_DEFAULT_PORT,
                        help=f"信息波比特流端口，默认 {RX_INFO_DEFAULT_PORT}")
    parser.add_argument("--interfere-port", type=int, default=RX_INTERFERE_DEFAULT_PORT,
                        help=f"干扰波比特流端口，默认 {RX_INTERFERE_DEFAULT_PORT}")
    args = parser.parse_args(argv)
    return run_watch(args) if args.mode == "watch" else run_bits(args)


if __name__ == "__main__":
    raise SystemExit(main())
