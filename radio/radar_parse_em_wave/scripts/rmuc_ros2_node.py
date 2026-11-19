#!/usr/bin/env python3
"""启动赛场无线电接收节点及其 GNU Radio 流图。"""

import argparse
from pathlib import Path
import signal
import subprocess

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.signals import SignalHandlerOptions

from rmuc.ros2_bridge import RadarParseEmWaveRos2Bridge
from rmuc.rx import RxWorker
from rmuc.sdr_control import SdrParameterController


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rx",), default="rx", help="赛场接收模式")
    parser.add_argument("--sdr-uri", help="接收板 IIO URI，默认 ip:192.168.2.1")
    parser.add_argument("--flowgraph-python", default="/usr/bin/python3", help="GNU Radio 的 Python")
    args, ros_args = parser.parse_known_args()

    share = Path(get_package_share_directory("radar_parse_em_wave"))
    flowgraph = share / "grc" / "RMUC2026_NanoSDR_RX.py"
    if not flowgraph.is_file():
        parser.error(f"流图未安装，请重新构建 radar_parse_em_wave：{flowgraph}")
    command = [args.flowgraph_python, str(flowgraph)]
    if args.sdr_uri:
        command.extend(("--sdr-rx-uri", args.sdr_uri))

    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = RadarParseEmWaveRos2Bridge()
    logger = node.get_logger()
    controller = SdrParameterController(
        lambda: node.demod_config,
        lambda message: logger.warning(message, throttle_duration_sec=5),
    )
    receiver = RxWorker(controller.active_config, node.publish_decoded_packet)
    process = None
    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def start_flowgraph() -> None:
        nonlocal process
        if process is not None:
            if process.poll() is None:
                return
            logger.warning(f"GNU Radio 流图已退出（{process.returncode}），正在重启")
        controller.reset_flowgraph()
        process = subprocess.Popen(command)

    try:
        start_flowgraph()
        controller.start()
        receiver.start()
        node.create_timer(2.0, start_flowgraph, clock=Clock(clock_type=ClockType.STEADY_TIME))
        logger.info("无线电接收节点已启动，等待 SDR 和有效业务帧")
        try:
            while not stop_requested and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            receiver.stop()
            controller.stop()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
