#!/usr/bin/env python3
"""Run JGRadar live vision and RMUC radio subscriptions on one monotonic clock."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jg_radar.config.loader import load_config
from jg_radar.integration.game_radio import POSITION_SLOTS, PROTOCOL, RadioAdapter
from jg_radar.io.sink import JsonlSink
from jg_radar.pipeline.build import build_loop, build_stage_b
from jg_radar.types import FIELD_MAX_X_CM, FIELD_MAX_Y_CM


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", choices=("red", "blue"), required=True, help="own team")
    parser.add_argument("--config", action="append", default=[], help="vision YAML; repeat to overlay")
    parser.add_argument("--radio-frame", choices=("red_field", "own_team"), default="red_field")
    parser.add_argument("--interference-level", choices=("1", "2", "3", "external"), default="1",
                        help="receiver level to publish, or external to follow /rmuc/demod_config")
    parser.add_argument("--input-latency", type=float, default=0.0,
                        help="measured RF-to-ROS-publication delay in seconds")
    parser.add_argument("--radio-only", action="store_true", help="subscribe and arbitrate without camera")
    parser.add_argument("--out", default="", help="optional fused JSONL; no IQ recording")
    parser.add_argument("--seconds", type=float, default=0.0, help="stop after this duration; 0=unlimited")
    parser.add_argument("--check", action="store_true", help="check ROS/types/config without opening devices")
    parser.add_argument("--referee-tx", action="store_true",
                        help="按 4.8Hz 把融合结果发成 0x0305（雷达→服务器→选手端）")
    parser.add_argument("--referee-serial", default="",
                        help="0x0305 的串口设备，例如 /dev/ttyUSB0；不填则只统计不发送")
    parser.add_argument("--referee-baud", type=int, default=115200)
    parser.add_argument("--referee-rate-hz", type=float, default=4.8,
                        help="发送频率，协议上限 5Hz，写大了会被夹到 5")
    parser.add_argument("--referee-stale-policy", choices=("drop", "send_stale"), default="drop",
                        help="坐标过期后：drop 写 (0,0)，send_stale 照发旧值")
    parser.add_argument("--referee-stale-after", type=float, default=0.5,
                        help="坐标年龄超过多少秒算过期")
    parser.add_argument("--referee-log", default="",
                        help="把每帧 0x0305 的内容按 JSONL 落盘，供对账与论文统计")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("--seconds must be finite and nonnegative")
    if not math.isfinite(args.input_latency) or args.input_latency < 0:
        parser.error("--input-latency must be finite and nonnegative")
    if not args.config and not args.radio_only and not args.check:
        parser.error("live vision requires --config with real camera calibration and models")
    return args


def _build_referee_tx(args):
    """构造 0x0305 发送端。串口打不开就退化成只统计，不拖垮视觉主循环。"""
    from jg_radar.integration.referee_tx import MAX_RATE_HZ, RobotPositionSender

    radio_package = ROOT / "radio" / "radar_parse_em_wave"
    if str(radio_package) not in sys.path:
        sys.path.insert(0, str(radio_package))
    from rmuc.referee_frame import RadarPositionWriter

    writer = None
    if args.referee_serial:
        try:
            writer = RadarPositionWriter(
                args.referee_serial,
                baudrate=args.referee_baud,
                logger=lambda message: print(message, flush=True),
            )
        except Exception as exc:
            print(json.dumps({"referee_tx_error": str(exc), "serial": args.referee_serial,
                              "mode": "dry_run"}, ensure_ascii=False), flush=True)

    if writer is not None:
        send = writer
    else:
        counted = {"frames": 0}

        def send(payload: bytes) -> None:  # 干跑：只数帧，不碰设备
            counted["frames"] += 1

    sender = RobotPositionSender(
        send,
        own_team=args.team,
        rate_hz=args.referee_rate_hz,
        stale_policy=args.referee_stale_policy,
        stale_after_s=args.referee_stale_after,
    )
    if sender.clamped:
        print(json.dumps({"referee_tx_error": f"发送频率被夹到协议上限 {MAX_RATE_HZ:g} Hz",
                          "requested": args.referee_rate_hz, "applied": sender.rate_hz},
                         ensure_ascii=False), flush=True)
    return sender, writer


def _log_referee_frame(handle, referee) -> None:
    """把发出的一帧 0x0305 记成一行 JSON，供对账和论文统计。"""
    if handle is None:
        return
    from jg_radar.integration.referee_tx import parse_radar_payload

    payload = referee.last_payload
    record = {
        "t": round(time.monotonic(), 3),
        "own_team": referee.own_team,
        "slots": {
            name: list(point)
            for name, point in parse_radar_payload(payload, own_team=referee.own_team).items()
        },
        "max_age_s": (
            None if referee.stats.last_max_age_s is None
            else round(referee.stats.last_max_age_s, 3)
        ),
        "stale_policy": referee.stale_policy,
        "rate_hz": referee.rate_hz,
    }
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def main(argv=None):
    args = parse_args(argv)
    paths = [ROOT / "config/default.yaml", *args.config]
    cfg = load_config(*paths)
    if (cfg.field.size_x_cm, cfg.field.size_y_cm) != (FIELD_MAX_X_CM, FIELD_MAX_Y_CM):
        raise ValueError("RMUC 2026 radio adapter requires a 2800 x 1500 cm field")
    if not all(side + role in cfg.slot_names() for side in ("R", "B") for role, _ in POSITION_SLOTS):
        raise ValueError("radio position slots require both sides and roles 1,2,3,4,6,7")
    # Keep ROS optional for all existing vision-only tools.
    try:
        from jg_radar.integration.ros_radio import RadioBridge
    except ImportError as exc:
        raise RuntimeError("ROS 2/radar_parse_em_wave unavailable. Use tools/start_radio_vision.sh "
                           "with a ROS-compatible Python environment.") from exc
    if args.check:
        referee = None
        if args.referee_tx:
            from jg_radar.integration.referee_tx import MAX_RATE_HZ
            referee = {
                "serial": args.referee_serial or None,
                "rate_hz": min(args.referee_rate_hz, MAX_RATE_HZ),
                "rate_clamped": args.referee_rate_hz > MAX_RATE_HZ,
                "stale_policy": args.referee_stale_policy,
                "stale_after_s": args.referee_stale_after,
            }
        print(json.dumps({"ros_messages": "ready", "team": args.team,
                          "protocol": PROTOCOL, "radio_frame": args.radio_frame,
                          "interference_level": args.interference_level,
                          "referee_tx": referee,
                          "vision_configs": [str(p) for p in paths]}, ensure_ascii=False))
        return 0
    loop = None
    if args.radio_only:
        stage_b = build_stage_b(cfg)
    else:
        if cfg.detect.backend != "onnx" or not cfg.detect.car_model:
            raise ValueError("live integration requires configured ONNX vision models")
        if tuple(cfg.transform.t) == (0.0, 0.0, 0.0):
            raise ValueError("camera extrinsics are placeholders; load your calibrated YAML")
        # Live ROS data must not be mixed with video/replay timestamps.
        cfg = replace(cfg, io=replace(cfg.io, frame_source="camera"))
        loop = build_loop(cfg, threaded=True)
        stage_b = loop.stage_b
    adapter = RadioAdapter(stage_b.buffer, team=args.team,
                           coordinate_frame=args.radio_frame, input_latency_s=args.input_latency)
    stop = threading.Event()
    previous_signals = {sig: signal.signal(sig, lambda *_: stop.set())
                        for sig in (signal.SIGINT, signal.SIGTERM)}
    bridge = None
    sink = None
    referee = None
    referee_writer = None
    referee_log = None
    iterator = None
    started = time.monotonic()
    frames = 0
    last_report = started
    try:
        level = None if args.interference_level == "external" else int(args.interference_level)
        bridge = RadioBridge(adapter, interference_level=level)
        if args.out:
            sink = JsonlSink(args.out)
        if args.referee_tx:
            referee, referee_writer = _build_referee_tx(args)
            if args.referee_log:
                referee_log = open(args.referee_log, "a", encoding="utf-8")
        if loop is not None:
            iterator = loop.run()
        while not stop.is_set():
            if args.seconds and time.monotonic() - started >= args.seconds:
                break
            if iterator is None:
                out = stage_b.tick(now=time.monotonic())
            else:
                out = next(iterator, None)
                if out is None:
                    break
            bridge.publish(out)
            if sink is not None:
                sink.write(out)
            if referee is not None:
                referee.update(out)
                if referee.tick():
                    _log_referee_frame(referee_log, referee)
            frames += 1
            now = time.monotonic()
            if now - last_report >= 2.0:
                active = {name: slot.source.value for name, slot in out.robots.items() if slot.is_valid}
                report = {"frames": frames, "radio_frames": adapter.position_frames,
                          "rejected_radio_frames": adapter.rejected_frames,
                          "active_sources": active}
                if referee is not None:
                    report["referee_tx"] = referee.snapshot()
                print(json.dumps(report, ensure_ascii=False), flush=True)
                last_report = now
            if iterator is None:
                stop.wait(0.05)
    finally:
        if iterator is not None:
            iterator.close()
        elif loop is not None:
            loop.source.close()
        if bridge is not None:
            bridge.close()
        if sink is not None:
            sink.close()
        if referee_log is not None:
            referee_log.close()
        if referee_writer is not None:
            referee_writer.close()
        if referee is not None:
            print(json.dumps({"referee_tx_final": referee.snapshot()}, ensure_ascii=False),
                  flush=True)
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
    print(json.dumps({"frames": frames, "radio_frames": adapter.position_frames}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"radio/vision: {exc}", file=sys.stderr)
        raise SystemExit(1)
