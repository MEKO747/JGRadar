#!/usr/bin/env python3
"""Run JGRadar live vision and RM_Radio subscriptions on one monotonic clock."""
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
from jg_radar.integration.rm_radio import RadioAdapter
from jg_radar.io.sink import JsonlSink
from jg_radar.pipeline.build import build_loop, build_stage_b


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", choices=("red", "blue"), required=True, help="own team")
    parser.add_argument("--config", action="append", default=[], help="vision YAML; repeat to overlay")
    parser.add_argument("--radio-frame", choices=("red_field", "own_team"), default="red_field")
    parser.add_argument("--input-latency", type=float, default=0.0,
                        help="measured RF/decode delay in seconds; headerless protocol cannot measure it")
    parser.add_argument("--radio-only", action="store_true", help="subscribe and arbitrate without camera")
    parser.add_argument("--out", default="", help="optional fused JSONL; no IQ recording")
    parser.add_argument("--seconds", type=float, default=0.0, help="stop after this duration; 0=unlimited")
    parser.add_argument("--check", action="store_true", help="check ROS/types/config without opening devices")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("--seconds must be finite and nonnegative")
    if not math.isfinite(args.input_latency) or args.input_latency < 0:
        parser.error("--input-latency must be finite and nonnegative")
    if not args.config and not args.radio_only and not args.check:
        parser.error("live vision requires --config with real camera calibration and models")
    return args


def main(argv=None):
    args = parse_args(argv)
    paths = [ROOT / "config/default.yaml", *args.config]
    cfg = load_config(*paths)
    if cfg.field.size_x_cm != 2800 or cfg.field.size_y_cm != 1500:
        raise ValueError("tdt_2026 radio adapter requires the existing 2800 x 1500 cm field")
    if not all(side + role in cfg.slot_names() for side in ("R", "B")
               for role in ("1", "2", "3", "4", "6", "7")):
        raise ValueError("radio position slots require both sides and roles 1,2,3,4,6,7")
    # Keep ROS optional for all existing vision-only tools.
    try:
        from jg_radar.integration.ros_radio import RadioBridge
    except ImportError as exc:
        raise RuntimeError("ROS 2/radio_interface unavailable; source radio/RM_Radio/env.sh "
                           "and use a Python 3.12 environment with rclpy") from exc
    if args.check:
        print(json.dumps({"ros_messages": "ready", "team": args.team,
                          "protocol": "tdt_2026", "radio_frame": args.radio_frame,
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
    iterator = None
    started = time.monotonic()
    frames = 0
    last_report = started
    try:
        bridge = RadioBridge(adapter)
        if args.out:
            sink = JsonlSink(args.out)
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
            frames += 1
            now = time.monotonic()
            if now - last_report >= 2.0:
                active = {name: slot.source.value for name, slot in out.robots.items() if slot.is_valid}
                print(json.dumps({"frames": frames, "radio_frames": adapter.position_frames,
                                  "rejected_radio_frames": adapter.rejected_frames,
                                  "active_sources": active}, ensure_ascii=False), flush=True)
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
