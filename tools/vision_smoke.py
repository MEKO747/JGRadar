#!/usr/bin/env python3
"""无设备时的两条视觉冒烟路径，结果落盘到 ``out/``。

存在的理由：设备没到位之前，"管线能跑"这种结论如果只停在聊天记录里，等于没有。
这里把两种冒烟跑法固定下来，每次输出 `summary.json`（带时间、commit、配置摘要），
谁说"跑通了"都要能指着文件说话。

**证据等级（见 docs/experiments/README.md）**：

- ``--mode replay``（默认）：回放夹具里带身份的检测 + 真投影链 → **C 级（合成回放）**，
  能证明跟踪/投影/融合/发布通，不能证明检测精度。
- ``--mode uvc``：内置 UVC 摄像头取图 → 只证明取图与 Stage A 不崩 → **不算视觉证据**，
  一律不进验收表。

用法::

    .venv/bin/python tools/vision_smoke.py --mode replay
    .venv/bin/python tools/vision_smoke.py --mode uvc --seconds 2
"""

from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jg_radar.config import load_config
from jg_radar.pipeline.build import build_loop

DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "replay_seq_001"
DEFAULT_OUT = ROOT / "out" / "vision_smoke" / datetime.now().strftime("%Y%m%d_%H%M%S")

FIELD_X_MAX_CM, FIELD_Y_MAX_CM = 2900, 1700


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 —— 没有 git 不该让冒烟跑不起来
        return "unknown"


def _summarise(loop, outs) -> dict:
    states: Counter[str] = Counter()
    positioned = 0
    inside = 0
    last_positions: dict[str, list[int]] = {}
    for out in outs:
        for name, slot in out.robots.items():
            states[slot.state.value] += 1
            if slot.position_cm is None:
                continue
            positioned += 1
            x, y = slot.position_cm
            if -100 <= x <= FIELD_X_MAX_CM + 100 and -100 <= y <= FIELD_Y_MAX_CM + 100:
                inside += 1
            last_positions[name] = [x, y]
    return {
        "frames": len(outs),
        "slot_states": dict(states),
        "slot_frames": sum(states.values()),
        "slots_with_position": positioned,
        "positions_within_field": inside,
        "last_positions": last_positions,
        "tier_share": loop.stage_b.publisher.tier_share(),
    }


def run_replay(fixture: Path) -> dict:
    cfg = load_config(ROOT / "config" / "default.yaml")
    cfg = dataclasses.replace(cfg, io=dataclasses.replace(
        cfg.io, frame_source="replay", replay_path=str(fixture)))
    loop = build_loop(cfg)
    outs = loop.run_to_completion()
    return {"evidence_grade": "C", "note": "检测是造出来的，只证明跟踪/投影/融合/发布通",
            **_summarise(loop, outs)}


def run_uvc(seconds: float) -> dict:
    import cv2

    cfg = load_config(ROOT / "config" / "default.yaml")
    video = Path("/tmp") / "vision_smoke_uvc.mp4"
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("打不开 /dev/video0")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
    captured = 0
    for _ in range(max(1, int(seconds * 30))):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        captured += 1
    cap.release()
    writer.release()

    cfg = dataclasses.replace(cfg, io=dataclasses.replace(
        cfg.io, frame_source="video", video_path=str(video)))
    cfg = dataclasses.replace(cfg, detect=dataclasses.replace(cfg.detect, backend="stub"))
    loop = build_loop(cfg)
    outs = loop.run_to_completion()
    return {"evidence_grade": "not-evidence", "note": "内置 UVC + stub 检测，只证明取图与 Stage A 不崩",
            "captured_frames": captured, "video": str(video),
            **_summarise(loop, outs)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("replay", "uvc"), default="replay")
    parser.add_argument("--frames", default=str(DEFAULT_FIXTURE))
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    payload = run_replay(Path(args.frames)) if args.mode == "replay" else run_uvc(args.seconds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": args.mode,
        "commit": _git_commit(),
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        **payload,
    }
    path = out / f"{args.mode}_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "written": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
