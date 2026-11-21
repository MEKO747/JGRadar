#!/usr/bin/env python3
"""用 pty 假串口跑**真 runner**，验裁判串口的单点所有权与等级通路。

pty 就是内核 tty，行为和 ``/dev/ttyUSB0`` 一样，区别只有物理层：
不接 Pluto、不接裁判机也能把「一个 fd 读写、等级变化才发布、拔线不崩」测到底。

两条用法（**输出落 ``out/referee_pty/<时间戳>/``，不许留在 /tmp**）::

    zsh -c 'source ~/.jg_env.zsh; source /opt/ros/jazzy/setup.zsh; \\
            source build/radio/install/local_setup.zsh; \\
            python3 tools/referee_pty_check.py --mode e2e'
    ... --mode unplug

证据等级 **C**：帧是合成注入的，不是空口；它证明实现正确性，
**不进阶段 3 验收表**（真机空口与实拍仍是零样本）。

退出码：全部断言通过 0，否则 1。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pty
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RADIO = ROOT / "radio" / "radar_parse_em_wave"
if str(RADIO) not in sys.path:
    sys.path.insert(0, str(RADIO))

from rmuc.coder import SerialFrameDecoder  # noqa: E402
from rmuc.referee_frame import build_frame  # noqa: E402

CMD_RADAR_POSITION = 0x0305


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 —— 没有 git 不影响跑，只是取证缺一项
        return ""


def radar_info_frame(level: int) -> bytes:
    """0x020E：byte0 的 bit3~4 是加密等级。"""
    return build_frame(0x020E, bytes([(int(level) & 0x03) << 3, 0, 0, 0, 0, 0, 0, 0]), seq=0)


def marking_frame(raw: int = 0b0000_0000_0000_0001) -> bytes:
    return build_frame(0x020C, int(raw).to_bytes(2, "little"), seq=1)


def drain(fd: int, sink: bytearray) -> None:
    try:
        chunk = os.read(fd, 65536)
    except (BlockingIOError, OSError):
        return
    if chunk:
        sink += chunk


def launch(session: str, log_path: Path, seconds: float) -> tuple[subprocess.Popen, int, int, str]:
    """开 pty、起真 runner（``--referee-serial`` 指到 pty 的从端）。"""
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    slave_name = os.ttyname(slave)
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "tools" / "run_radio_vision.py"),
         "--team", "red", "--radio-only", "--referee-tx",
         "--referee-serial", slave_name, "--referee-level",
         "--referee-session", session, "--referee-log", str(log_path),
         "--seconds", str(seconds)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc, master, slave, slave_name


def parse_reports(stdout: str) -> list[dict]:
    reports = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{") or '"interference_level"' not in line:
            continue
        try:
            reports.append(json.loads(line)["interference_level"])
        except (json.JSONDecodeError, KeyError):
            continue
    return reports


def mode_e2e(out_dir: Path) -> dict:
    session = f"pty_e2e_{datetime.now():%Y%m%d_%H%M%S}"
    log_path = out_dir / "referee_0305.jsonl"
    proc, master, slave, slave_name = launch(session, log_path, seconds=7)
    rx = bytearray()
    sent: set[int] = set()
    started = time.monotonic()
    deadline = started + 10
    while time.monotonic() < deadline and proc.poll() is None:
        drain(master, rx)
        elapsed = time.monotonic() - started
        if 3 not in sent and elapsed > 1.5:
            os.write(master, radar_info_frame(3))
            sent.add(3)
        if 2 not in sent and elapsed > 3.5:
            os.write(master, radar_info_frame(2))
            sent.add(2)
        if 0x020C not in sent and elapsed > 2.0:
            os.write(master, marking_frame())
            sent.add(0x020C)
        time.sleep(0.02)
    stdout = proc.communicate(timeout=15)[0]
    os.close(master)
    os.close(slave)

    frames = SerialFrameDecoder().feed(bytes(rx))
    positions = [frame for frame in frames if frame.cmd_id == CMD_RADAR_POSITION]
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()] if log_path.is_file() else []
    reports = parse_reports(stdout)
    return {
        "session_id": session,
        "pty": slave_name,
        "runner_rc": proc.returncode,
        "crashed": "Traceback" in stdout,
        "frames_on_wire": len(positions),
        "all_payload_48B": all(len(frame.payload) == 48 for frame in positions),
        "levels_reported": reports,
        "level_publishes": [report.get("level_publishes") for report in reports],
        "referee_log_rows": len(rows),
        "log_session_matches": all(row.get("session_id") == session for row in rows) if rows else False,
        "all_write_ok": all(row.get("write_ok") is True for row in rows) if rows else False,
        "write_mode": rows[0].get("write_mode") if rows else None,
        "log_path": str(log_path.relative_to(ROOT)),
        "log_sha256": sha256(log_path) if log_path.is_file() else None,
    }


def mode_unplug(out_dir: Path) -> dict:
    session = f"pty_unplug_{datetime.now():%Y%m%d_%H%M%S}"
    log_path = out_dir / "referee_unplug.jsonl"
    proc, master, slave, slave_name = launch(session, log_path, seconds=6)
    time.sleep(2.0)
    os.close(master)          # 拔线：master 端消失
    os.close(slave)
    stdout = proc.communicate(timeout=20)[0]
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()] if log_path.is_file() else []
    good = [row for row in rows if row.get("write_ok") is True]
    bad = [row for row in rows if row.get("write_ok") is not True]
    return {
        "session_id": session,
        "pty": slave_name,
        "runner_rc": proc.returncode,
        "crashed": "Traceback" in stdout,
        "rows": len(rows),
        "write_ok_true": len(good),
        "write_ok_false": len(bad),
        "last_error": (bad[-1].get("write_error") or "")[:120] if bad else None,
        "log_session_matches": all(row.get("session_id") == session for row in rows) if rows else False,
        "log_path": str(log_path.relative_to(ROOT)),
        "log_sha256": sha256(log_path) if log_path.is_file() else None,
    }


def verdict(mode: str, result: dict) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if result["runner_rc"] != 0:
        problems.append(f"runner 退出码 {result['runner_rc']}")
    if result["crashed"]:
        problems.append("runner 打了 Traceback")
    if not result["log_session_matches"]:
        problems.append("日志行的 session_id 与本次运行不符")
    if mode == "e2e":
        # 等级 1 → 3 → 2：初始发布 1 次，之后每次变化 +1，重复等级不再发
        publishes = [value for value in result["level_publishes"] if value is not None]
        if not publishes or publishes[-1] != 2:
            problems.append(f"等级发布次数不对：{publishes}")
        if 3 not in [report.get("interference_level") for report in result["levels_reported"]]:
            problems.append("注入 3 级后没有观测到等级 3")
        if result["frames_on_wire"] <= 0:
            problems.append("同一 pty 上没有看到 0x0305 帧")
        if not result["all_payload_48B"]:
            problems.append("有 0x0305 帧载荷不是 48 字节")
        if not result["all_write_ok"]:
            problems.append("日志里有 write_ok 不为 true 的行")
    else:
        if result["write_ok_false"] <= 0:
            problems.append("拔线后没有出现失败帧，注入没生效")
        if result["write_ok_true"] <= 0:
            problems.append("拔线前没有成功帧")
        if not result["last_error"]:
            problems.append("失败行没有带底层错误原文")
    return (not problems), problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("e2e", "unplug"), required=True)
    parser.add_argument("--out", default=str(ROOT / "out" / "referee_pty"))
    args = parser.parse_args(argv)

    out_dir = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = (mode_e2e if args.mode == "e2e" else mode_unplug)(out_dir)
    passed, problems = verdict(args.mode, result)
    report = {
        "schema": "referee-pty/1",
        "mode": args.mode,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evidence_grade": "C",
        # 边界写死：pty 是合成帧，证明实现正确性，不是空口证据
        "caveat": "pty 注入的是合成帧：证明单点所有权/等级通路/拔线不崩，"
                  "不证明空口能收到 0x020C/0x020E，也不进阶段 3 验收表",
        "commit": git("rev-parse", "--short", "HEAD") or "unknown",
        "dirty_files": len([line for line in git("status", "--porcelain").splitlines()
                            if line.strip()]),
        "tool": {"path": str(Path(__file__).relative_to(ROOT)),
                 "sha256": sha256(Path(__file__).resolve())},
        "result": result,
        "passed": passed,
        "problems": problems,
    }
    path = out_dir / "result.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(path), "passed": passed, "problems": problems,
                      "result": result}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
