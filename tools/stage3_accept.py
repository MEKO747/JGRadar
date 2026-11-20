#!/usr/bin/env python3
"""阶段 3 验收：一条命令产出 ``report.json``（schema ``stage3-accept/1``）。

判据写死，工具不许自创：

| check id | 判据 | 阈值 |
| --- | --- | --- |
| ``radio_accuracy`` | ``0x0A01`` 逐次判定 ``d<0.8m`` 比例 | ``>= 0.90`` |
| ``radio_uplink`` | ``0x0305`` 实发频率 / 1s 窗口 / 断流 | ``>= 4.7Hz``、``<=5 帧``、``gap<=0.5s`` |
| ``vision_calibration`` | 标定四要件 | 序列号 + 重标日期 + ``reproj_rms_px`` 实测 + 文件 sha256 |
| ``fusion_switch`` | 换源延迟 / 坐标跳变 | ``<= 0.4s``、``<= 150cm`` |

三态 ``pass | fail | blocked``：缺输入一律 ``blocked``，**绝不判 pass**。
报告结构与 ``docs/experiments/STAGE3_REPORT_SCHEMA.md`` 对齐（研究员定，改结构先改那份文档）。

证据等级：``A`` 只能由真机现场人工签字，**工具最高只给 B**；合成输入 ``C``；什么都没跑 ``D``。

用法::

    .venv/bin/python tools/stage3_accept.py --selftest
    .venv/bin/python tools/stage3_accept.py --radio-log out/live.jsonl --truth out/points.csv \\
        --fusion-log out/fused.jsonl --switch-event B1=12.3 --out out/stage3_accept
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import glob
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCHEMA = "stage3-accept/1"
# ---- 判据（改这里 = 改验收标准，必须有人拍板） ----
ACCURATE_M = 0.8
ACCURATE_RATIO_MIN = 0.90
UPLINK_RATE_MIN_HZ = 4.7
UPLINK_WINDOW_MAX_FRAMES = 5
UPLINK_MAX_GAP_S = 0.5
SWITCH_LATENCY_MAX_S = 0.4
JUMP_MAX_CM = 150.0

PASS, FAIL, BLOCKED = "pass", "fail", "blocked"
CHECK_ORDER = ("radio_accuracy", "radio_uplink", "vision_calibration", "fusion_switch")
_GRADE_RANK = {"A": 0, "B": 1, "C": 2, "D": 3}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 —— 没有 git 不该让验收跑不起来
        return ""


def _artifact(path: Path | None, kind: str, duration_s: float | None = None) -> dict | None:
    if path is None or not path.is_file():
        return None
    item = {"kind": kind, "path": str(path), "sha256": _sha256(path)}
    if duration_s is not None:
        item["duration_s"] = round(float(duration_s), 3)
    return item


def _sample_duration(path: Path, key: str = "t") -> float | None:
    try:
        times = [float(json.loads(line)[key]) for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    except Exception:  # noqa: BLE001
        return None
    return times[-1] - times[0] if len(times) > 1 else None


# ---------------------------------------------------------------------------
# 1) 无线电定位精度
# ---------------------------------------------------------------------------
def load_truth(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({"t0": float(row["t0"]), "t1": float(row["t1"]),
                         "slot": row["slot"].strip(),
                         "x_cm": float(row["x_cm"]), "y_cm": float(row["y_cm"])})
    return rows


def check_radio_accuracy(radio_log: Path | None, truth: list[dict] | None) -> dict:
    threshold = f"d<{ACCURATE_M}m 比例 >= {ACCURATE_RATIO_MIN:.0%}"
    if radio_log is None or not radio_log.is_file():
        return {"status": BLOCKED, "reason": "缺少 --radio-log", "value": None,
                "threshold": threshold, "artifacts": [], "grade": "D"}
    if not truth:
        return {"status": BLOCKED, "reason": "缺少 --truth：没有真值就算不出 d",
                "value": None, "threshold": threshold, "artifacts": [], "grade": "D"}

    distances = []
    for line in radio_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        t = float(row["t"])
        for slot, position in (row.get("slots") or {}).items():
            x_cm, y_cm = float(position[0]), float(position[1])
            if x_cm == 0.0 and y_cm == 0.0:
                continue  # (0,0) = 未发送，不构成判定
            for segment in truth:
                if segment["slot"] == slot and segment["t0"] <= t < segment["t1"]:
                    distances.append(math.hypot(x_cm - segment["x_cm"], y_cm - segment["y_cm"]))
                    break
    if not distances:
        return {"status": BLOCKED, "reason": "真值时间段内没有任何有效上报",
                "value": {"samples": 0}, "threshold": threshold,
                "artifacts": [], "grade": "B"}

    accurate = sum(1 for d in distances if d < ACCURATE_M * 100)
    ratio = accurate / len(distances)
    status = PASS if ratio >= ACCURATE_RATIO_MIN else FAIL
    return {
        "status": status, "reason": "" if status == PASS else f"准确率 {ratio:.1%} 低于阈值",
        "value": {"samples": len(distances), "accurate_ratio": round(ratio, 4),
                  "median_cm": round(sorted(distances)[len(distances) // 2], 2),
                  "max_cm": round(max(distances), 2)},
        "threshold": threshold, "artifacts": [radio_log], "grade": "B",
    }


# ---------------------------------------------------------------------------
# 2) 无线电上报节拍
# ---------------------------------------------------------------------------
def check_radio_uplink(radio_log: Path | None) -> dict:
    threshold = f">={UPLINK_RATE_MIN_HZ}Hz、窗口<={UPLINK_WINDOW_MAX_FRAMES}、gap<={UPLINK_MAX_GAP_S}s"
    if radio_log is None or not radio_log.is_file():
        return {"status": BLOCKED, "reason": "缺少 --radio-log", "value": None,
                "threshold": threshold, "artifacts": [], "grade": "D"}
    times = sorted(float(json.loads(line)["t"])
                   for line in radio_log.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(times) < 2:
        return {"status": BLOCKED, "reason": "帧数不足", "value": {"frames": len(times)},
                "threshold": threshold, "artifacts": [], "grade": "B"}

    span = times[-1] - times[0]
    rate = (len(times) - 1) / span if span > 0 else 0.0
    max_gap = max(b - a for a, b in zip(times, times[1:]))
    window_max, left = 0, 0
    for right, t in enumerate(times):
        while t - times[left] >= 1.0:
            left += 1
        window_max = max(window_max, right - left + 1)

    failures = []
    if rate < UPLINK_RATE_MIN_HZ:
        failures.append(f"频率 {rate:.2f}Hz < {UPLINK_RATE_MIN_HZ}")
    if window_max > UPLINK_WINDOW_MAX_FRAMES:
        failures.append(f"1s 窗口 {window_max} 帧 > {UPLINK_WINDOW_MAX_FRAMES}")
    if max_gap > UPLINK_MAX_GAP_S:
        failures.append(f"最大断流 {max_gap:.2f}s > {UPLINK_MAX_GAP_S}s")
    status = FAIL if failures else PASS
    return {"status": status, "reason": "；".join(failures),
            "value": {"frames": len(times), "effective_rate_hz": round(rate, 3),
                      "max_frames_1s": window_max, "max_gap_s": round(max_gap, 3)},
            "threshold": threshold, "artifacts": [radio_log], "grade": "B"}


# ---------------------------------------------------------------------------
# 3) 视觉标定四要件
# ---------------------------------------------------------------------------
def check_vision_calibration(camera_path: Path | None) -> dict:
    threshold = "序列号 + 重标日期 + reproj_rms_px 实测 + 文件 sha256"
    if camera_path is None or not camera_path.is_file():
        return {"status": BLOCKED, "reason": "缺少相机配置", "value": None,
                "threshold": threshold, "artifacts": [], "grade": "D"}
    import yaml

    data = yaml.safe_load(camera_path.read_text(encoding="utf-8")) or {}
    camera = data.get("camera") or {}
    calibration = data.get("calibration") or {}
    digest = _sha256(camera_path)

    missing = []
    name = str(camera.get("name", "") or "")
    if not name or name.lower() in ("unselected", "placeholder"):
        missing.append("相机序列号（name 仍是占位）")
    if camera.get("reproj_rms_px") is None:
        missing.append("reproj_rms_px 实测值")
    if not str(calibration.get("serial", "") or "").strip():
        missing.append("calibration.serial")
    if not str(calibration.get("date", "") or "").strip():
        missing.append("calibration.date")

    value = {"config": str(camera_path), "sha256": digest, "camera_name": name,
             "reproj_rms_px": camera.get("reproj_rms_px"),
             "calibration_serial": calibration.get("serial"),
             "calibration_date": calibration.get("date")}
    if missing:
        return {"status": BLOCKED, "reason": "标定证据不全：" + "、".join(missing),
                "value": value, "threshold": threshold, "artifacts": [], "grade": "D"}
    rms = float(camera["reproj_rms_px"])
    limit = float(camera.get("reproj_rms_max_px", 3.0) or 3.0)
    status = PASS if rms <= limit else FAIL
    return {"status": status,
            "reason": "" if status == PASS else f"reproj_rms_px {rms} > 上限 {limit}",
            "value": {**value, "limit_px": limit}, "threshold": threshold,
            "artifacts": [camera_path], "grade": "B"}


# ---------------------------------------------------------------------------
# 4) 多源融合
# ---------------------------------------------------------------------------
def check_fusion_switch(fusion_log: Path | None, switch_event: tuple[str, float] | None) -> dict:
    threshold = f"换源<={SWITCH_LATENCY_MAX_S}s、跳变<={JUMP_MAX_CM:.0f}cm"
    if fusion_log is None or not fusion_log.is_file():
        return {"status": BLOCKED, "reason": "缺少 --fusion-log", "value": None,
                "threshold": threshold, "artifacts": [], "grade": "D"}
    frames = [json.loads(line) for line in
              fusion_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not frames:
        return {"status": BLOCKED, "reason": "融合日志为空", "value": None,
                "threshold": threshold, "artifacts": [], "grade": "B"}

    previous: dict[str, tuple[float, tuple[float, float]]] = {}
    max_jump, jump_slot = 0.0, ""
    for frame in frames:
        t = float(frame.get("published_at") or frame.get("captured_at") or 0.0)
        for slot, robot in (frame.get("robots") or {}).items():
            position = robot.get("position_cm")
            if not position:
                continue
            point = (float(position[0]), float(position[1]))
            if slot in previous:
                last_t, last_point = previous[slot]
                if t > last_t:
                    jump = math.hypot(point[0] - last_point[0], point[1] - last_point[1])
                    if jump > max_jump:
                        max_jump, jump_slot = jump, slot
            previous[slot] = (t, point)

    value = {"max_jump_cm": round(max_jump, 1), "jump_slot": jump_slot}
    if switch_event is None:
        return {"status": BLOCKED, "reason": "缺少 --switch-event：没有刺激时刻量不出换源延迟",
                "value": value, "threshold": threshold, "artifacts": [], "grade": "B"}

    slot, event_t = switch_event
    baseline, latency = None, None
    for frame in frames:
        t = float(frame.get("published_at") or frame.get("captured_at") or 0.0)
        source = ((frame.get("robots") or {}).get(slot) or {}).get("source")
        if source is None:
            continue
        if t <= event_t:
            baseline = source
        elif baseline is not None and source != baseline:
            latency = t - event_t
            break
    if latency is None:
        return {"status": BLOCKED, "reason": f"事件后 {slot} 没有发生换源（时刻或槽位可能选错）",
                "value": value, "threshold": threshold, "artifacts": [], "grade": "B"}

    failures = []
    if latency > SWITCH_LATENCY_MAX_S:
        failures.append(f"换源 {latency:.2f}s > {SWITCH_LATENCY_MAX_S}s")
    if max_jump > JUMP_MAX_CM:
        failures.append(f"跳变 {max_jump:.0f}cm > {JUMP_MAX_CM:.0f}cm（{jump_slot}）")
    status = FAIL if failures else PASS
    return {"status": status, "reason": "；".join(failures),
            "value": {**value, "switch_latency_s": round(latency, 3), "switch_slot": slot},
            "threshold": threshold, "artifacts": [fusion_log], "grade": "B"}


# ---------------------------------------------------------------------------
# 设备与指纹
# ---------------------------------------------------------------------------
def detect_devices() -> list[dict]:
    """如实报"在不在"，不猜序列号。"""
    usb = ""
    try:
        usb = subprocess.run(["lsusb"], capture_output=True, text=True, check=False).stdout
    except Exception:  # noqa: BLE001
        usb = ""
    video = sorted(glob.glob("/dev/video*"))
    tty = sorted(glob.glob("/dev/ttyUSB*"))
    hik = "hikrobot" in usb.lower()
    return [
        {"kind": "sdr", "model": "ADALM-PLUTO", "serial": "",
         "transport": "usb", "present": "0456:b673" in usb},
        {"kind": "camera", "model": "Hikrobot" if hik else ("USB UVC" if video else "none"),
         "serial": "", "transport": "usb3", "present": bool(hik)},
        {"kind": "referee_serial", "model": "USB-TTL" if tty else "none",
         "serial": "", "transport": "usb", "present": bool(tty)},
    ]


def _fingerprint() -> dict:
    hashes = {}
    for name in ("default.yaml", "camera.yaml", "field.yaml"):
        path = ROOT / "config" / name
        if path.is_file():
            hashes[name] = _sha256(path)
    return hashes


def dirty_file_count() -> int:
    status = _git("status", "--porcelain")
    return len([line for line in status.splitlines() if line.strip()]) if status else 0


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def build_report(*, radio_log: Path | None, truth: list[dict] | None,
                 fusion_log: Path | None, switch_event: tuple[str, float] | None,
                 camera: Path | None, evidence_grade: str = "B",
                 devices: list[dict] | None = None, commands: list[str] | None = None) -> dict:
    checks = [
        {"id": "radio_accuracy", **check_radio_accuracy(radio_log, truth)},
        {"id": "radio_uplink", **check_radio_uplink(radio_log)},
        {"id": "vision_calibration", **check_vision_calibration(camera)},
        {"id": "fusion_switch", **check_fusion_switch(fusion_log, switch_event)},
    ]

    # 硬约束：pass 必须带 value/threshold/artifact(含 sha256)/grade，缺一个按 fail 处理
    for item in checks:
        artifacts = [{"path": str(path), "sha256": _sha256(Path(path))}
                     for path in item.get("artifacts") or [] if Path(path).is_file()]
        item["artifacts"] = artifacts
        if item["status"] == PASS:
            missing = []
            if not item.get("value"):
                missing.append("value")
            if not item.get("threshold"):
                missing.append("threshold")
            if not artifacts:
                missing.append("artifacts(含 sha256)")
            if not item.get("grade"):
                missing.append("grade")
            if missing:
                item["status"] = FAIL
                item["reason"] = "拿不出完整证据：" + "、".join(missing)
        if item["status"] == BLOCKED and not str(item.get("reason", "")).strip():
            item["status"] = FAIL
            item["reason"] = "blocked 但没有写原因"

    statuses = {item["id"]: item["status"] for item in checks}
    if FAIL in statuses.values():
        verdict = FAIL
    elif all(status == PASS for status in statuses.values()):
        verdict = PASS
    else:
        verdict = BLOCKED

    grades = {item["id"]: item.get("grade", "D") for item in checks}
    worst = max(grades.values(), key=lambda grade: _GRADE_RANK.get(grade, 3))
    # 工具永不自发 A：A 只能由真机现场 + 人工签字产生
    grade = "B" if _GRADE_RANK.get(worst, 3) <= _GRADE_RANK["B"] else worst
    grade = "C" if evidence_grade == "C" else grade
    if all(item["grade"] == "D" for item in checks):
        grade = "D"

    artifacts = []
    for path, kind in ((radio_log, "radio_log"), (fusion_log, "fusion_log")):
        item = _artifact(path, kind, _sample_duration(path) if path else None)
        if item:
            artifacts.append(item)
    camera_item = _artifact(camera, "camera_config")
    if camera_item:
        artifacts.append(camera_item)
    if truth:
        artifacts.append({"kind": "truth_segments", "segments": len(truth),
                          "slots": sorted({seg["slot"] for seg in truth})})

    caveats = ["本报告只验证四项判据，不证明检测精度；内置 UVC 画面不算视觉证据。"]
    blocked_ids = [item["id"] for item in checks if item["status"] == BLOCKED]
    if blocked_ids:
        caveats.append("以下项无证据，verdict 因此不是 pass：" + "、".join(blocked_ids))
    if any(item["status"] == FAIL for item in checks):
        caveats.append("存在 fail 项：先修现场问题再重跑，不要用离线数据顶替。")

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "goal": "JGRadar 2027 雷达站：无线电解调 → 多源融合 → 0x0305 上报",
        "operator": "",
        "commit": _git("rev-parse", "--short", "HEAD") or "unknown",
        "dirty_files": dirty_file_count(),
        "config_sha256": _fingerprint(),
        "evidence_grade": grade,
        "a_grade_requires_signoff": True,
        "verdict": verdict,
        "checks": checks,
        "devices": devices if devices is not None else detect_devices(),
        "artifacts": artifacts,
        "caveats": caveats,
        "reproduce": {"commands": commands or []},
    }


# ---------------------------------------------------------------------------
# 自测：合成输入跑通四条逻辑，并证明坏输入会被判 fail
# ---------------------------------------------------------------------------
def _synthetic_inputs(tmp: Path, *, bad: bool) -> dict:
    tmp.mkdir(parents=True, exist_ok=True)
    interval = 1.0 / 4.7
    base = 100.0
    lines = []
    for index in range(60):
        offset_cm = 90 if bad else 20
        lines.append(json.dumps({"t": base + index * interval,
                                 "slots": {"B1": [500 + offset_cm, 700]}}))
    if bad:
        tail = [json.dumps({"t": base + 20 * interval + 1.2 + i * interval,
                            "slots": {"B1": [540, 700]}}) for i in range(10)]
        lines = lines[:20] + tail
    radio = tmp / "radio.jsonl"
    radio.write_text("\n".join(lines) + "\n", encoding="utf-8")

    truth_path = tmp / "truth.csv"
    truth_path.write_text("t0,t1,slot,x_cm,y_cm\n0,1000,B1,500,700\n", encoding="utf-8")

    fusion = tmp / "fusion.jsonl"
    frames = []
    for index in range(80):
        t = base + index * 0.05
        source = "vision" if t >= 100.25 else "radio"
        shift = 200.0 if (bad and t > 100.3) else 0.0
        frames.append(json.dumps({
            "published_at": t, "captured_at": t,
            "robots": {"B1": {"position_cm": [1000.0 + shift + index * 0.1, 700.0],
                              "source": source, "state": "measured"}},
        }))
    fusion.write_text("\n".join(frames) + "\n", encoding="utf-8")

    camera = tmp / "camera.yaml"
    if bad:
        camera.write_text("camera:\n  name: unselected\n  reproj_rms_px: null\n", encoding="utf-8")
    else:
        camera.write_text("camera:\n  name: MV-CS060-10GC\n  reproj_rms_px: 0.62\n"
                          "  reproj_rms_max_px: 3.0\n"
                          "calibration:\n  serial: 'DA1234567'\n  date: '2026-12-01'\n",
                          encoding="utf-8")
    return {"radio_log": radio, "truth": load_truth(truth_path), "fusion_log": fusion,
            "switch_event": ("B1", 100.2), "camera": camera}


def _synthetic_devices() -> list[dict]:
    return [{"kind": "sdr", "model": "ADALM-PLUTO", "serial": "SYNTH", "transport": "usb", "present": True},
            {"kind": "camera", "model": "synthetic", "serial": "SYNTH", "transport": "usb3", "present": True},
            {"kind": "referee_serial", "model": "synthetic", "serial": "SYNTH", "transport": "usb", "present": True}]


def selftest() -> int:
    import tempfile

    results = {}
    with tempfile.TemporaryDirectory() as workdir:
        for label, bad, expected in (("good", False, PASS), ("bad", True, FAIL)):
            report = build_report(**_synthetic_inputs(Path(workdir) / label, bad=bad),
                                  evidence_grade="C", devices=_synthetic_devices(),
                                  commands=[f"selftest --case {label}"])
            results[label] = {"verdict": report["verdict"], "grade": report["evidence_grade"],
                              "checks": {item["id"]: item["status"] for item in report["checks"]}}
            if report["verdict"] != expected:
                print(json.dumps({"selftest": "failed", "case": label, "expected": expected,
                                  "got": results[label]}, ensure_ascii=False))
                return 1
        # 无设备基线：四项必须全 blocked，verdict 不能是 pass，等级不能高于 C
        report = build_report(radio_log=None, truth=None, fusion_log=None, switch_event=None,
                              camera=ROOT / "config" / "camera.yaml", evidence_grade="B",
                              devices=detect_devices(), commands=["selftest --no-device"])
        baseline = {"verdict": report["verdict"], "grade": report["evidence_grade"],
                    "checks": {item["id"]: item["status"] for item in report["checks"]}}
        if baseline["verdict"] != BLOCKED or any(status != BLOCKED for status in
                                                 baseline["checks"].values()):
            print(json.dumps({"selftest": "failed", "case": "no-device", "got": baseline},
                             ensure_ascii=False))
            return 1
        results["no_device"] = baseline
    print(json.dumps({"selftest": "ok", "cases": results}, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--radio-log", default="")
    parser.add_argument("--truth", default="")
    parser.add_argument("--fusion-log", default="")
    parser.add_argument("--switch-event", default="", help="slot=秒，例如 B1=12.3")
    parser.add_argument("--camera", default=str(ROOT / "config" / "camera.yaml"))
    parser.add_argument("--out", default=str(ROOT / "out" / "stage3_accept"))
    parser.add_argument("--evidence-grade", choices=("B", "C"), default="B",
                        help="A 级只能由真机现场人工签字，工具不发 A")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    switch_event = None
    if args.switch_event:
        slot, _, seconds = args.switch_event.partition("=")
        switch_event = (slot.strip(), float(seconds))

    report = build_report(
        radio_log=Path(args.radio_log) if args.radio_log else None,
        truth=load_truth(Path(args.truth)) if args.truth else None,
        fusion_log=Path(args.fusion_log) if args.fusion_log else None,
        switch_event=switch_event,
        camera=Path(args.camera) if args.camera else None,
        evidence_grade=args.evidence_grade,
        commands=[f"python tools/stage3_accept.py --radio-log {args.radio_log} "
                  f"--truth {args.truth} --fusion-log {args.fusion_log} "
                  f"--switch-event {args.switch_event}"],
    )

    out = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(path), "verdict": report["verdict"],
                      "evidence_grade": report["evidence_grade"],
                      "checks": {item["id"]: item["status"] for item in report["checks"]}},
                     ensure_ascii=False))
    return 0 if report["verdict"] == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
