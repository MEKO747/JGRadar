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
        --fusion-log out/fused.jsonl --switch-event B1=12.3 --out out/stage3_accept \\
        --operator 张三
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
#: 定位准确率的最少有效样本数。样本太少时 1/1 = 100% 也能过，那是假象：
#: 真值窗口里只报一个点就判 pass，等于把「没测」当成「测得好」。
ACCURACY_MIN_SAMPLES = 30
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


def resolve_operator(explicit: str | None = None) -> tuple[str, bool]:
    """``--operator`` → ``git config user.name`` → ``"unset"``，返回 ``(名字, 是否未填)``。

    空串比 ``"unset"`` 更糟——它看着像填了，所以拿不到就显式写 ``unset`` 并进 caveats。
    """
    for candidate in (explicit, _git("config", "user.name")):
        name = str(candidate or "").strip()
        if name:
            return name, False
    return "unset", True


def interpreter_hint() -> str:
    """复现命令里的解释器：用真跑起来的那个。

    本机 ``python`` 不在 PATH 上，写死 ``python`` 的复现命令照抄会 command not found；
    在仓库内就用相对路径（``.venv/bin/python``），换机器退到绝对路径。
    """
    try:
        return str(Path(sys.executable).relative_to(ROOT))
    except ValueError:
        return sys.executable


def reproduce_command(*, radio_log="", truth="", fusion_log="", switch_event="",
                      camera="", out="", evidence_grade="", operator="") -> str:
    """只拼实际生效的参数：空值与默认值一律省略。

    旧版把空 flag 也拼进去（``--radio-log  --truth  ...``），现场照抄当场报错；
    命令能原样复制运行，才算复现包。
    """
    defaults = {"--camera": str(ROOT / "config" / "camera.yaml"),
                "--out": str(ROOT / "out" / "stage3_accept"),
                "--evidence-grade": "B"}
    parts = [interpreter_hint(), "tools/stage3_accept.py"]
    for flag, value in (("--radio-log", radio_log), ("--truth", truth),
                        ("--fusion-log", fusion_log), ("--switch-event", switch_event),
                        ("--camera", camera), ("--out", out),
                        ("--evidence-grade", evidence_grade), ("--operator", operator)):
        text = str(value or "").strip()
        if not text or text == defaults.get(flag):
            continue
        parts.append(f"{flag} {text}")
    return " ".join(parts)


def empty_flag_values(command: str) -> list[str]:
    """列出「后面没跟值」的 flag：``--radio-log  --truth`` 里的 ``--radio-log`` 就是空值。"""
    tokens = command.split()
    return [token for index, token in enumerate(tokens)
            if token.startswith("--") and (index + 1 >= len(tokens)
                                           or tokens[index + 1].startswith("--"))]


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


def _load_log(path: Path) -> list[dict]:
    """读 JSONL。坏行直接抛，不静默跳过——验收器不能替人补数据。"""
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number} 不是合法 JSON：{exc}") from exc
    return rows


def _missing_write_ok(rows: list[dict]) -> int:
    """数出没有 ``write_ok`` 字段的行。

    旧日志只有组帧时间戳，无法区分「组了帧」和「真的写进串口」。按
    「缺证据 = 不通过」处理，不能当 pass。
    """
    return sum(1 for row in rows if "write_ok" not in row)


def check_radio_accuracy(radio_log: Path | None, truth: list[dict] | None) -> dict:
    threshold = (f"d<{ACCURATE_M}m 比例 >= {ACCURATE_RATIO_MIN:.0%}，"
                 f"有效上报 >= {ACCURACY_MIN_SAMPLES} 个且每个真值段至少 1 个")
    if radio_log is None or not radio_log.is_file():
        return {"status": BLOCKED, "reason": "缺少 --radio-log", "value": None,
                "threshold": threshold, "artifacts": [], "grade": "D"}
    if not truth:
        return {"status": BLOCKED, "reason": "缺少 --truth：没有真值就算不出 d",
                "value": None, "threshold": threshold, "artifacts": [], "grade": "D"}

    rows = _load_log(radio_log)
    if _missing_write_ok(rows):
        return {"status": FAIL,
                "reason": f"{_missing_write_ok(rows)}/{len(rows)} 行日志缺 write_ok："
                          "无法证明这些坐标真的发给了服务器",
                "value": {"rows": len(rows)}, "threshold": threshold,
                "artifacts": [], "grade": "B"}

    distances = []
    per_segment: dict[int, int] = {}
    for row in rows:
        if row.get("write_ok") is not True:
            continue
        t = float(row["t"])
        for slot, position in (row.get("slots") or {}).items():
            x_cm, y_cm = float(position[0]), float(position[1])
            if x_cm == 0.0 and y_cm == 0.0:
                continue  # (0,0) = 未发送，不构成判定
            for index, segment in enumerate(truth):
                if segment["slot"] == slot and segment["t0"] <= t < segment["t1"]:
                    distances.append(math.hypot(x_cm - segment["x_cm"], y_cm - segment["y_cm"]))
                    per_segment[index] = per_segment.get(index, 0) + 1
                    break

    uncovered = [segment for index, segment in enumerate(truth) if per_segment.get(index, 0) == 0]
    if not distances or uncovered:
        reason = (f"{len(uncovered)}/{len(truth)} 个真值段一个采样点都没有"
                  if uncovered else "真值时间段内没有任何有效上报")
        return {"status": BLOCKED, "reason": reason,
                "value": {"samples": len(distances), "truth_segments": len(truth),
                          "segments_covered": len(truth) - len(uncovered)},
                "threshold": threshold, "artifacts": [], "grade": "B"}
    if len(distances) < ACCURACY_MIN_SAMPLES:
        return {"status": BLOCKED,
                "reason": f"有效上报只有 {len(distances)} 个，少于 {ACCURACY_MIN_SAMPLES} 个："
                          "样本不足时百分比没有意义",
                "value": {"samples": len(distances), "truth_segments": len(truth),
                          "segments_covered": len(truth) - len(uncovered)},
                "threshold": threshold, "artifacts": [], "grade": "B"}

    accurate = sum(1 for d in distances if d < ACCURATE_M * 100)
    ratio = accurate / len(distances)
    status = PASS if ratio >= ACCURATE_RATIO_MIN else FAIL
    return {
        "status": status, "reason": "" if status == PASS else f"准确率 {ratio:.1%} 低于阈值",
        "value": {"samples": len(distances), "accurate_ratio": round(ratio, 4),
                  "truth_segments": len(truth), "segments_covered": len(truth) - len(uncovered),
                  "median_cm": round(sorted(distances)[len(distances) // 2], 2),
                  "max_cm": round(max(distances), 2)},
        "threshold": threshold, "artifacts": [radio_log], "grade": "B",
    }


# ---------------------------------------------------------------------------
# 2) 无线电上报节拍
# ---------------------------------------------------------------------------
def check_radio_uplink(radio_log: Path | None) -> dict:
    threshold = (f">={UPLINK_RATE_MIN_HZ}Hz、窗口<={UPLINK_WINDOW_MAX_FRAMES}、"
                 f"gap<={UPLINK_MAX_GAP_S}s（只认 write_ok:true）")
    if radio_log is None or not radio_log.is_file():
        return {"status": BLOCKED, "reason": "缺少 --radio-log", "value": None,
                "threshold": threshold, "artifacts": [], "grade": "D"}
    rows = _load_log(radio_log)
    if _missing_write_ok(rows):
        return {"status": FAIL,
                "reason": f"{_missing_write_ok(rows)}/{len(rows)} 行缺 write_ok："
                          "只有组帧时间戳证明不了真的写进了串口",
                "value": {"rows": len(rows)}, "threshold": threshold,
                "artifacts": [], "grade": "B"}

    written = sorted(float(row["t"]) for row in rows if row.get("write_ok") is True)
    failed = len(rows) - len(written)
    value = {"frames_composed": len(rows), "frames_written": len(written),
             "frames_failed": failed,
             "write_ok_ratio": round(len(written) / len(rows), 4) if rows else None}
    if not written:
        return {"status": FAIL,
                "reason": f"组帧 {len(rows)} 次、写成功 0 次：串口链路没通，不算发送",
                "value": value, "threshold": threshold, "artifacts": [], "grade": "B"}
    if len(written) < 2:
        return {"status": BLOCKED, "reason": "写成功的帧数不足", "value": value,
                "threshold": threshold, "artifacts": [], "grade": "B"}

    span = written[-1] - written[0]
    rate = (len(written) - 1) / span if span > 0 else 0.0
    max_gap = max(b - a for a, b in zip(written, written[1:]))
    window_max, left = 0, 0
    for right, t in enumerate(written):
        while t - written[left] >= 1.0:
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
            "value": {**value, "effective_rate_hz": round(rate, 3),
                      "max_frames_1s": window_max, "max_gap_s": round(max_gap, 3)},
            "threshold": threshold, "artifacts": [radio_log], "grade": "B"}


# ---------------------------------------------------------------------------
# 3) 视觉标定四要件
# ---------------------------------------------------------------------------
def check_vision_calibration(camera_path: Path | None) -> dict:
    """标定四要件，**全部取运行时接受的字段**。

    这一步以前读 ``camera.reproj_rms_px`` 和顶层 ``calibration.serial``——
    那两个键运行时根本不接受（``_build`` 遇未知键直接抛 ``ConfigError``）。
    于是「验收 pass、主程序起不来」。现在先拿 ``RadarConfig.from_dict``
    跑一遍，验收器和运行时用同一份 schema，不再各说各话。
    """
    threshold = "序列号 + 重标日期 + reproj_rms_px 实测 + 文件 sha256（运行时字段）"
    if camera_path is None or not camera_path.is_file():
        return {"status": BLOCKED, "reason": "缺少相机配置", "value": None,
                "threshold": threshold, "artifacts": [], "grade": "D"}
    import yaml

    from jg_radar.config.schema import ConfigError, RadarConfig

    data = yaml.safe_load(camera_path.read_text(encoding="utf-8")) or {}
    digest = _sha256(camera_path)
    try:
        RadarConfig.from_dict(data)
    except ConfigError as exc:
        return {"status": FAIL,
                "reason": f"这份 YAML 运行时加载不了（验收不能跑在运行时拒绝的配置上）：{exc}",
                "value": {"config": str(camera_path), "sha256": digest},
                "threshold": threshold, "artifacts": [], "grade": "D"}

    camera = data.get("camera") or {}
    transform = data.get("transform") or {}
    name = str(camera.get("name", "") or "")
    serial = str(camera.get("serial", "") or "").strip()
    calibrated_at = str(camera.get("calibrated_at", "") or "").strip()
    rms = transform.get("reproj_rms_px")
    limit = float(transform.get("reproj_rms_max_px", 3.0) or 3.0)

    missing = []
    if not name or name.lower() in ("unselected", "placeholder"):
        missing.append("相机型号（camera.name 仍是占位）")
    if not serial:
        missing.append("相机序列号（camera.serial）")
    if not calibrated_at:
        missing.append("重标日期（camera.calibrated_at）")
    if rms is None:
        missing.append("reproj_rms_px 实测值（transform.reproj_rms_px）")

    value = {"config": str(camera_path), "sha256": digest, "camera_name": name,
             "serial": serial, "calibrated_at": calibrated_at,
             "reproj_rms_px": rms, "limit_px": limit}
    if missing:
        return {"status": BLOCKED, "reason": "标定证据不全：" + "、".join(missing),
                "value": value, "threshold": threshold, "artifacts": [], "grade": "D"}
    rms = float(rms)
    status = PASS if rms <= limit else FAIL
    return {"status": status,
            "reason": "" if status == PASS else f"reproj_rms_px {rms} > 上限 {limit}",
            "value": value, "threshold": threshold,
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
                 devices: list[dict] | None = None, commands: list[str] | None = None,
                 operator: str | None = None) -> dict:
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
    caveats.append("write_ok 只证明数据写进了串口，不证明裁判服务器收到——"
                   "第三态在本地不可观测，只能由裁判端/小地图确认。")
    operator_name, operator_missing = resolve_operator(operator)
    if operator_missing:
        caveats.append("operator 未填：--operator 与 git config user.name 都没拿到，"
                       "报告里写 unset —— 这次运行没有可追溯的操作者。")
    calibration = next(item for item in checks if item["id"] == "vision_calibration")
    if calibration["status"] != BLOCKED:
        caveats.append("标定配置四项齐全且 rms 在限内，不等于标定确实在真机上做过；"
                       "A 级签核补这一层。")
    blocked_ids = [item["id"] for item in checks if item["status"] == BLOCKED]
    if blocked_ids:
        caveats.append("以下项无证据，verdict 因此不是 pass：" + "、".join(blocked_ids))
    if any(item["status"] == FAIL for item in checks):
        caveats.append("存在 fail 项：先修现场问题再重跑，不要用离线数据顶替。")

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "goal": "JGRadar 2027 雷达站：无线电解调 → 多源融合 → 0x0305 上报",
        "operator": operator_name,
        "operator_source": "unset" if operator_missing else (
            "arg" if str(operator or "").strip() else "git"),
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
                                 "write_ok": True,
                                 "slots": {"B1": [500 + offset_cm, 700]}}))
    if bad:
        tail = [json.dumps({"t": base + 20 * interval + 1.2 + i * interval,
                            "write_ok": True,
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
        camera.write_text("camera:\n  name: unselected\n"
                          "transform:\n  reproj_rms_px: null\n", encoding="utf-8")
    else:
        camera.write_text("camera:\n  name: MV-CS060-10GC\n  serial: 'DA1234567'\n"
                          "  calibrated_at: '2026-12-01'\n"
                          "transform:\n  reproj_rms_px: 0.62\n  reproj_rms_max_px: 3.0\n",
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
                              "operator": report["operator"],
                              "checks": {item["id"]: item["status"] for item in report["checks"]}}
            if not str(report["operator"]).strip():
                print(json.dumps({"selftest": "failed", "case": f"{label}:empty-operator",
                                  "got": results[label]}, ensure_ascii=False))
                return 1
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
    # 复现命令里不许留空 flag：现场照抄 ``--radio-log  --truth`` 会当场报错
    old_bug = ("python tools/stage3_accept.py --radio-log  --truth  "
               "--fusion-log  --switch-event ")
    for label, command in (("bare", reproduce_command()),
                           ("partial", reproduce_command(radio_log="out/live.jsonl",
                                                         switch_event="B1=12.3")),
                           ("old_bug", old_bug)):
        empty = empty_flag_values(command)
        results[f"command_{label}"] = {"command": command, "empty_flags": empty}
        if bool(empty) != (label == "old_bug"):
            print(json.dumps({"selftest": "failed", "case": f"command_{label}",
                              "got": results[f"command_{label}"]}, ensure_ascii=False))
            return 1
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
    parser.add_argument("--operator", default="",
                        help="谁跑的这次验收；缺省退到 git config user.name，都拿不到写 unset")
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
        operator=args.operator,
        commands=[reproduce_command(radio_log=args.radio_log, truth=args.truth,
                                    fusion_log=args.fusion_log,
                                    switch_event=args.switch_event, camera=args.camera,
                                    out=args.out, evidence_grade=args.evidence_grade,
                                    operator=args.operator)],
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
