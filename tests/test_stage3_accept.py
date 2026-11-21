"""阶段 3 验收工具的契约：三态、证据等级、以及"pass 必须能自证"。

规格在 `docs/experiments/STAGE3_REPORT_SCHEMA.md`。这里的测试盯着最容易退化的几条：
缺输入不许 pass、blocked 必须带原因、pass 必须带 value/threshold/artifacts+sha256、
等级取最低且工具永不发 A、checks 顺序固定。
"""

from __future__ import annotations

import json

import pytest

from tools.stage3_accept import (
    BLOCKED,
    CHECK_ORDER,
    FAIL,
    PASS,
    build_report,
    check_radio_accuracy,
    check_radio_uplink,
    check_vision_calibration,
    empty_flag_values,
    interpreter_hint,
    reproduce_command,
    selftest,
)


def write_radio(path, times, slot="B1", position=(500, 700), *, write_ok=True,
                session="s1"):
    path.write_text("\n".join(json.dumps({"session_id": session, "t": t,
                                          "write_ok": write_ok,
                                          "slots": {slot: list(position)}})
                              for t in times) + "\n", encoding="utf-8")
    return path


def write_calibrated_camera(path, *, rms=0.62, serial="DA1234567", date="2026-12-01",
                            name="MV-CS060-10GC"):
    path.write_text(f"camera:\n  name: {name}\n  serial: '{serial}'\n"
                    f"  calibrated_at: '{date}'\n"
                    f"transform:\n  reproj_rms_px: {rms}\n  reproj_rms_max_px: 3.0\n",
                    encoding="utf-8")
    return path


def write_truth(path, segments):
    rows = ["t0,t1,slot,x_cm,y_cm"]
    rows += [f"{t0},{t1},{slot},{x},{y}" for t0, t1, slot, x, y in segments]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_selftest_passes_its_own_expectations():
    assert selftest() == 0


def test_missing_inputs_are_blocked_never_pass(tmp_path):
    report = build_report(radio_log=None, truth=None, fusion_log=None, switch_event=None,
                          camera=tmp_path / "absent.yaml", evidence_grade="B",
                          devices=[], commands=[])
    assert report["verdict"] == BLOCKED
    assert all(item["status"] == BLOCKED for item in report["checks"])
    assert all(item["reason"].strip() for item in report["checks"])
    assert report["evidence_grade"] == "D"


def test_checks_are_reported_in_the_agreed_order(tmp_path):
    report = build_report(radio_log=None, truth=None, fusion_log=None, switch_event=None,
                          camera=None, evidence_grade="C", devices=[], commands=[])
    assert tuple(item["id"] for item in report["checks"]) == CHECK_ORDER


def test_tool_never_issues_grade_a(tmp_path):
    """A 级只能由真机现场人工签字，工具自己给不出来。"""
    report = build_report(radio_log=None, truth=None, fusion_log=None, switch_event=None,
                          camera=None, evidence_grade="B", devices=[], commands=[])
    assert report["evidence_grade"] in ("B", "C", "D")
    assert report["a_grade_requires_signoff"] is True


def test_uplink_metrics_and_failures(tmp_path):
    # 4.7Hz、60 帧 → 通过
    times = [100 + index / 4.7 for index in range(60)]
    log = write_radio(tmp_path / "ok.jsonl", times)
    result = check_radio_uplink(log)
    assert result["status"] == PASS
    assert 4.65 < result["value"]["effective_rate_hz"] < 4.75
    assert result["value"]["max_frames_1s"] <= 5
    # schema §2.1：通过时这三个字段必须自洽
    assert result["value"]["frames_sent"] == result["value"]["frames_written"]
    assert result["value"]["write_ok_ratio"] == 1.0

    # 1 秒断流 + 之后 2Hz → 频率与断流都超
    slow = write_radio(tmp_path / "slow.jsonl",
                       times[:20] + [times[19] + 1.0 + index * 0.5 for index in range(10)])
    result = check_radio_uplink(slow)
    assert result["status"] == FAIL
    assert "断流" in result["reason"]


def test_calibration_requires_all_four_items(tmp_path):
    partial = tmp_path / "camera.yaml"
    partial.write_text("camera:\n  name: MV-CS060-10GC\n"
                       "transform:\n  reproj_rms_px: 0.5\n", encoding="utf-8")
    result = check_vision_calibration(partial)
    assert result["status"] == BLOCKED
    assert "serial" in result["reason"]

    full = write_calibrated_camera(tmp_path / "full.yaml")
    result = check_vision_calibration(full)
    assert result["status"] == PASS
    assert len(result["value"]["sha256"]) == 64
    assert result["value"]["serial"] == "DA1234567"
    # schema §2.1：验收读的就是运行时那一套，且必须记下「加载成功」
    assert result["value"]["config_loads"] is True
    assert result["value"]["config_load_error"] == ""


def test_pass_without_artifacts_is_downgraded_to_fail(tmp_path, monkeypatch):
    """拿不出证据 = 不通过，而不是 blocked。"""
    import tools.stage3_accept as module

    fake = {"status": PASS, "reason": "", "value": {"x": 1}, "threshold": "t",
            "artifacts": [], "grade": "B"}
    monkeypatch.setattr(module, "check_radio_uplink", lambda _log, **_kw: dict(fake))
    report = build_report(radio_log=None, truth=None, fusion_log=None, switch_event=None,
                          camera=None, evidence_grade="B", devices=[], commands=[])
    uplink = next(item for item in report["checks"] if item["id"] == "radio_uplink")
    assert uplink["status"] == FAIL
    assert "拿不出完整证据" in uplink["reason"]


def test_caveats_are_never_empty(tmp_path):
    report = build_report(radio_log=None, truth=None, fusion_log=None, switch_event=None,
                          camera=None, evidence_grade="C", devices=[], commands=[])
    assert report["caveats"]
    assert any("不证明检测精度" in text for text in report["caveats"])


def test_report_carries_traceable_fingerprints(tmp_path):
    report = build_report(radio_log=None, truth=None, fusion_log=None, switch_event=None,
                          camera=None, evidence_grade="C", devices=[], commands=["x"])
    assert report["schema"] == "stage3-accept/1"
    assert report["commit"]
    assert "default.yaml" in report["config_sha256"]
    assert len(report["config_sha256"]["default.yaml"]) == 64
    assert isinstance(report["dirty_files"], int)
    assert report["reproduce"]["commands"] == ["x"]


@pytest.mark.parametrize("times_ok", [True, False])
def test_verdict_aggregation(tmp_path, times_ok):
    times = [100 + index / 4.7 for index in range(60)]
    if not times_ok:
        times = times[:20] + [times[19] + 1.5 + index * 0.5 for index in range(10)]
    log = write_radio(tmp_path / (("ok" if times_ok else "bad") + ".jsonl"), times)
    report = build_report(radio_log=log, truth=None, fusion_log=None, switch_event=None,
                          camera=None, evidence_grade="B", devices=[], commands=[])
    # 没有真值 → radio_accuracy blocked；uplink 通过也可能是 fail
    statuses = {item["id"]: item["status"] for item in report["checks"]}
    if statuses["radio_uplink"] == FAIL:
        assert report["verdict"] == FAIL
    else:
        assert report["verdict"] == BLOCKED


# ---------------------------------------------------------------------------
# 2026-11-21 追加：标定 fail 路径、复现命令不留空 flag、operator 不许是空串
# ---------------------------------------------------------------------------
def test_calibration_fail_when_rms_over_limit(tmp_path):
    """四项齐全但实测 rms 超限 → fail（既不是 blocked，更不是 pass）。"""
    camera = write_calibrated_camera(tmp_path / "over.yaml", rms=5.2)
    result = check_vision_calibration(camera)
    assert result["status"] == FAIL
    assert "reproj_rms_px" in result["reason"]
    assert result["value"]["reproj_rms_px"] == 5.2
    assert result["value"]["limit_px"] == 3.0


def test_calibration_rejects_a_yaml_the_runtime_would_refuse(tmp_path):
    """故障注入：验收器编一套 `calibration:` 节点，主程序加载会直接报错。

    这正是 `d06776c` 的真 bug——验收 pass、运行时 `ConfigError`。验收器必须
    先过运行时 schema，任何运行时拒绝的文件都不能判 pass。
    """
    camera = tmp_path / "phantom.yaml"
    camera.write_text("camera:\n  name: MV-CS060-10GC\n  reproj_rms_px: 0.62\n"
                      "  reproj_rms_max_px: 3.0\n"
                      "calibration:\n  serial: 'DA1'\n  date: '2026-12-01'\n", encoding="utf-8")
    result = check_vision_calibration(camera)
    assert result["status"] == FAIL
    assert "运行时加载不了" in result["reason"]
    assert "calibration" in result["reason"]
    assert result["value"]["config_loads"] is False
    assert result["value"]["config_load_error"]


def test_calibration_pass_is_caveated_as_config_only(tmp_path):
    """配置合格 ≠ 标定在真机上做过，这句必须出现在 caveats 里。"""
    camera = write_calibrated_camera(tmp_path / "full.yaml")
    report = build_report(radio_log=None, truth=None, fusion_log=None, switch_event=None,
                          camera=camera, evidence_grade="B", devices=[], commands=[])
    calibration = next(item for item in report["checks"] if item["id"] == "vision_calibration")
    assert calibration["status"] == PASS
    assert any("不等于标定确实在真机上做过" in text for text in report["caveats"])


def test_reproduce_command_never_leaves_empty_flags():
    assert reproduce_command() == f"{interpreter_hint()} tools/stage3_accept.py"
    assert "python" in interpreter_hint()  # 本机 `python` 不在 PATH 上，不能写死它
    assert empty_flag_values(reproduce_command()) == []

    partial = reproduce_command(radio_log="out/live.jsonl", switch_event="B1=12.3")
    assert empty_flag_values(partial) == []
    assert partial == (f"{interpreter_hint()} tools/stage3_accept.py "
                       "--radio-log out/live.jsonl --switch-event B1=12.3")
    # 没给的参数、以及等于默认值的参数都不许出现
    assert "out/fused.jsonl" not in partial
    assert "--camera" not in partial and "--out" not in partial


def test_empty_flag_detector_catches_the_old_bug():
    """回归：`d06776c` 生成的复现命令就长这样，现场照抄会当场报错。"""
    old = ("python tools/stage3_accept.py --radio-log  --truth  "
           "--fusion-log  --switch-event ")
    assert empty_flag_values(old) == ["--radio-log", "--truth", "--fusion-log", "--switch-event"]


def test_operator_is_never_empty():
    kwargs = dict(radio_log=None, truth=None, fusion_log=None, switch_event=None,
                  camera=None, evidence_grade="C", devices=[], commands=[])
    explicit = build_report(operator="张三", **kwargs)
    assert explicit["operator"] == "张三"
    assert explicit["operator_source"] == "arg"
    # 不给 --operator 就退到 git config user.name；本机一定有值，但绝不允许空串
    fallback = build_report(**kwargs)
    assert fallback["operator"].strip()
    assert fallback["operator"] != ""


def test_operator_unset_is_flagged_in_caveats(monkeypatch):
    import tools.stage3_accept as module

    monkeypatch.setattr(module, "_git", lambda *args: "")
    report = build_report(radio_log=None, truth=None, fusion_log=None, switch_event=None,
                          camera=None, evidence_grade="C", devices=[], commands=[])
    assert report["operator"] == "unset"
    assert report["operator_source"] == "unset"
    assert any("operator 未填" in text for text in report["caveats"])


# ---------------------------------------------------------------------------
# 故障注入：链路坏掉时验收器必须判错，而不是判过
# ---------------------------------------------------------------------------
def test_uplink_rejects_a_log_that_never_wrote(tmp_path):
    """串口全失败：组帧 48 次、写成功 0 次——旧版会判 pass。"""
    log = write_radio(tmp_path / "dead.jsonl", [100 + index / 4.7 for index in range(48)],
                      write_ok=False)
    result = check_radio_uplink(log)
    assert result["status"] == FAIL
    assert result["value"]["frames_composed"] == 48
    assert result["value"]["frames_written"] == 0
    assert "写成功 0 次" in result["reason"]


def test_uplink_fails_when_the_log_has_no_write_ok_field(tmp_path):
    """旧格式日志只有时间戳：证明不了写进串口，缺证据按 fail。"""
    log = tmp_path / "old.jsonl"
    log.write_text("\n".join(json.dumps({"session_id": "s1", "t": 100 + index / 4.7})
                             for index in range(48)) + "\n", encoding="utf-8")
    result = check_radio_uplink(log)
    assert result["status"] == FAIL
    assert "write_ok" in result["reason"]


def test_uplink_only_counts_frames_that_were_really_written(tmp_path):
    """一半写失败：频率要按写成功的那些算，不能拿组帧数充数。"""
    times = [100 + index / 4.7 for index in range(48)]
    log = tmp_path / "half.jsonl"
    rows = [json.dumps({"session_id": "s1", "t": t, "write_ok": index < 24,
                        "slots": {"B1": [500, 700]}}) for index, t in enumerate(times)]
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")
    result = check_radio_uplink(log)
    assert result["value"]["frames_composed"] == 48
    assert result["value"]["frames_written"] == 24
    # 频率按 24 个写成功的帧算（4.69Hz），拿 48 个组帧数充数就会显示 4.7+
    assert result["status"] == FAIL
    assert "频率" in result["reason"]


def test_accuracy_blocks_on_too_few_samples(tmp_path):
    """1 个样本 100% 也不算数：样本不足时百分比没有意义。"""
    log = write_radio(tmp_path / "one.jsonl", [100.0])
    truth = [{"t0": 0.0, "t1": 400.0, "slot": "B1", "x_cm": 500.0, "y_cm": 700.0}]
    result = check_radio_accuracy(log, truth)
    assert result["status"] == BLOCKED
    assert "少于" in result["reason"]


def test_accuracy_blocks_when_a_truth_segment_has_no_sample(tmp_path):
    """覆盖率：每个真值段至少一个采样点，缺一段就 blocked。"""
    times = [100 + index / 4.7 for index in range(48)]
    log = write_radio(tmp_path / "enough.jsonl", times)
    truth = [{"t0": 0.0, "t1": 400.0, "slot": "B1", "x_cm": 500.0, "y_cm": 700.0}]
    assert check_radio_accuracy(log, truth)["status"] == PASS

    truth.append({"t0": 500.0, "t1": 600.0, "slot": "B1", "x_cm": 500.0, "y_cm": 700.0})
    result = check_radio_accuracy(log, truth)
    assert result["status"] == BLOCKED
    assert "真值段" in result["reason"]


def test_accuracy_ignores_frames_that_never_wrote(tmp_path):
    """没写出去的上报不能算进分母，也不能凑样本数。"""
    log = write_radio(tmp_path / "dead.jsonl", [100 + index / 4.7 for index in range(48)],
                      write_ok=False)
    truth = [{"t0": 0.0, "t1": 400.0, "slot": "B1", "x_cm": 500.0, "y_cm": 700.0}]
    result = check_radio_accuracy(log, truth)
    assert result["status"] == BLOCKED
    assert result["value"]["samples"] == 0


# ---------------------------------------------------------------------------
# session_id：日志是 append 的，两次跑写同一个文件必须能分开
# ---------------------------------------------------------------------------
def test_mixed_sessions_are_refused_until_one_is_named(tmp_path):
    """现场真踩过的坑：33 帧上线路，日志却有 66 行（两次跑接在一起）。"""
    times = [100 + index / 4.7 for index in range(48)]
    log = tmp_path / "mixed.jsonl"
    rows = []
    for session in ("run_a", "run_b"):
        rows += [json.dumps({"session_id": session, "t": t, "write_ok": True,
                             "slots": {"B1": [500, 700]}}) for t in times]
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")

    result = check_radio_uplink(log)
    assert result["status"] == FAIL
    assert "混了 2 个 session" in result["reason"]
    assert "run_a" in result["reason"] and "run_b" in result["reason"]

    # 指定 session 后只算那一次：48 帧、约 4.7Hz，正常通过
    result = check_radio_uplink(log, session="run_a")
    assert result["status"] == PASS
    assert result["value"]["frames_composed"] == 48
    assert result["value"]["frames_sent"] == 48


def test_rows_without_session_id_are_fail(tmp_path):
    log = tmp_path / "nosession.jsonl"
    log.write_text("\n".join(json.dumps({"t": 100 + index / 4.7, "write_ok": True})
                             for index in range(48)) + "\n", encoding="utf-8")
    result = check_radio_uplink(log)
    assert result["status"] == FAIL
    assert "session_id" in result["reason"]


def test_unknown_session_name_is_fail_not_empty_pass(tmp_path):
    log = write_radio(tmp_path / "one.jsonl", [100 + index / 4.7 for index in range(48)],
                      session="run_a")
    result = check_radio_uplink(log, session="run_z")
    assert result["status"] == FAIL
    assert "run_z" in result["reason"]
