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
    check_radio_uplink,
    check_vision_calibration,
    empty_flag_values,
    interpreter_hint,
    reproduce_command,
    selftest,
)


def write_radio(path, times, slot="B1", position=(500, 700)):
    path.write_text("\n".join(json.dumps({"t": t, "slots": {slot: list(position)}})
                              for t in times) + "\n", encoding="utf-8")
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

    # 1 秒断流 + 之后 2Hz → 频率与断流都超
    slow = write_radio(tmp_path / "slow.jsonl",
                       times[:20] + [times[19] + 1.0 + index * 0.5 for index in range(10)])
    result = check_radio_uplink(slow)
    assert result["status"] == FAIL
    assert "断流" in result["reason"]


def test_calibration_requires_all_four_items(tmp_path):
    partial = tmp_path / "camera.yaml"
    partial.write_text("camera:\n  name: MV-CS060-10GC\n  reproj_rms_px: 0.5\n", encoding="utf-8")
    result = check_vision_calibration(partial)
    assert result["status"] == BLOCKED
    assert "calibration.serial" in result["reason"] or "serial" in result["reason"]

    full = tmp_path / "full.yaml"
    full.write_text("camera:\n  name: MV-CS060-10GC\n  reproj_rms_px: 0.62\n"
                    "  reproj_rms_max_px: 3.0\n"
                    "calibration:\n  serial: 'DA1'\n  date: '2026-12-01'\n", encoding="utf-8")
    result = check_vision_calibration(full)
    assert result["status"] == PASS
    assert len(result["value"]["sha256"]) == 64


def test_pass_without_artifacts_is_downgraded_to_fail(tmp_path, monkeypatch):
    """拿不出证据 = 不通过，而不是 blocked。"""
    import tools.stage3_accept as module

    fake = {"status": PASS, "reason": "", "value": {"x": 1}, "threshold": "t",
            "artifacts": [], "grade": "B"}
    monkeypatch.setattr(module, "check_radio_uplink", lambda _log: dict(fake))
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
    camera = tmp_path / "over.yaml"
    camera.write_text("camera:\n  name: MV-CS060-10GC\n  reproj_rms_px: 5.2\n"
                      "  reproj_rms_max_px: 3.0\n"
                      "calibration:\n  serial: 'DA1'\n  date: '2026-12-01'\n", encoding="utf-8")
    result = check_vision_calibration(camera)
    assert result["status"] == FAIL
    assert "reproj_rms_px" in result["reason"]
    assert result["value"]["reproj_rms_px"] == 5.2
    assert result["value"]["limit_px"] == 3.0


def test_calibration_pass_is_caveated_as_config_only(tmp_path):
    """配置合格 ≠ 标定在真机上做过，这句必须出现在 caveats 里。"""
    camera = tmp_path / "full.yaml"
    camera.write_text("camera:\n  name: MV-CS060-10GC\n  reproj_rms_px: 0.62\n"
                      "  reproj_rms_max_px: 3.0\n"
                      "calibration:\n  serial: 'DA1'\n  date: '2026-12-01'\n", encoding="utf-8")
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
