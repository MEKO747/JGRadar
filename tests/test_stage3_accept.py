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
