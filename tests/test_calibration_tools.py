"""标定工具链的端到端测试 —— M2 的验收门。

走的是一条真实路径，只是把"人拿尺子量场地、用鼠标点像素"换成了合成数据：

    关键点 JSON ──calibrate_extrinsics.py──▶ extrinsics.yaml ──▶ load_config
                        │
                        └──validate_alignment.py──▶ 反投影 RMS / 往返 / 退出码

**退出码是重点。** 门限的价值全在"不过就拦住"上；一个永远返回 0 的体检工具
比没有还糟，因为它会让人以为已经查过了。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from jg_radar.config import load_config

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
KEYPOINTS = FIXTURES / "keypoints_synth.json"
DEFAULT_CFG = ROOT / "config" / "default.yaml"


def run_tool(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "tools" / name), *args],
        capture_output=True, text=True, cwd=ROOT,
    )


@pytest.fixture(scope="module")
def fixture_doc() -> dict:
    return json.loads(KEYPOINTS.read_text(encoding="utf-8"))


@pytest.fixture
def workdir(tmp_path: Path, fixture_doc: dict) -> Path:
    """把 fixture 的相机写成一份**未标定**的配置（transform 为单位阵）。"""
    cam = fixture_doc["camera"]
    text = (
        "camera:\n"
        f"  name: synthetic\n"
        f"  width: {cam['width']}\n"
        f"  height: {cam['height']}\n"
        "  K:\n"
        + "".join("    - [" + ", ".join(repr(float(v)) for v in row) + "]\n"
                  for row in cam["K"])
        + "  dist: [" + ", ".join(repr(float(v)) for v in cam["dist"]) + "]\n"
        "transform:\n"
        "  R:\n"
        "    - [1.0, 0.0, 0.0]\n"
        "    - [0.0, 1.0, 0.0]\n"
        "    - [0.0, 0.0, 1.0]\n"
        "  t: [0.0, 0.0, 0.0]\n"
        "field:\n"
        "  ground_plane_z: 0.0\n"
    )
    (tmp_path / "camera_synth.yaml").write_text(text, encoding="utf-8")
    return tmp_path


# ----------------------------------------------------------------------
# 标定
# ----------------------------------------------------------------------
def test_calibrate_writes_a_loadable_config(workdir: Path, fixture_doc: dict):
    out = workdir / "extrinsics.yaml"
    proc = run_tool("calibrate_extrinsics.py",
                    "--keypoints", str(KEYPOINTS),
                    "--camera", str(workdir / "camera_synth.yaml"),
                    "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    assert "✓ 已写出" in proc.stdout

    # 严格加载器能读回来 —— 说明写出的 YAML 没有重复键/未知键
    cfg = load_config(DEFAULT_CFG, workdir / "camera_synth.yaml", out)
    assert cfg.transform.reproj_rms_px is not None
    assert cfg.transform.reproj_rms_px < 1e-3

    # 解出来的位姿必须等于生成 fixture 时用的位姿
    from jg_radar.transform import look_at

    pose = fixture_doc["pose"]
    R_true, t_true = look_at(pose["eye"], pose["target"])
    assert np.allclose(np.asarray(cfg.transform.R), R_true, atol=1e-5)
    assert np.allclose(np.asarray(cfg.transform.t), t_true, atol=0.05)


def test_calibrate_tolerates_a_couple_of_bad_keypoints(workdir: Path, fixture_doc: dict):
    """10 个点里点错 2 个，RANSAC 应当照样解出正确位姿。

    这条守的是一个很容易踩空的坑：RANSAC 的最小样本若取成 8 而不是 6，
    "抽到全内点子集"的概率就是 C(8,8)/C(10,8)=0 —— 它会在最该发挥作用的
    场合必然失败，而且失败得很安静（返回 None，看起来像"数据不行"）。
    """
    doc = json.loads(KEYPOINTS.read_text(encoding="utf-8"))
    for p in doc["points"][:2]:
        p["pixel"] = [p["pixel"][0] + 600.0, p["pixel"][1] - 400.0]
    bad_kps = workdir / "two_bad.json"
    bad_kps.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    from jg_radar.transform import look_at

    pose = fixture_doc["pose"]
    R_true, t_true = look_at(pose["eye"], pose["target"])

    out = workdir / "tolerant.yaml"
    proc = run_tool("calibrate_extrinsics.py",
                    "--keypoints", str(bad_kps),
                    "--camera", str(workdir / "camera_synth.yaml"),
                    "--out", str(out), "--force")
    assert proc.returncode == 0, proc.stderr
    cfg = load_config(DEFAULT_CFG, workdir / "camera_synth.yaml", out)
    assert np.allclose(np.asarray(cfg.transform.R), R_true, atol=1e-4)
    assert np.allclose(np.asarray(cfg.transform.t), t_true, atol=0.5)


def test_calibrate_refuses_to_write_when_residuals_fail(workdir: Path, fixture_doc: dict):
    """点错了 → 宁可不写，也不要一份系统性偏了的 R|t 悄悄进到配置里。

    注意卡住它的必须是**残差门**，不是"PnP 解不出来"：这里只有 2 个点错，
    位姿照样解得出来（见上一个用例），但体检报告里的最大反投影误差是几
    百像素，远过不了 8px 的门限。
    """
    doc = json.loads(KEYPOINTS.read_text(encoding="utf-8"))
    for p in doc["points"][:2]:
        p["pixel"] = [p["pixel"][0] + 600.0, p["pixel"][1] - 400.0]
    bad_kps = workdir / "bad_keypoints.json"
    bad_kps.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    out = workdir / "should_not_exist.yaml"
    proc = run_tool("calibrate_extrinsics.py",
                    "--keypoints", str(bad_kps),
                    "--camera", str(workdir / "camera_synth.yaml"),
                    "--out", str(out))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert not out.exists(), "残差不过门限却写出了配置"
    assert "残差未过门限" in proc.stderr
    assert "反投影" in proc.stderr


def test_calibrate_needs_six_points(workdir: Path):
    doc = json.loads(KEYPOINTS.read_text(encoding="utf-8"))
    doc["points"] = doc["points"][:4]
    few = workdir / "few.json"
    few.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    proc = run_tool("calibrate_extrinsics.py",
                    "--keypoints", str(few),
                    "--camera", str(workdir / "camera_synth.yaml"),
                    "--out", str(workdir / "nope.yaml"))
    assert proc.returncode == 2
    assert "至少需要 6 个" in proc.stderr or "PnP 至少需要 6" in proc.stderr


# ----------------------------------------------------------------------
# 体检
# ----------------------------------------------------------------------
def test_validate_alignment_passes_on_ground_truth(workdir: Path, fixture_doc: dict):
    """方案 §4.2 的验收条款：反投影 RMS < 3px、往返 < 2cm。"""
    out = workdir / "extrinsics.yaml"
    assert run_tool("calibrate_extrinsics.py",
                    "--keypoints", str(KEYPOINTS),
                    "--camera", str(workdir / "camera_synth.yaml"),
                    "--out", str(out)).returncode == 0

    report_json = workdir / "report.json"
    proc = run_tool("validate_alignment.py",
                    "--config", str(DEFAULT_CFG),
                    "--camera", str(workdir / "camera_synth.yaml"),
                    "--field", str(out),
                    "--keypoints", str(KEYPOINTS),
                    "--json", str(report_json))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "通过" in proc.stdout

    rep = json.loads(report_json.read_text(encoding="utf-8"))
    assert rep["rms_px"] < 3.0
    assert rep["max_roundtrip_cm"] < 2.0
    assert rep["num_calibratable"] == 10


def test_validate_alignment_exits_nonzero_on_a_bad_pose(workdir: Path, fixture_doc: dict):
    """门限必须真的会拦人：把光心挪 50cm，体检工具应以非零码退出。"""
    from jg_radar.transform import look_at

    pose = fixture_doc["pose"]
    eye = np.asarray(pose["eye"], dtype=np.float64)
    eye[0] += 50.0
    R, t = look_at(eye, pose["target"])

    bad = workdir / "bad_extrinsics.yaml"
    bad.write_text(
        "transform:\n"
        "  R:\n"
        + "".join("    - [" + ", ".join(repr(float(v)) for v in row) + "]\n" for row in R)
        + "  t: [" + ", ".join(repr(float(v)) for v in t) + "]\n",
        encoding="utf-8",
    )

    proc = run_tool("validate_alignment.py",
                    "--config", str(DEFAULT_CFG),
                    "--camera", str(workdir / "camera_synth.yaml"),
                    "--field", str(bad),
                    "--keypoints", str(KEYPOINTS))
    assert proc.returncode == 1, proc.stdout
    assert "未通过" in proc.stdout


def test_validate_alignment_suggest_mode():
    """陌生 PLY 的第一次接触：只给轴向与建议变换，不需要关键点。"""
    proc = run_tool("validate_alignment.py", "--mesh", str(FIXTURES / "field_mesh_simple.ply"))
    assert proc.returncode == 0, proc.stderr
    assert "推测朝上轴：z" in proc.stdout
    assert "建议 mesh_to_field" in proc.stdout


def test_suggest_reports_the_flattest_axis_as_a_second_opinion():
    """建议模式要同时给出两个独立判据。

    只报一个数会让人无从判断该不该信；两个都指向同一根轴时才值得放心。
    """
    proc = run_tool("validate_alignment.py", "--mesh", str(FIXTURES / "field_mesh_yup.ply"))
    assert proc.returncode == 0
    assert "推测朝上轴：y" in proc.stdout
    assert "最扁的轴：y" in proc.stdout
    assert "与推测一致" in proc.stdout
    assert "两个判据都不支持" not in proc.stdout


def test_suggest_warns_when_the_two_criteria_disagree():
    """两个判据打架时必须报警，而不是挑一个报完了事。

    region_field 这份网格正是如此：共面判据说 z 向上（对的），但它的最扁轴
    是 y——因为里面塞了大片围墙和一根 18m 高的杆件。把不确定性明说出来，
    比给一个"看起来很确定"的错误答案强。
    """
    path = (
        ROOT.parent / "radar" / "TCR-RM2026-Radar-OpenSource-main"
        / "ground_vision" / "field" / "RM2026_region_field_v1.0_tri.ply"
    )
    if not path.exists():
        pytest.skip("参考仓库不在本机")
    proc = run_tool("validate_alignment.py", "--mesh", str(path))
    assert proc.returncode == 0
    assert "与推测不一致" in proc.stdout
    assert "两个判据都不支持" in proc.stdout
