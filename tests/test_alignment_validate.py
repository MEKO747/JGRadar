"""对齐体检的端到端测试 —— M2 的验收门。

fixture 是**自描述**的：`keypoints_synth.json` 里除了关键点，还写了生成这些
像素时用的相机与位姿。所以这里不需要在测试里重抄一遍常量，"真值"与"被检验
的对象"天然分离——如果哪天生成脚本和验证代码各自演化出不同的假设，这组测试
会立刻炸掉。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jg_radar.field import (
    KeypointSet,
    coverage_report,
    validate_alignment,
)
from jg_radar.transform import (
    CameraModel,
    Intrinsics,
    Pose,
    look_at,
    solve_pnp_ransac,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
KEYPOINTS = FIXTURES / "keypoints_synth.json"

# 方案 §4.2 定的门限
RMS_MAX_PX = 3.0
MAX_MAX_PX = 8.0
ROUNDTRIP_MAX_CM = 2.0


# ----------------------------------------------------------------------
# fixture 装载
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def fixture():
    doc = json.loads(KEYPOINTS.read_text(encoding="utf-8"))
    kps = KeypointSet.from_dict(doc)

    cam = doc["camera"]
    intr = Intrinsics.from_matrix(cam["K"], cam["dist"])
    camera = CameraModel(intr, cam["width"], cam["height"])

    R, t = look_at(doc["pose"]["eye"], doc["pose"]["target"])
    return {"doc": doc, "kps": kps, "camera": camera, "intr": intr, "pose": Pose(R, t)}


# ----------------------------------------------------------------------
# fixture 本身要自洽
# ----------------------------------------------------------------------
def test_fixture_is_self_describing(fixture):
    assert len(fixture["kps"]) == 10
    assert len(fixture["kps"].calibratable) == 10
    assert fixture["camera"].intrinsics.has_distortion


def test_fixture_keypoints_are_on_the_ground(fixture):
    """全部关键点必须位于地面高度。

    不是洁癖：射线只能与地面平面求交，拿一个悬空点做往返检验会得到
    一个恒定的假阳性，把整份报告污染掉。
    """
    z = fixture["kps"].world_array()[:, 2]
    assert np.allclose(z, 0.0)


# ----------------------------------------------------------------------
# 真值位姿必须通过
# ----------------------------------------------------------------------
def test_ground_truth_pose_passes(fixture):
    report = validate_alignment(fixture["camera"], fixture["pose"], fixture["kps"],
                                plane_z=0.0)
    assert report.rms_px < RMS_MAX_PX
    assert report.max_px < MAX_MAX_PX
    assert report.max_roundtrip_cm < ROUNDTRIP_MAX_CM
    assert report.acceptable(rms_max_px=RMS_MAX_PX, max_max_px=MAX_MAX_PX,
                             roundtrip_max_cm=ROUNDTRIP_MAX_CM)
    assert report.failures(rms_max_px=RMS_MAX_PX, max_max_px=MAX_MAX_PX,
                           roundtrip_max_cm=ROUNDTRIP_MAX_CM) == []


def test_ground_truth_pose_is_essentially_exact(fixture):
    """合成数据的残差应当接近数值噪声，而不是"刚好卡在门限内"。

    如果这里跑出 2.9px，说明链路里混进了系统误差（最典型的是畸变被施加了
    两次），那门限就成了掩盖问题的遮羞布。
    """
    report = validate_alignment(fixture["camera"], fixture["pose"], fixture["kps"])
    assert report.rms_px < 1e-3
    assert report.max_roundtrip_cm < 0.01


def test_report_shape(fixture):
    report = validate_alignment(fixture["camera"], fixture["pose"], fixture["kps"])
    assert report.num_points == 10
    assert report.num_calibratable == 10
    assert len(report.residuals) == 10
    assert len(report.worst(3)) == 3

    d = report.to_dict()
    assert set(d) >= {"rms_px", "max_px", "rms_roundtrip_cm", "max_roundtrip_cm"}
    assert len(d["residuals"]) == 10
    assert "对齐体检" in report.summary()
    assert "通过" in report.summary()


# ----------------------------------------------------------------------
# 错误位姿必须被拦住
# ----------------------------------------------------------------------
def _translated(pose: Pose, delta_cm) -> Pose:
    """光心平移 delta_cm：R 不变，t 要按相机系重新算。"""
    center = pose.center_cm + np.asarray(delta_cm, dtype=np.float64)
    t = -pose.R @ center
    return Pose(pose.R, t)


def test_small_pose_error_is_caught_by_roundtrip(fixture):
    """光心偏 30cm：反投影能过，往返必须过不去。

    这正是"只测反投影不够"的证据——针孔模型下的反投影对小位移不敏感，
    而运行时真正走的射线求交会立刻暴露它。
    """
    bad = _translated(fixture["pose"], [30.0, 0.0, 0.0])
    report = validate_alignment(fixture["camera"], bad, fixture["kps"])
    assert not report.acceptable(rms_max_px=RMS_MAX_PX, max_max_px=MAX_MAX_PX,
                                 roundtrip_max_cm=ROUNDTRIP_MAX_CM)
    assert report.max_roundtrip_cm > ROUNDTRIP_MAX_CM
    assert any("往返" in f for f in report.failures())


def test_rotated_pose_fails(fixture):
    """绕 z 轴转 1°：足以让远处的往返误差远超 2cm。"""
    from jg_radar.transform import rodrigues_to_matrix

    dR = rodrigues_to_matrix(np.array([0.0, 0.0, np.deg2rad(1.0)]))
    R = dR @ fixture["pose"].R
    bad = Pose(R, -R @ fixture["pose"].center_cm)
    report = validate_alignment(fixture["camera"], bad, fixture["kps"])
    assert not report.acceptable(rms_max_px=RMS_MAX_PX, max_max_px=MAX_MAX_PX,
                                 roundtrip_max_cm=ROUNDTRIP_MAX_CM)


def test_wrong_plane_z_shows_up_as_roundtrip_error(fixture):
    """地面高度填错 20cm → 往返残差随距离成比例放大。"""
    report = validate_alignment(fixture["camera"], fixture["pose"], fixture["kps"],
                                plane_z=20.0)
    assert report.max_roundtrip_cm > ROUNDTRIP_MAX_CM


# ----------------------------------------------------------------------
# PnP：从关键点反解位姿
# ----------------------------------------------------------------------
def test_pnp_recovers_ground_truth_pose(fixture):
    kps = fixture["kps"]
    result = solve_pnp_ransac(kps.world_array(only_calibratable=True),
                              kps.pixel_array(), fixture["intr"],
                              thresh_px=6.0, refine=True)
    assert result is not None
    assert result.inliers.sum() >= 8
    assert result.acceptable(rms_max_px=RMS_MAX_PX)

    truth = fixture["pose"]
    assert np.allclose(result.pose.R, truth.R, atol=1e-4)
    assert np.allclose(result.pose.center_cm, truth.center_cm, atol=0.1)


def test_pnp_result_passes_validation(fixture):
    """PnP 解出来的位姿，直接送进体检也必须通过。"""
    kps = fixture["kps"]
    result = solve_pnp_ransac(kps.world_array(only_calibratable=True),
                              kps.pixel_array(), fixture["intr"], refine=True)
    report = validate_alignment(fixture["camera"], result.pose, kps)
    assert report.acceptable(rms_max_px=RMS_MAX_PX, max_max_px=MAX_MAX_PX,
                             roundtrip_max_cm=ROUNDTRIP_MAX_CM)


def test_pnp_needs_six_points(fixture):
    """少于 6 个点必须干脆地返回 None，而不是给一个看着像样的坏位姿。"""
    kps = fixture["kps"]
    assert solve_pnp_ransac(kps.world_array()[:5], kps.pixel_array()[:5],
                            fixture["intr"]) is None


# ----------------------------------------------------------------------
# 覆盖面
# ----------------------------------------------------------------------
def test_coverage_report(fixture):
    cov = coverage_report(fixture["camera"], fixture["pose"], plane_z=0.0)
    assert cov["samples"] > 0
    assert 0.0 <= cov["coverage"] <= 1.0
    assert cov["visible"] <= cov["samples"]
    # 合成相机架在红线后方俯视全场，地面覆盖应当很可观
    assert cov["coverage"] > 0.5


def test_coverage_report_is_in_diagnostics(fixture):
    report = validate_alignment(
        fixture["camera"], fixture["pose"], fixture["kps"],
        extra_diagnostics=coverage_report(fixture["camera"], fixture["pose"]),
    )
    assert "coverage" in report.diagnostics
    assert "coverage" in report.to_dict()["diagnostics"]
