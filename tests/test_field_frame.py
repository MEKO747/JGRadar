"""场地坐标系、阵营翻转、网格朝向判定。

这一组测试对应方案里的**头号风险**（CAD 坐标系对齐）。这里的每个断言都
是在防"网格朝向猜错了但没人发现"——那种错误不会让程序崩溃，只会让所有
坐标静静地偏上几米。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jg_radar.field import (
    FieldMesh,
    detect_up_axis,
    in_field,
    clip_to_field,
    invert_transform,
    apply_transform,
    load_mesh,
    mirror_to_side,
    rotation_from_axes,
    suggest_mesh_to_field,
    unmirror_to_field,
)
from jg_radar.types import FIELD_MAX_X_CM, FIELD_MAX_Y_CM

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SIMPLE_PLY = FIXTURES / "field_mesh_simple.ply"
YUP_PLY = FIXTURES / "field_mesh_yup.ply"


# ======================================================================
# 阵营
# ======================================================================
def test_mirror_red_is_identity():
    assert mirror_to_side((300.0, 200.0), "R") == (300.0, 200.0)


def test_mirror_blue_is_point_reflection():
    """蓝方 = 绕场地中心的点对称。来源：TCR solidwork2uwb.py 的红蓝公式差。"""
    x, y = mirror_to_side((300.0, 200.0), "B")
    assert x == FIELD_MAX_X_CM - 300.0
    assert y == FIELD_MAX_Y_CM - 200.0


def test_mirror_is_involution():
    """点对称是自逆的：翻两次必须回到原处（否则双方看到的场地不一致）。"""
    for side in ("R", "B"):
        p = (1234.5, 987.6)
        again = mirror_to_side(mirror_to_side(p, side), side)
        assert again == pytest.approx(p)


def test_unmirror_matches_mirror():
    p = (1000.0, 400.0)
    assert unmirror_to_field(mirror_to_side(p, "B"), "B") == pytest.approx(p)


def test_mirror_center_is_fixed_point():
    assert mirror_to_side((1400.0, 750.0), "B") == pytest.approx((1400.0, 750.0))


def test_mirror_unknown_side():
    with pytest.raises(ValueError, match="阵营"):
        mirror_to_side((0.0, 0.0), "G")


def test_in_field_and_clip():
    assert in_field(0.0, 0.0) and in_field(FIELD_MAX_X_CM, FIELD_MAX_Y_CM)
    assert not in_field(-1.0, 0.0)
    assert not in_field(0.0, FIELD_MAX_Y_CM + 1.0)
    assert in_field(-5.0, 0.0, margin=10.0)
    assert clip_to_field(-100.0, 9999.0) == (0.0, FIELD_MAX_Y_CM)


# ======================================================================
# 轴向旋转
# ======================================================================
def test_rotation_z_up_x_forward_is_identity():
    """网格已经是 Z 向上、x 指向蓝方 → 变换应为单位阵。"""
    assert np.allclose(rotation_from_axes("z", "x"), np.eye(3))


def test_rotation_det_is_always_plus_one():
    """行列式恒为 +1 —— 镜像会让所有面法线反向，遮挡判断随之失效。"""
    for up in "xyz":
        for forward in "xyz":
            if up == forward:
                continue
            R = rotation_from_axes(up, forward)
            assert np.linalg.det(R) == pytest.approx(1.0)
            assert np.allclose(R @ R.T, np.eye(3))


def test_rotation_maps_axes_as_advertised():
    R = rotation_from_axes("y", "x")
    # 网格的 forward 轴 (1,0,0) → 场地 +x
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0])
    # 网格的 up 轴 (0,1,0) → 场地 +z
    assert np.allclose(R @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0])


def test_rotation_signs():
    R = rotation_from_axes("y", "x", up_sign=-1)
    assert np.allclose(R @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, -1.0])
    R = rotation_from_axes("y", "x", forward_sign=-1)
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), [-1.0, 0.0, 0.0])


def test_rotation_rejects_bad_axes():
    with pytest.raises(ValueError, match="轴名"):
        rotation_from_axes("w", "x")
    with pytest.raises(ValueError, match="同一根轴"):
        rotation_from_axes("z", "z")


def test_rotation_undoes_the_yup_fixture():
    """把 Y 向上的网格旋回 Z 向上，必须逐顶点等于原始网格。

    这条断言就是 fixture 里 to_y_up() 的逆：旋转猜错的话，误差不会是小
    数点级别，而是整根轴对调。
    """
    zup = load_mesh(SIMPLE_PLY)
    yup = load_mesh(YUP_PLY)
    R = rotation_from_axes("y", "x")
    recovered = yup.vertices @ R.T
    assert np.allclose(recovered, zup.vertices, atol=1e-6)


# ======================================================================
# 轴向检测
# ======================================================================
def test_detect_up_axis_on_zup_mesh():
    raw = load_mesh(SIMPLE_PLY)
    up, conf = detect_up_axis(raw.vertices, raw.faces)
    assert up == "z"
    assert conf > 0.5


def test_detect_up_axis_on_yup_mesh():
    raw = load_mesh(YUP_PLY)
    up, conf = detect_up_axis(raw.vertices, raw.faces)
    assert up == "y"
    assert conf > 0.5


def test_detect_up_axis_empty_mesh():
    assert detect_up_axis(np.zeros((0, 3)), np.zeros((0, 3), dtype=int)) == ("z", 0.0)


# ======================================================================
# 摆放建议
# ======================================================================
def test_suggest_leaves_a_zup_cm_mesh_in_place():
    """已经是 Z 向上、cm 单位、已落位的网格，建议变换应近似单位阵。"""
    raw = load_mesh(SIMPLE_PLY)
    place = suggest_mesh_to_field(raw.vertices, raw.faces)
    assert place.up_axis == "z"
    assert place.forward_axis == "x"          # 28m 那条边
    assert place.unit_scale == 1.0
    assert np.allclose(place.matrix[:3, :3], np.eye(3), atol=1e-9)
    assert np.allclose(place.matrix[:3, 3], 0.0, atol=1e-6)


def test_suggest_guesses_millimetre_units():
    """跨度 ~2800 判为 cm，跨度 ~28 判为 m→需 ×100。

    这里把网格缩小 100 倍模拟"以米为单位导出"，单位换算必须被猜出来，
    否则场地会变成 28cm × 15cm。
    """
    raw = load_mesh(SIMPLE_PLY)
    metres = type(raw)(raw.vertices / 100.0, raw.faces, raw.source)
    place = suggest_mesh_to_field(metres.vertices, metres.faces)
    assert place.unit_scale == 100.0
    lo, hi = place.post_bounds
    assert hi[0] - lo[0] == pytest.approx(FIELD_MAX_X_CM, abs=1.0)


def test_suggest_centers_on_field():
    """居中后场地中心应落在 (1400, 750)。"""
    raw = load_mesh(SIMPLE_PLY)
    place = suggest_mesh_to_field(raw.vertices, raw.faces)
    placed = apply_transform(raw.vertices, place.matrix)
    center = (placed.min(axis=0) + placed.max(axis=0)) * 0.5
    assert center[0] == pytest.approx(FIELD_MAX_X_CM / 2)
    assert center[1] == pytest.approx(FIELD_MAX_Y_CM / 2)
    assert placed[:, 2].min() == pytest.approx(0.0)


def test_suggest_on_yup_mesh_puts_field_in_the_right_plane():
    """Y 向上的网格落位后，也必须变成"z 向上、28×15m"的场地。"""
    raw = load_mesh(YUP_PLY)
    place = suggest_mesh_to_field(raw.vertices, raw.faces)
    assert place.up_axis == "y"
    mesh = FieldMesh.from_raw(raw, mesh_to_field=place.matrix,
                              unit_scale=place.unit_scale)
    lo, hi = mesh.bounds()
    assert hi[0] - lo[0] == pytest.approx(FIELD_MAX_X_CM, abs=1.0)
    assert hi[1] - lo[1] == pytest.approx(FIELD_MAX_Y_CM, abs=1.0)
    # 落位后的地面高度必须是 0（mesh_to_field 已把最低点对齐）
    assert mesh.ground_z == pytest.approx(0.0, abs=1e-6)
    # 立柱依然被识别为遮挡面 —— 朝向没被镜像翻转
    assert mesh.occluder_mask.sum() == 4


def test_placement_describe_is_readable():
    raw = load_mesh(SIMPLE_PLY)
    text = suggest_mesh_to_field(raw.vertices, raw.faces).describe()
    assert "up=z" in text and "scale=1" in text


# ======================================================================
# 变换工具
# ======================================================================
def test_invert_transform_roundtrip():
    raw = load_mesh(SIMPLE_PLY)
    place = suggest_mesh_to_field(raw.vertices, raw.faces)
    m = place.matrix
    back = invert_transform(m)
    assert np.allclose(back @ m, np.eye(4), atol=1e-9)
    pts = raw.vertices[:5]
    assert np.allclose(apply_transform(apply_transform(pts, m), back), pts, atol=1e-9)


def test_apply_transform_accepts_single_point():
    """(3,) 与 (N,3) 都要能吃下，省得调用方到处 reshape。"""
    m = np.eye(4)
    m[:3, 3] = [1.0, 2.0, 3.0]
    assert np.allclose(apply_transform(np.array([0.0, 0.0, 0.0]), m), [1.0, 2.0, 3.0])
