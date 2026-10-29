"""变换链的合成往返测试。**不需要任何真实数据、硬件或权重。**

这是整个项目的精度地基：如果 3D → 像素 → 射线 → 3D 的往返在合成数据上都
做不到 1cm，那么真实标定的精度讨论就没有意义。

合成设置模仿真实雷达站：相机架在 3.6m 高、场外一侧，看向场地中心，
1200 万像素 + 12mm 镜头。
"""

from __future__ import annotations

import numpy as np
import pytest

from jg_radar.transform import (
    CameraModel,
    Intrinsics,
    build_lookup_map,
    camera_center,
    compute_signature,
    intersect_mesh,
    intersect_plane,
    intersect_triangles,
    look_at,
    matrix_to_rodrigues,
    orthonormalize,
    refine_pose,
    reprojection_errors,
    rodrigues_to_matrix,
    solve_pnp_ransac,
    world_to_camera,
)
from jg_radar.field.mesh import estimate_ground_z
from jg_radar.transform.lookup_map import LookupMap

# ----------------------------------------------------------------------
# 合成场景
# ----------------------------------------------------------------------
FIELD_X = 2800.0
FIELD_Y = 1500.0
CAM_EYE = np.array([1400.0, -500.0, 360.0])
CAM_TARGET = np.array([1400.0, 750.0, 0.0])
PLANE_Z = 0.0

K = [[3478.0, 0.0, 2012.0], [0.0, 3478.0, 1518.0], [0.0, 0.0, 1.0]]
DIST_REAL = [-0.12, 0.03, 1e-4, -1e-4, 0.0]


def make_camera(dist=None, width=4024, height=3036) -> CameraModel:
    intr = Intrinsics.from_matrix(K, dist if dist is not None else [0.0] * 5)
    return CameraModel(intr, width, height)


def make_pose():
    return look_at(CAM_EYE, CAM_TARGET)


def make_small_camera(width: int, height: int) -> CameraModel:
    """小图相机：内参与分辨率匹配（大 K 配小图会让所有像素落在视野外）。"""
    fx = 1.2 * float(width)
    intr = Intrinsics(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0)
    return CameraModel(intr, width, height)


def ground_points(step: float = 200.0) -> np.ndarray:
    xs = np.arange(0.0, FIELD_X + 1.0, step)
    ys = np.arange(0.0, FIELD_Y + 1.0, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)


# ----------------------------------------------------------------------
# 往返精度
# ----------------------------------------------------------------------
def test_roundtrip_no_distortion_under_1cm():
    """无畸变：3D → 像素 → 射线 → 地面，误差应远小于 1cm。"""
    cam = make_camera()
    R, t = make_pose()

    world = ground_points()
    pixels, valid = cam.project_world(world, R, t)
    visible = valid & cam.in_image(pixels)
    assert visible.sum() > 30, "合成场景里可见的地面点太少，测试没有意义"

    origins, dirs = cam.rays_world(pixels[visible], R, t)
    hit = intersect_plane(origins, dirs, PLANE_Z)
    assert hit.valid.all()

    err = np.linalg.norm(hit.points_cm[:, :2] - world[visible][:, :2], axis=1)
    assert err.max() < 0.01, f"最大往返误差 {err.max():.4f} cm"
    assert err.max() < 1.0  # 方案门限


def test_roundtrip_with_real_distortion_under_1cm():
    """开畸变后仍然满足 1cm 门限——验证迭代去畸变真的收敛。"""
    cam = make_camera(DIST_REAL)
    R, t = make_pose()

    world = ground_points()
    pixels, valid = cam.project_world(world, R, t)
    visible = valid & cam.in_image(pixels)

    origins, dirs = cam.rays_world(pixels[visible], R, t)
    hit = intersect_plane(origins, dirs, PLANE_Z)

    err = np.linalg.norm(hit.points_cm[:, :2] - world[visible][:, :2], axis=1)
    assert err.max() < 1.0, f"最大往返误差 {err.max():.4f} cm"


def test_distort_undistort_is_inverse():
    """归一化坐标 → 畸变 → 去畸变 应还原。"""
    cam = make_camera(DIST_REAL)
    xy = np.array(
        [[-0.5, -0.35], [0.0, 0.0], [0.5, 0.35], [0.2, -0.1], [-0.3, 0.4], [0.45, 0.05]]
    )
    px = cam.distort_normalized(xy)
    back = cam.undistort_to_normalized(px)
    assert np.abs(back - xy).max() < 1e-6


def test_pixel_roundtrip_through_undistort():
    """像素 → 去畸变 → 再畸变 应还原（同一内参）。

    5 次不动点迭代在画面边缘的残差量级约 0.01px——足以支撑 1cm 的往返门限
    （20m 处 0.01px ≈ 0.006cm），但不要指望它到 1e-6。
    """
    cam = make_camera(DIST_REAL)
    px = np.array([[100.0, 200.0], [2012.0, 1518.0], [3900.0, 2900.0], [2000.0, 100.0]])
    xy = cam.undistort_to_normalized(px)
    back = cam.distort_normalized(xy)
    assert np.abs(back - px).max() < 0.05


def test_undistort_iterations_converge():
    """迭代次数增加，残差单调不增且很快饱和。"""
    intr = Intrinsics.from_matrix(K, DIST_REAL)
    xy = np.array([[0.5, 0.35], [-0.45, -0.3], [0.1, 0.4]])
    errs = []
    for iters in (1, 2, 3, 5, 10):
        cam = CameraModel(intr, 100, 100, undistort_iters=iters)
        px = cam.distort_normalized(xy)
        back = cam.undistort_to_normalized(px)
        errs.append(np.abs(back - xy).max())
    assert errs[-1] < 1e-8
    assert errs[3] < 1e-6


# ----------------------------------------------------------------------
# 外参与位姿
# ----------------------------------------------------------------------
def test_look_at_puts_target_at_principal_point():
    cam = make_camera()
    R, t = make_pose()
    px, valid = cam.project_world(CAM_TARGET[None, :], R, t)
    assert valid[0]
    assert px[0] == pytest.approx([2012.0, 1518.0], abs=1e-6)


def test_camera_center_recovered():
    R, t = make_pose()
    assert np.allclose(camera_center(R, t), CAM_EYE, atol=1e-9)


def test_world_to_camera_then_back():
    R, t = make_pose()
    pts = ground_points(500.0)
    cam_pts = world_to_camera(pts, R, t)
    back = (cam_pts - t) @ R
    assert np.allclose(back, pts, atol=1e-9)


def test_look_at_axes_follow_opencv_convention():
    """相机系 z 向前、y 向下。"""
    R, t = make_pose()
    z_axis, y_axis = R[2], R[1]
    forward = CAM_TARGET - CAM_EYE
    forward /= np.linalg.norm(forward)
    assert np.allclose(z_axis, forward, atol=1e-12)
    assert y_axis[2] < 0, "y 轴应朝下"


def test_look_at_rejects_degenerate():
    with pytest.raises(ValueError, match="重合"):
        look_at([0, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError, match="平行"):
        look_at([0, 0, 0], [0, 0, 1], up=(0, 0, 1))


def test_points_behind_camera_are_invalid():
    cam = make_camera()
    R, t = make_pose()
    behind = np.array([[CAM_EYE[0], CAM_EYE[1] - 1000.0, 0.0]])
    px, valid = cam.project_world(behind, R, t)
    assert not valid[0]
    assert np.isnan(px[0]).all()


def test_rodrigues_roundtrip():
    for rvec in ([0.1, -0.2, 0.3], [0.0, 0.0, 0.0], [0.001, 0.0, 0.0]):
        R = rodrigues_to_matrix(rvec)
        assert np.allclose(matrix_to_rodrigues(R), rvec, atol=1e-9)


def test_orthonormalize_fixes_drift():
    R, _ = make_pose()
    noisy = R + 1e-3 * np.random.default_rng(0).normal(size=(3, 3))
    Rn = orthonormalize(noisy)
    assert np.allclose(Rn @ Rn.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(Rn) == pytest.approx(1.0)


# ----------------------------------------------------------------------
# PnP
# ----------------------------------------------------------------------
def _calibration_points() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """在场地里布 12 个已知点（含不同高度），并投影成像素。"""
    rng = np.random.default_rng(7)
    world = np.stack(
        [
            rng.uniform(200.0, FIELD_X - 200.0, 12),
            rng.uniform(100.0, FIELD_Y - 100.0, 12),
            rng.choice([0.0, 30.0, 120.0, 250.0], 12),
        ],
        axis=1,
    )
    cam = make_camera(DIST_REAL)
    R, t = make_pose()
    pixels, valid = cam.project_world(world, R, t)
    assert valid.all()
    return world, pixels, R, t


def test_pnp_recovers_known_pose():
    """喂**原始含畸变**像素，PnP 应自己处理去畸变。"""
    world, pixels, R_true, t_true = _calibration_points()
    intr = Intrinsics.from_matrix(K, DIST_REAL)

    result = solve_pnp_ransac(world, pixels, intr, thresh_px=1.0, seed=0)
    assert result is not None
    assert result.rms_px < 0.5, f"RMS {result.rms_px:.4f} px"
    assert result.acceptable()

    assert np.allclose(camera_center(result.pose.R, result.pose.t), CAM_EYE, atol=1.0)
    assert np.allclose(result.pose.R, R_true, atol=1e-2)
    assert np.allclose(result.pose.t, t_true, atol=2.0)


def test_pnp_coplanar_points():
    """**共面**的点也必须能解出位姿。

    这不是边角情形，而是主场景：场地上能量出来、又能命名的特征（角点、中线
    端点）几乎全在地面 z=0 上。共面会让 DLT 里"乘 z"的整列完全不受数据约束，
    零空间从 1 维涨到 4 维——照搬 Vt[-1] 拿到的是零空间里任意一个方向，实测
    直接返回过全零的 A3，随后 RANSAC 一个内点都找不到。
    """
    cam = make_camera(DIST_REAL)
    R_true, t_true = make_pose()
    intr = Intrinsics.from_matrix(K, DIST_REAL)

    world = ground_points(400.0)          # 全部 z=0
    assert world.shape[0] >= 20
    pixels, valid = cam.project_world(world, R_true, t_true)
    assert valid.all()

    result = solve_pnp_ransac(world, pixels, intr, thresh_px=1.0, seed=0)
    assert result is not None, "共面点让 DLT 退化成了无解"
    assert result.rms_px < 0.5
    assert np.allclose(result.pose.R, R_true, atol=1e-3)
    assert np.allclose(camera_center(result.pose.R, result.pose.t), CAM_EYE, atol=0.5)


def test_pnp_double_distortion_would_show_up():
    """守住上面那个坑：若拿去畸变像素配含畸变内参，残差会爆炸。"""
    world, pixels, _, _ = _calibration_points()
    intr_dist = Intrinsics.from_matrix(K, DIST_REAL)
    cam = make_camera(DIST_REAL)
    px_undist = cam.undistort_pixels(pixels)

    good = solve_pnp_ransac(world, px_undist, intr_dist.without_distortion(),
                            thresh_px=1.0, seed=0)
    assert good is not None and good.rms_px < 0.5

    # 同一个位姿，用含畸变内参去算重投影 → 误差巨大
    bad_errs = reprojection_errors(good.pose, world, px_undist, intr_dist)
    assert np.median(bad_errs) > 10.0


def test_pnp_rejects_outliers():
    """两个点被故意投错位置，RANSAC 应把它们剔出内点集。"""
    world, pixels, _, _ = _calibration_points()
    pixels[3] += np.array([400.0, -300.0])
    pixels[9] += np.array([-250.0, 500.0])

    result = solve_pnp_ransac(world, pixels, Intrinsics.from_matrix(K, DIST_REAL),
                              thresh_px=2.0, seed=1)
    assert result is not None
    assert not result.inliers[3]
    assert not result.inliers[9]
    assert result.inliers.sum() >= 10
    assert result.rms_px < 1.0


def test_pnp_too_few_points():
    world = np.zeros((3, 3))
    pixels = np.zeros((3, 2))
    assert solve_pnp_ransac(world, pixels, Intrinsics.from_matrix(K)) is None


def test_refine_pose_improves_reprojection():
    world, pixels, _, _ = _calibration_points()
    intr = Intrinsics.from_matrix(K, DIST_REAL)

    result = solve_pnp_ransac(world, pixels, intr, thresh_px=1.0, seed=0)
    assert result is not None
    errs = reprojection_errors(result.pose, world, pixels, intr)
    refined = refine_pose(result.pose, world, pixels, intr)
    errs_refined = reprojection_errors(refined, world, pixels, intr)
    assert errs_refined.max() <= errs.max() + 1e-6


# ----------------------------------------------------------------------
# 射线求交
# ----------------------------------------------------------------------
def test_plane_intersection_basic():
    origins = np.array([[0.0, 0.0, 100.0]])
    dirs = np.array([[0.0, 0.0, -1.0]])
    hit = intersect_plane(origins, dirs, 0.0)
    assert hit.valid[0]
    assert hit.t[0] == pytest.approx(100.0)
    assert np.allclose(hit.points_cm[0], [0.0, 0.0, 0.0])


def test_plane_intersection_rejects_parallel_and_backward():
    origins = np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 100.0]])
    dirs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 平行 / 背向
    hit = intersect_plane(origins, dirs, 0.0)
    assert not hit.valid.any()
    assert np.isnan(hit.points_cm).all()


def test_triangle_intersection():
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([100.0, 0.0, 0.0])
    v2 = np.array([0.0, 100.0, 0.0])
    origins = np.array([[20.0, 20.0, 50.0], [80.0, 80.0, 50.0], [20.0, 20.0, -50.0]])
    dirs = np.array([[0.0, 0.0, -1.0]] * 3)
    t = intersect_triangles(origins, dirs, v0, v1, v2)
    assert t[0, 0] == pytest.approx(50.0)   # 命中
    assert np.isnan(t[1, 0])                # 在三角形外
    assert np.isnan(t[2, 0])                # 在射线反方向


def test_intersect_mesh_nearest_hit():
    verts = np.array(
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0],
         [0.0, 0.0, 50.0], [100.0, 0.0, 50.0], [0.0, 100.0, 50.0]]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    origins = np.array([[10.0, 10.0, 200.0]])
    dirs = np.array([[0.0, 0.0, -1.0]])
    hit = intersect_mesh(origins, dirs, verts, faces)
    assert hit.valid[0]
    assert hit.t[0] == pytest.approx(150.0)  # 先撞到 z=50 那块
    assert hit.face_index[0] == 1
    assert hit.points_cm[0, 2] == pytest.approx(50.0)


def test_intersect_mesh_with_mask():
    verts = np.array(
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0],
         [0.0, 0.0, 50.0], [100.0, 0.0, 50.0], [0.0, 100.0, 50.0]]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    origins = np.array([[10.0, 10.0, 200.0]])
    dirs = np.array([[0.0, 0.0, -1.0]])
    hit = intersect_mesh(origins, dirs, verts, faces, mask=np.array([True, False]))
    assert hit.t[0] == pytest.approx(200.0)  # 忽略上层，撞地面


def test_estimate_ground_z():
    verts = np.array(
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0],
         [0.0, 0.0, 3.0], [100.0, 0.0, 3.0], [0.0, 100.0, 3.0]]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    z = estimate_ground_z(verts, faces)
    assert z in (0.0, 3.0)


# ----------------------------------------------------------------------
# 查找表
# ----------------------------------------------------------------------
def test_lookup_map_matches_direct_raycast():
    """查找表必须与逐像素射线求交**逐点一致**。"""
    cam = make_small_camera(200, 150)
    R, t = make_pose()
    lm = build_lookup_map(cam, R, t, PLANE_Z)
    assert lm.width == 200 and lm.height == 150
    assert lm.coverage > 0.0

    vs, us = np.nonzero(lm.valid)
    pixels = np.stack([us, vs], axis=1).astype(np.float64)
    origins, dirs = cam.rays_world(pixels, R, t)
    hit = intersect_plane(origins, dirs, PLANE_Z)
    assert hit.valid.all()
    err = np.linalg.norm(lm.coords_cm[vs, us] - hit.points_cm[:, :2], axis=1)
    assert err.max() < 1e-9


def test_lookup_map_sample_and_bbox():
    cam = make_small_camera(200, 150)
    R, t = make_pose()
    lm = build_lookup_map(cam, R, t, PLANE_Z)

    world, ok = lm.sample(np.array([[100.0, 75.0]]))
    assert ok[0]
    assert np.allclose(world[0], lm.coords_cm[75, 100])

    # 越界像素：无效但不抛异常
    _, ok_out = lm.sample(np.array([[-5.0, 10.0], [500.0, 10.0]]))
    assert not ok_out.any()

    w2, ok2 = lm.sample_bbox_center(np.array([[90.0, 65.0, 110.0, 85.0]]))
    assert ok2[0]
    assert np.allclose(w2[0], lm.coords_cm[75, 100])


def test_lookup_map_signature_sensitivity():
    intr = Intrinsics.from_matrix(K)
    R, t = make_pose()
    base = compute_signature(intr, R, t, 200, 150, 0.0)

    assert base == compute_signature(intr, R, t, 200, 150, 0.0)  # 稳定
    assert base != compute_signature(intr, R, t, 200, 150, 3.0)  # 地面高度变了
    assert base != compute_signature(intr, R, t, 201, 150, 0.0)  # 分辨率变了
    assert base != compute_signature(
        Intrinsics.from_matrix(K, DIST_REAL), R, t, 200, 150, 0.0
    )
    assert base != compute_signature(intr, R, t + np.array([1.0, 0.0, 0.0]),
                                     200, 150, 0.0)
    assert base != compute_signature(intr, R, t, 200, 150, 0.0, mesh_hash="abc")


def test_lookup_map_save_load(tmp_path):
    cam = make_small_camera(120, 90)
    R, t = make_pose()
    lm = build_lookup_map(cam, R, t, PLANE_Z)
    path = lm.save(tmp_path / "lookup.npz")

    back = LookupMap.load(path)
    assert back.signature == lm.signature
    assert back.plane_z == pytest.approx(lm.plane_z)
    # 落盘用 float32，量化误差 ~1e-4 cm（1 微米）
    assert np.allclose(back.coords_cm, lm.coords_cm, rtol=1e-6, atol=1e-3, equal_nan=True)
    assert (back.valid == lm.valid).all()
    assert back.matches(lm.signature)
    assert not back.matches("deadbeef")


def test_lookup_map_stride_leaves_holes():
    cam = make_small_camera(64, 48)
    R, t = make_pose()
    full = build_lookup_map(cam, R, t, PLANE_Z, stride=1)
    coarse = build_lookup_map(cam, R, t, PLANE_Z, stride=4)
    assert coarse.coverage < full.coverage
    # 抽稀表上的点仍与全表一致
    for v in range(0, 48, 4):
        for u in range(0, 64, 4):
            if coarse.valid[v, u]:
                assert np.allclose(coarse.coords_cm[v, u], full.coords_cm[v, u], atol=1e-9)


# ----------------------------------------------------------------------
# 锚点选择：框中心 vs 底边中点
# ----------------------------------------------------------------------
ROBOT_HEIGHT_CM = 60.0
"""车顶相对地面的高度。撑出一个有真实透视缩放的框，而不是给个常数像素高。"""


def _robot_bbox(cam, R, t, gx: float, gy: float):
    """由地面接触点撑出一个轴对齐车框。底边中点 == 投影出的地面点。"""
    px, valid = cam.project_world(
        np.array([[gx, gy, 0.0], [gx, gy, ROBOT_HEIGHT_CM]]), R, t
    )
    assert valid.all()
    u = float(px[0, 0])
    bottom, top = float(px[0, 1]), float(px[1, 1])
    half_w = (bottom - top) * 0.575
    return (u - half_w, top, u + half_w, bottom)


def _raycast_pixel(cam, R, t, pixel):
    """一个像素 → 射线 → 与地面求交 → 场地坐标。"""
    origins, dirs = cam.rays_world(np.asarray([pixel], dtype=np.float64), R, t)
    hit = intersect_plane(origins, dirs, PLANE_Z)
    assert hit.valid.all()
    return hit.points_cm[0, :2]


def test_the_anchor_pixel_decides_the_position_by_metres():
    """**回归测试。** 框中心当锚点，会把机器人投到它身后一两米。

    这是本项目最贵的一个像素选择。相机是俯视的，穿过框中心的射线比穿过
    底边中点的射线更"平"，打到地面平面时要多走一段——**车越高、相机越平、
    距离越远，这段越长**。

    量级不是可以忽略的那种：

    - 底边中点锚点：**厘米级以下**（合成场景里 0–1.5cm）
    - 框中心锚点：**米级**（87–182cm，且随距离增长）

    而 ``switch_jump_cm`` 才 150cm。也就是说，光是把锚点选错，远处目标的
    每一次换源都会被判成"跳变"，滞回门被迫抬到 ``reacquire_s``——
    这个项目最在意的"丝滑"就没了，而且是**静默地**没掉：没有任何报错，
    只是每次换源都多等 500ms 并且挑战者永远攒不够一致性。

    所以这条测试盯的不是数值精度，是**不许把这两个锚点搞混**。
    """
    cam = make_camera(dist=DIST_REAL)
    R, t = make_pose()

    worst_center, worst_ground = 0.0, 0.0
    for gx, gy in [(1500.0, 450.0), (1400.0, 750.0), (2100.0, 1400.0), (900.0, 880.0)]:
        bbox = _robot_bbox(cam, R, t, gx, gy)
        truth = np.array([gx, gy])
        u = (bbox[0] + bbox[2]) * 0.5
        by_center = np.linalg.norm(
            _raycast_pixel(cam, R, t, (u, (bbox[1] + bbox[3]) * 0.5)) - truth
        )
        by_ground = np.linalg.norm(_raycast_pixel(cam, R, t, (u, bbox[3])) - truth)
        worst_center = max(worst_center, by_center)
        worst_ground = max(worst_ground, by_ground)

    assert worst_ground < 3.0, f"底边中点锚点应当复现接触点，实测偏差 {worst_ground:.2f}cm"
    assert worst_center > 80.0, f"框中心锚点应当明显偏后，实测只有 {worst_center:.2f}cm"
    assert worst_center > 20 * worst_ground, "两个锚点必须差出一个量级以上"


def test_the_bias_grows_with_distance():
    """偏差随距离增长——所以"近处看着还行"不能用来论证锚点选对了。

    这条是给未来的人看的：如果哪天有人拿一组近场数据验证"框中心也差不多"，
    把它贴到 20m 外的目标上就会翻车。
    """
    cam = make_camera(dist=DIST_REAL)
    R, t = make_pose()

    def bias_at(gx, gy):
        bbox = _robot_bbox(cam, R, t, gx, gy)
        u = (bbox[0] + bbox[2]) * 0.5
        got = _raycast_pixel(cam, R, t, (u, (bbox[1] + bbox[3]) * 0.5))
        return float(np.linalg.norm(got - np.array([gx, gy])))

    near = bias_at(1400.0, 450.0)   # 离相机近
    far = bias_at(2100.0, 1400.0)   # 远角
    assert far > near * 1.5, f"远处偏差 {far:.1f}cm 应当显著大于近处 {near:.1f}cm"


def test_lookup_map_ground_anchor_uses_the_bottom_edge():
    """``sample_bbox_ground`` 取的必须是 (底边中点)，不是 (框中心)。"""
    cam = make_small_camera(200, 150)
    R, t = make_pose()
    lm = build_lookup_map(cam, R, t, PLANE_Z)

    bbox = np.array([[90.0, 65.0, 110.0, 85.0]])
    world, ok = lm.sample_bbox_ground(bbox)
    assert ok[0]
    # u = (90+110)/2 = 100，v = y2 = 85
    assert np.allclose(world[0], lm.coords_cm[85, 100])
    # 和框中心锚点必须落在不同的格子上（框中心是 v=75）
    center, ok_c = lm.sample_bbox_center(bbox)
    assert ok_c[0]
    assert not np.allclose(world[0], center[0])
