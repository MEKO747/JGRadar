"""外参：位姿构造、PnP 求解与精化。**纯 numpy + scipy，不依赖 OpenCV。**

标定流程（见 tools/calibrate_extrinsics.py）：
  1. 在场地里量 ≥6 个可命名的特征点，得到「场地坐标 ↔ 像素」对应；
  2. ``solve_pnp_ransac`` 求 R|t；
  3. ``refine_pose`` 用全部内点做最小二乘精化；
  4. 残差写回配置（``transform.reproj_rms_px``），作为变换噪声下限。

接受条件：RMS < 3px 且 max < 8px。不满足就拒绝启动——宁可没有坐标，
也不要一个系统性偏了的坐标。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from jg_radar.transform.camera_model import (
    Intrinsics,
    camera_center,
    camera_to_world,
    project_points,
    undistort_pixels,
    world_to_camera,
)


# ----------------------------------------------------------------------
# 旋转表示
# ----------------------------------------------------------------------
def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """罗德里格斯向量 → 3×3 旋转矩阵。"""
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """3×3 旋转矩阵 → 罗德里格斯向量。"""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_t = (np.trace(R) - 1.0) * 0.5
    cos_t = float(np.clip(cos_t, -1.0, 1.0))
    theta = float(np.arccos(cos_t))
    if theta < 1e-9:
        return np.zeros(3)
    if abs(np.pi - theta) < 1e-6:
        # 接近 180°：从 R + I 的列里取主轴
        A = R + np.eye(3)
        col = A[:, int(np.argmax(np.linalg.norm(A, axis=0)))]
        axis = col / np.linalg.norm(col)
        return axis * theta
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis * (theta / (2.0 * np.sin(theta)))


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """把近似旋转矩阵投影回 SO(3)（SVD 极分解）。"""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64).reshape(3, 3))
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:  # 防止反射
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


# ----------------------------------------------------------------------
# 位姿构造
# ----------------------------------------------------------------------
def look_at(
    eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0.0, 0.0, 1.0)
) -> tuple[np.ndarray, np.ndarray]:
    """由「相机在哪、看向哪、哪个方向朝上」构造 ``(R, t)``。

    相机系遵循 OpenCV 惯例：z 向前、x 向右、y 向下。
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)

    z_axis = target - eye
    n = np.linalg.norm(z_axis)
    if n < 1e-9:
        raise ValueError("eye 与 target 重合，无法确定朝向")
    z_axis = z_axis / n

    x_axis = np.cross(z_axis, up)
    n = np.linalg.norm(x_axis)
    if n < 1e-9:
        raise ValueError("视线与 up 平行，请换一个 up 向量")
    x_axis = x_axis / n

    y_axis = np.cross(z_axis, x_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=0)  # 行 = 相机轴在场地系中的表示
    t = -R @ eye
    return R, t


@dataclass(frozen=True)
class Pose:
    """相机外参。``R`` 场地→相机，``t`` 场地→相机平移（cm）。"""

    R: np.ndarray
    t: np.ndarray

    @property
    def center_cm(self) -> np.ndarray:
        return camera_center(self.R, self.t)

    def world_to_camera(self, pts: np.ndarray) -> np.ndarray:
        return world_to_camera(pts, self.R, self.t)

    def camera_to_world(self, pts: np.ndarray) -> np.ndarray:
        return camera_to_world(pts, self.R, self.t)

    def as_matrix(self) -> np.ndarray:
        """4×4 齐次矩阵。"""
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    def inverse(self) -> "Pose":
        Ri = self.R.T
        return Pose(Ri, -Ri @ self.t)


# ----------------------------------------------------------------------
# PnP：DLT + RANSAC + 迭代精化
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PnPResult:
    pose: Pose
    inliers: np.ndarray
    reproj_errors_px: np.ndarray
    rms_px: float
    max_px: float
    num_iterations: int = 0

    def acceptable(self, rms_max_px: float = 3.0, max_max_px: float = 8.0) -> bool:
        """门限与 ``AlignmentReport.acceptable`` 同名同义——两处用不同的
        参数名会让调用方在两者之间复制粘贴时静默传错。"""
        return self.rms_px <= rms_max_px and self.max_px <= max_max_px


def _normalize_2d(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley 归一化：质心到原点、均方距离到 √2。返回 ``(T, 归一化点)``。"""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    mean = pts.mean(axis=0)
    d = pts - mean
    mean_dist = float(np.sqrt((d ** 2).sum(axis=1)).mean())
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    T = np.array(
        [[scale, 0.0, -scale * mean[0]], [0.0, scale, -scale * mean[1]], [0.0, 0.0, 1.0]]
    )
    return T, d * scale


def _normalize_3d(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """同上，均方距离到 √3。"""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    mean = pts.mean(axis=0)
    d = pts - mean
    mean_dist = float(np.sqrt((d ** 2).sum(axis=1)).mean())
    scale = np.sqrt(3.0) / mean_dist if mean_dist > 1e-12 else 1.0
    T = np.eye(4)
    T[:3, :3] = scale * np.eye(3)
    T[:3, 3] = -scale * mean
    return T, d * scale


def _dlt(
    world_pts: np.ndarray, pixels: np.ndarray, intrinsics, *, _local_plane: bool = False
) -> tuple[np.ndarray, np.ndarray] | None:
    """已知内参的 DLT。返回 ``(R, t)``，退化时返回 None。"""
    world = np.asarray(world_pts, dtype=np.float64).reshape(-1, 3)
    px = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if len(world) < 6:
        return None
    if not np.isfinite(world).all() or not np.isfinite(px).all():
        return None
    if not _local_plane:
        center = world.mean(axis=0)
        _, singular, axes = np.linalg.svd(world - center, full_matrices=False)
        if singular[1] <= max(singular[0], 1.0) * 1e-10:
            return None
        if singular[2] <= max(singular[0], 1.0) * 1e-8:
            # 平面先换成局部 z=0，分解后转回世界系，保留非零高度/朝向。
            axes[2] = np.cross(axes[0], axes[1])
            local = (world - center) @ axes.T
            local[:, 2] = 0.0
            result = _dlt(local, px, intrinsics, _local_plane=True)
            if result is None:
                return None
            local_R, local_t = result
            R = local_R @ axes
            return R, local_t - R @ center

    T2, xy_n = _normalize_2d(px)
    T3, xyz_n = _normalize_3d(world)

    n = len(world)
    A = np.zeros((2 * n, 12), dtype=np.float64)
    homog = np.concatenate([xyz_n, np.ones((n, 1))], axis=1)
    for i in range(n):
        X = homog[i]
        u, v = xy_n[i]
        A[2 * i, 0:4] = X
        A[2 * i, 8:12] = -u * X
        A[2 * i + 1, 4:8] = X
        A[2 * i + 1, 8:12] = -v * X

    # 共面点（场地关键点几乎总是共面）会让 A 里"乘 z"的三列恒为零，
    # 零空间因此从 1 维涨到 4 维。此时 Vt[-1] 只是零空间里任意一个方向，
    # 完全可能整根落在那些自由分量上——实测就真的返回过 A3 全零。
    # 先把恒零列摘掉再解，最后补零回去，自由分量对求解没有任何贡献。
    col_scale = np.linalg.norm(A, axis=0)
    peak = float(col_scale.max())
    keep = col_scale > peak * 1e-12
    if peak <= 0.0 or int(keep.sum()) < 6:
        return None
    try:
        _, _, Vt = np.linalg.svd(A[:, keep])
    except np.linalg.LinAlgError:
        return None
    p = np.zeros(12, dtype=np.float64)
    p[keep] = Vt[-1]
    Pn = p.reshape(3, 4)

    # 去掉归一化：x_n = Pn X_n，且 x_n = T2 x，X_n = T3 X
    T2i = np.linalg.inv(T2)
    P = T2i @ Pn @ T3

    # 已知 K：M = K⁻¹ P = [A | b] = λ[R | t]
    try:
        M = np.linalg.inv(intrinsics.to_matrix()) @ P
    except np.linalg.LinAlgError:
        return None

    A3 = M[:, :3]
    lam = float(np.linalg.norm(A3[:, 0]))
    if lam < 1e-12:
        return None
    r0 = A3[:, 0] / lam
    r1 = A3[:, 1] / lam
    t = M[:, 3] / lam

    # 前两列正交归一，第三列由叉积补出。
    #
    # **不能**直接拿 A3 去做 orthonormalize：当场地点几乎总是共面（z=0）时，
    # DLT 里乘 z 的那一列完全不受数据约束，SVD 会给它一个任意值；把它一起
    # 正交化，会把本来正确的 r0/r1 也拽偏。叉积补第三列则是精确的——
    # 旋转矩阵恒有 c2 = c0 × c1。
    n0 = float(np.linalg.norm(r0))
    if n0 < 1e-12:
        return None
    r0 = r0 / n0
    r1 = r1 - float(r1 @ r0) * r0
    n1 = float(np.linalg.norm(r1))
    if n1 < 1e-12:  # 前两列平行 → 点共线，无解
        return None
    r1 = r1 / n1
    r2 = np.cross(r0, r1)

    # 手性：R 与 t 同时取反仍是同一组投影，但只有一组把点放在相机前方。
    # 共面点下 det(A3) 不可信（第三列是噪声），所以用点在不在前方来定符号。
    R = np.stack([r0, r1, r2], axis=1)
    if float(np.mean((world @ R.T + t)[:, 2])) < 0.0:
        R = np.stack([-r0, -r1, r2], axis=1)  # 两次取反，det 仍为 +1
        t = -t
    return R, t


def _prepare(
    world_pts: np.ndarray, pixels: np.ndarray, intrinsics: Intrinsics
) -> tuple[np.ndarray, np.ndarray, Intrinsics]:
    """去畸变 + 换成理想内参。PnP 只接受理想针孔模型。

    若把含畸变的像素直接喂给 DLT，畸变会在重投影时被**再施加一次**，
    结果是位姿看起来对、残差却大得离谱。
    """
    world = np.asarray(world_pts, dtype=np.float64).reshape(-1, 3)
    px = undistort_pixels(intrinsics, pixels)
    return world, px, intrinsics.without_distortion()


def _reprojection_errors_ideal(
    pose: Pose, world: np.ndarray, px_ideal: np.ndarray, ideal: Intrinsics
) -> np.ndarray:
    proj, valid = project_points(ideal, pose.world_to_camera(world))
    err = np.full(len(proj), np.inf, dtype=np.float64)
    err[valid] = np.linalg.norm(proj[valid] - px_ideal[valid], axis=1)
    return err


def reprojection_errors(
    pose: Pose, world_pts: np.ndarray, pixels: np.ndarray, intrinsics: Intrinsics
) -> np.ndarray:
    """逐点重投影误差（像素）。``pixels`` 是原始含畸变像素。"""
    world, px, ideal = _prepare(world_pts, pixels, intrinsics)
    return _reprojection_errors_ideal(pose, world, px, ideal)


def solve_pnp_dlt(
    world_pts: np.ndarray, pixels: np.ndarray, intrinsics: Intrinsics
) -> Pose | None:
    """纯 DLT 解，用于 RANSAC 的最小样本。``pixels`` 含畸变。"""
    world, px, ideal = _prepare(world_pts, pixels, intrinsics)
    out = _dlt(world, px, ideal)
    if out is None:
        return None
    return Pose(*out)


def refine_pose(
    pose: Pose,
    world_pts: np.ndarray,
    pixels: np.ndarray,
    intrinsics: Intrinsics,
    *,
    max_nfev: int = 200,
) -> Pose:
    """用全部对应点做最小二乘精化（Levenberg-Marquardt）。"""
    from scipy.optimize import least_squares  # 局部导入：只有精化才需要 scipy

    world, px, ideal = _prepare(world_pts, pixels, intrinsics)
    x0 = np.concatenate([matrix_to_rodrigues(pose.R), pose.t])

    def residuals(x: np.ndarray) -> np.ndarray:
        R = rodrigues_to_matrix(x[:3])
        t = x[3:]
        proj, valid = project_points(ideal, world_to_camera(world, R, t))
        res = np.where(valid[:, None], proj - px, 1e3)
        return res.reshape(-1)

    try:
        sol = least_squares(residuals, x0, method="lm", max_nfev=max_nfev)
    except Exception:  # noqa: BLE001 —— 精化失败就退回原解，不吞掉主流程
        return pose
    return Pose(orthonormalize(rodrigues_to_matrix(sol.x[:3])), sol.x[3:])


def solve_pnp_ransac(
    world_pts: np.ndarray,
    pixels: np.ndarray,
    intrinsics: Intrinsics,
    *,
    thresh_px: float = 6.0,
    max_iters: int = 300,
    min_inliers: int = 6,
    seed: int = 0,
    refine: bool = True,
) -> PnPResult | None:
    """RANSAC + DLT 求解外参。``pixels`` 是**原始含畸变**像素。

    点太少或解不出来返回 None。
    """
    world, px, ideal = _prepare(world_pts, pixels, intrinsics)
    n = len(world)
    if n < min_inliers or len(px) != n:
        return None

    rng = np.random.default_rng(seed)
    # 用**最小样本**（6 点 = 12 个方程，刚好盖住 [R|t] 的 11 个自由度）。
    #
    # 取更多点不但没有好处（DLT 是线性解，超定样本不会更抗噪），还会把
    # "抽到全内点子集"的概率压垮：10 个点里 2 个外点时，抽 8 个点全中的
    # 概率是 C(8,8)/C(10,8)=0 —— RANSAC 会**必然失败**，而这恰恰是它该
    # 起作用的场合。抽 6 个则是 C(8,6)/C(10,6)≈13%，几百次迭代绰绰有余。
    sample_size = 6
    best_inliers: np.ndarray | None = None
    best_err: np.ndarray | None = None
    best_pose: Pose | None = None
    iters = 0

    for iters in range(1, max_iters + 1):
        idx = rng.choice(n, size=sample_size, replace=False)
        out = _dlt(world[idx], px[idx], ideal)
        if out is None:
            continue
        pose = Pose(*out)
        err = _reprojection_errors_ideal(pose, world, px, ideal)
        inliers = err < thresh_px
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers, best_err, best_pose = inliers, err, pose
            if inliers.sum() == n:  # 全内点，提前收工
                break

    if best_pose is None or best_inliers is None or best_inliers.sum() < min_inliers:
        return None

    pose = best_pose
    if refine:
        pose = refine_pose(pose, world[best_inliers], px[best_inliers], ideal)
        best_err = _reprojection_errors_ideal(pose, world, px, ideal)
        best_inliers = best_err < thresh_px
        # 精化后若内点数骤减，说明被离群点带偏，退回粗解
        if best_inliers.sum() < min_inliers:
            pose = best_pose
            best_err = _reprojection_errors_ideal(pose, world, px, ideal)
            best_inliers = best_err < thresh_px

    err_in = best_err[best_inliers]
    return PnPResult(
        pose=pose,
        inliers=best_inliers,
        reproj_errors_px=best_err,
        rms_px=float(np.sqrt((err_in ** 2).mean())),
        max_px=float(err_in.max()),
        num_iterations=iters,
    )
