"""针孔 + Brown-Conrady 畸变模型。**纯 numpy，不依赖 OpenCV。**

为什么不用 cv2：
  1. 运行时要保持零重依赖（本机连 opencv-python 都没装）；
  2. 投影链是精度核心，自己实现才能对每一步写测试；
  3. 公式与 OpenCV 一致，必要时可用 cv2 交叉验证（见 tests 的 gpu/可选用例）。

坐标约定（全项目统一）：
  - **场地系**：x ∈ [0, 2800], y ∈ [0, 1500], z 向上，单位 cm
  - **相机系**：OpenCV 惯例——x 右、y 下、z 向前（指向场景）
  - **像素系**：左上原点，整数坐标即像素中心

畸变参数顺序与 OpenCV 相同：k1, k2, p1, p2, k3。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# 迭代去畸变的默认次数。OpenCV undistortPoints 用 5 次，这里保持一致。
UNDISTORT_ITERS = 5


@dataclass(frozen=True)
class Intrinsics:
    """相机内参。"""

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    @classmethod
    def from_matrix(
        cls, K: Sequence[Sequence[float]], dist: Sequence[float] | None = None
    ) -> "Intrinsics":
        """从 3×3 内参矩阵与 5 元畸变向量构造。"""
        k = np.asarray(K, dtype=np.float64)
        if k.shape != (3, 3):
            raise ValueError(f"K 必须是 3×3，得到 {k.shape}")
        d = list(dist) if dist is not None else [0.0] * 5
        if len(d) != 5:
            raise ValueError(f"dist 必须是 5 个元素（k1 k2 p1 p2 k3），得到 {len(d)}")
        return cls(
            fx=float(k[0, 0]), fy=float(k[1, 1]),
            cx=float(k[0, 2]), cy=float(k[1, 2]),
            k1=float(d[0]), k2=float(d[1]), p1=float(d[2]), p2=float(d[3]), k3=float(d[4]),
        )

    def to_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def dist_vector(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

    @property
    def has_distortion(self) -> bool:
        return bool(np.any(self.dist_vector() != 0.0))

    def without_distortion(self) -> "Intrinsics":
        """去掉畸变、保留 fx/fy/cx/cy。

        PnP 一类只接受**理想针孔模型**的算法必须先去掉畸变，否则畸变会被
        重复施加一次（这是本项目早期真实踩过的坑）。
        """
        return Intrinsics(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)


def distort_normalized(intr: Intrinsics, xy: np.ndarray) -> np.ndarray:
    """归一化平面坐标 → 畸变后的像素坐标。输入 (N,2)。"""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    radial = 1.0 + r2 * (intr.k1 + r2 * (intr.k2 + r2 * intr.k3))
    xd = x * radial + 2.0 * intr.p1 * x * y + intr.p2 * (r2 + 2.0 * x * x)
    yd = y * radial + intr.p1 * (r2 + 2.0 * y * y) + 2.0 * intr.p2 * x * y
    return np.stack([intr.fx * xd + intr.cx, intr.fy * yd + intr.cy], axis=-1)


def project_points(intr: Intrinsics, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """相机系 3D 点 → 像素，返回 ``(pixels, valid)``。

    ``z <= 0`` 的点（在相机背后）标为无效，像素填 NaN——静默把背后的点投到
    画面上是最难查的一类 bug。
    """
    pts = np.asarray(pts_cam, dtype=np.float64).reshape(-1, 3)
    z = pts[:, 2]
    valid = z > 1e-9
    pixels = np.full((len(pts), 2), np.nan, dtype=np.float64)
    if np.any(valid):
        pixels[valid] = distort_normalized(intr, pts[valid, :2] / z[valid, None])
    return pixels, valid


def undistort_to_normalized(
    intr: Intrinsics, pixels: np.ndarray, iters: int = UNDISTORT_ITERS
) -> np.ndarray:
    """像素 → 去畸变的归一化平面坐标 (N,2)。

    不动点迭代，与 OpenCV ``undistortPoints`` 同一套做法：
    从畸变坐标出发，反复把"当前估计的畸变量"减掉。
    """
    px = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    xd = (px[:, 0] - intr.cx) / intr.fx
    yd = (px[:, 1] - intr.cy) / intr.fy

    x, y = xd.copy(), yd.copy()
    for _ in range(int(iters)):
        r2 = x * x + y * y
        radial = 1.0 + r2 * (intr.k1 + r2 * (intr.k2 + r2 * intr.k3))
        dx = 2.0 * intr.p1 * x * y + intr.p2 * (r2 + 2.0 * x * x)
        dy = intr.p1 * (r2 + 2.0 * y * y) + 2.0 * intr.p2 * x * y
        # 保护：radial 接近 0 时（极端畸变）不更新，避免除零放大误差
        safe = np.abs(radial) > 1e-12
        x = np.where(safe, (xd - dx) / radial, x)
        y = np.where(safe, (yd - dy) / radial, y)
    return np.stack([x, y], axis=-1)


def undistort_pixels(
    intr: Intrinsics, pixels: np.ndarray, iters: int = UNDISTORT_ITERS
) -> np.ndarray:
    """像素 → 去畸变后的像素（保持同一 fx/fy/cx/cy）。"""
    xy = undistort_to_normalized(intr, pixels, iters)
    return np.stack([intr.fx * xy[:, 0] + intr.cx, intr.fy * xy[:, 1] + intr.cy], axis=-1)


class CameraModel:
    """一台已标定的相机。"""

    def __init__(
        self,
        intrinsics: Intrinsics,
        width: int,
        height: int,
        *,
        undistort_iters: int = UNDISTORT_ITERS,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("图像尺寸必须为正")
        self.intrinsics = intrinsics
        self.width = int(width)
        self.height = int(height)
        self.undistort_iters = int(undistort_iters)

    # ------------------------------------------------------------------
    # 正向：归一化坐标 → 像素
    # ------------------------------------------------------------------
    def distort_normalized(self, xy: np.ndarray) -> np.ndarray:
        """归一化平面坐标 → 畸变后的像素坐标。输入 (N,2)。"""
        return distort_normalized(self.intrinsics, xy)

    def project(self, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """相机系 3D 点 → 像素，返回 ``(pixels, valid)``。"""
        return project_points(self.intrinsics, pts_cam)

    def project_world(
        self, pts_world: np.ndarray, R: np.ndarray, t: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """场地系 3D 点 → 像素。"""
        return self.project(world_to_camera(pts_world, R, t))

    # ------------------------------------------------------------------
    # 反向：像素 → 归一化坐标（去畸变）
    # ------------------------------------------------------------------
    def undistort_to_normalized(self, pixels: np.ndarray) -> np.ndarray:
        """像素 → 去畸变的归一化平面坐标 (N,2)。"""
        return undistort_to_normalized(self.intrinsics, pixels, self.undistort_iters)

    def undistort_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """像素 → 去畸变后的像素（保持同一内参）。"""
        return undistort_pixels(self.intrinsics, pixels, self.undistort_iters)

    # ------------------------------------------------------------------
    # 像素 → 射线
    # ------------------------------------------------------------------
    def rays_camera(self, pixels: np.ndarray) -> np.ndarray:
        """像素 → 相机系单位方向向量 (N,3)。"""
        xy = self.undistort_to_normalized(pixels)
        d = np.concatenate([xy, np.ones((len(xy), 1), dtype=np.float64)], axis=1)
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def rays_world(
        self, pixels: np.ndarray, R: np.ndarray, t: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """像素 → 场地系射线 ``(origin, direction)``。

        origin 是相机光心（N 份），direction 已单位化。
        """
        d_cam = self.rays_camera(pixels)
        d_world = d_cam @ np.asarray(R, dtype=np.float64)
        """R 把场地向量转到相机系，故相机向量转回场地系要乘 Rᵀ——
        对行向量而言就是右乘 R。"""
        origin = camera_center(R, t)
        origins = np.broadcast_to(origin, d_world.shape).copy()
        return origins, d_world

    # ------------------------------------------------------------------
    def in_image(self, pixels: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """像素是否落在图像范围内。"""
        px = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        return (
            (px[:, 0] >= -margin) & (px[:, 0] <= self.width - 1 + margin)
            & (px[:, 1] >= -margin) & (px[:, 1] <= self.height - 1 + margin)
        )


# ----------------------------------------------------------------------
# 外参工具（放在本模块底部，供 rays_world 使用，也对外导出）
# ----------------------------------------------------------------------
def world_to_camera(pts_world: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """场地系 3D 点 → 相机系。``Xc = R @ Xw + t``。"""
    pts = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return pts @ R.T + t


def camera_to_world(pts_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """相机系 3D 点 → 场地系。"""
    pts = np.asarray(pts_cam, dtype=np.float64).reshape(-1, 3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return (pts - t) @ R


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """相机光心在场地系中的位置（cm）。"""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    return -R.T @ t
