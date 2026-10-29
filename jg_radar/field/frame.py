"""场地坐标系约定与网格摆放。

## 三个坐标系

1. **场地系**（本项目内部唯一使用的坐标系）
   ``x ∈ [0, 2800]``、``y ∈ [0, 1500]``、``z`` 向上，单位 **cm**。
   x=0 是红方基地一侧，x=2800 是蓝方基地一侧；y=0/1500 是两条边线。

2. **网格系**：PLY/STL 文件里的原始坐标。**原点任意、朝向任意、单位任意。**
   实测 T 家仓库的 ``field/`` 目录下就同时存在两种约定：
   ``RMUC2026_National.PLY`` 是 **Y 向上**（法线分析：±Y 占 75% 面积），
   而 ``RM2026_region_field_v1.0_tri.ply`` 是 **Z 向上**（Blender 导出）。
   所以这里绝不"猜一个"就用——见 ``detect_up_axis`` 与 ``suggest_mesh_to_field``，
   最终以 ``validate_alignment`` 的残差为准。

3. **阵营系**：裁判系统按**红方视角**定义场地系。蓝方视角是绕场地中心
   的点对称：``(x, y) → (2800 − x, 1500 − y)``。
   来源：TCR ``solidwork2uwb.py`` 的红 ``(x+14, y+7.5)`` / 蓝 ``(14−x, 7.5−y)``
   （米），两者相差的正是这个点对称。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jg_radar.field.mesh import face_areas, face_normals
from jg_radar.types import FIELD_MAX_X_CM, FIELD_MAX_Y_CM

FIELD_CENTER_CM = (FIELD_MAX_X_CM / 2.0, FIELD_MAX_Y_CM / 2.0)
"""场地中心 (1400, 750) cm。"""

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


# ======================================================================
# 阵营
# ======================================================================
def mirror_to_side(pos_cm, side: str):
    """把一个场地系坐标转换到指定阵营的视角。

    ``side="R"`` 原样返回（场地系本就按红方定义）；``side="B"`` 做绕场地中心
    的点对称。红蓝双方看到的"我方基地"都在 x=0 一侧，靠的就是这一步。
    """
    x, y = float(pos_cm[0]), float(pos_cm[1])
    if side.upper() == "R":
        return (x, y)
    if side.upper() == "B":
        return (FIELD_MAX_X_CM - x, FIELD_MAX_Y_CM - y)
    raise ValueError(f"未知阵营 {side!r}，应为 'R' 或 'B'")


def unmirror_to_field(pos_cm, side: str):
    """``mirror_to_side`` 的逆（点对称是自逆的，但保留语义化名字）。"""
    return mirror_to_side(pos_cm, side)


def in_field(x: float, y: float, margin: float = 0.0) -> bool:
    return (
        -margin <= x <= FIELD_MAX_X_CM + margin
        and -margin <= y <= FIELD_MAX_Y_CM + margin
    )


def clip_to_field(x: float, y: float) -> tuple[float, float]:
    return (
        float(np.clip(x, 0.0, FIELD_MAX_X_CM)),
        float(np.clip(y, 0.0, FIELD_MAX_Y_CM)),
    )


# ======================================================================
# 网格朝向
# ======================================================================
def rotation_from_axes(
    up: str,
    forward: str,
    *,
    up_sign: int = 1,
    forward_sign: int = 1,
) -> np.ndarray:
    """构造「网格系 → 场地系」的 3×3 旋转。

    ``up`` 是网格里朝上的轴，``forward`` 是网格里指向**场地 x 正方向**
    （蓝方基地）的轴。第三个轴由右手系自动定出，因此行列式恒为 +1，
    不会出现镜像翻转（镜像会让所有面法线反向，遮挡判断随之失效）。
    """
    if up not in AXIS_INDEX or forward not in AXIS_INDEX:
        raise ValueError(f"轴名必须是 x/y/z，得到 up={up!r} forward={forward!r}")
    if up == forward:
        raise ValueError("up 与 forward 不能是同一根轴")

    f = np.zeros(3)
    f[AXIS_INDEX[forward]] = float(np.sign(forward_sign))
    u = np.zeros(3)
    u[AXIS_INDEX[up]] = float(np.sign(up_sign))

    r = np.cross(u, f)
    return np.stack([f, r, u], axis=0)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float(np.median(values))
    return float(v[min(int(np.searchsorted(cw, cw[-1] * 0.5)), len(v) - 1)])


def detect_up_axis(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    horizontal_cos: float = 0.7,
    planar_band: float = 0.02,
) -> tuple[str, float]:
    """猜哪根轴朝上。返回 ``(轴名, 置信度)``。

    判据是「**法线朝这根轴 + 且共面**的面积占比」——地面是一大块**平**的东西，
    两个条件缺一不可。

    只看"哪根轴的法线面积最多"会翻车：实测 ``RM2026_region_field_v1.0_tri.ply``
    里围栏墙面的面积比地面还大，纯面积投票会把**水平方向**选成朝上。加上共面
    条件后，墙（分布在多个平面上）自然出局。

    **这只是建议**——最终必须由 ``validate_alignment`` 的残差确认。
    """
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    n = face_normals(v, f)
    a = face_areas(v, f)
    total = float(a.sum())
    if not len(f) or total <= 0:
        return "z", 0.0

    centers = v[f].mean(axis=1)
    # 用 1–99 分位跨度而不是极差：网格里常有几根远处的杂散杆件，极差会被它们撑爆
    lo = np.percentile(v, 1.0, axis=0)
    hi = np.percentile(v, 99.0, axis=0)

    best_axis, best_score = "z", 0.0
    for i, axis in enumerate("xyz"):
        facing = np.abs(n[:, i]) > horizontal_cos
        if not facing.any() or a[facing].sum() <= 0:
            continue
        med = _weighted_median(centers[facing, i], a[facing])
        tol = planar_band * float(hi[i] - lo[i]) + 1e-9
        coplanar = facing & (np.abs(centers[:, i] - med) <= tol)
        score = float(a[coplanar].sum()) / total
        if score > best_score:
            best_axis, best_score = axis, score
    return best_axis, best_score


@dataclass(frozen=True)
class MeshPlacement:
    """网格摆放结果，连同它的依据。"""

    matrix: np.ndarray
    """4×4 齐次变换，网格系 → 场地系（cm）。"""
    up_axis: str
    forward_axis: str
    unit_scale: float
    confidence: float
    """``detect_up_axis`` 的得分：法线朝该轴且共面的面积占比。偏低就说明
    这个网格不像场地（或地面被拆得很碎），别信它的建议。"""
    pre_bounds: tuple[np.ndarray, np.ndarray]
    post_bounds: tuple[np.ndarray, np.ndarray]

    def describe(self) -> str:
        lo, hi = self.post_bounds
        return (
            f"网格摆放：up={self.up_axis} forward={self.forward_axis} "
            f"scale={self.unit_scale:g} 置信度={self.confidence:.2f} | "
            f"落位后 x[{lo[0]:.0f},{hi[0]:.0f}] y[{lo[1]:.0f},{hi[1]:.0f}] "
            f"z[{lo[2]:.0f},{hi[2]:.0f}]"
        )


def suggest_mesh_to_field(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    up_axis: str | None = None,
    forward_axis: str | None = None,
    up_sign: int = 1,
    forward_sign: int = 1,
    unit_scale: float | None = None,
    center_on_field: bool = True,
    ground_z_cm: float = 0.0,
) -> MeshPlacement:
    """给一个**建议**的网格 → 场地变换。

    步骤：选轴 → 旋转 → 按包围盒把水平中心对到场地中心、把最低点对到地面高度。
    这是标定的起点而不是终点：先用它跑通链路，再用关键点残差把它拧准。
    """
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)

    confidence = 1.0
    if up_axis is None:
        up_axis, confidence = detect_up_axis(v, f)

    # forward 默认取剩下两根水平轴里较长的那根（28m 那条）
    if forward_axis is None:
        horizontal = [ax for ax in "xyz" if ax != up_axis]
        extents = {ax: v[:, AXIS_INDEX[ax]].ptp() for ax in horizontal}
        forward_axis = max(extents, key=extents.get)

    if unit_scale is None:
        # 用最长水平跨度猜单位：28m 场地 → 若跨度 ~28 则是米，~2800 则是厘米
        span = v[:, AXIS_INDEX[forward_axis]].ptp()
        unit_scale = 100.0 if span < 200.0 else 1.0

    R = rotation_from_axes(up_axis, forward_axis, up_sign=up_sign, forward_sign=forward_sign)

    scaled = v * unit_scale
    rotated = scaled @ R.T

    offset = np.zeros(3)
    lo, hi = rotated.min(axis=0), rotated.max(axis=0)
    if center_on_field:
        offset[0] = FIELD_CENTER_CM[0] - (lo[0] + hi[0]) * 0.5
        offset[1] = FIELD_CENTER_CM[1] - (lo[1] + hi[1]) * 0.5
    offset[2] = ground_z_cm - lo[2]

    matrix = np.eye(4)
    matrix[:3, :3] = R
    matrix[:3, 3] = offset

    placed = rotated + offset
    return MeshPlacement(
        matrix=matrix,
        up_axis=up_axis,
        forward_axis=forward_axis,
        unit_scale=float(unit_scale),
        confidence=float(confidence),
        pre_bounds=(lo, hi),
        post_bounds=(placed.min(axis=0), placed.max(axis=0)),
    )


def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """对 (N,3) 点施加 4×4 齐次变换。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    m = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    return (np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ m.T)[:, :3]


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    """刚体变换求逆。"""
    m = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    R = m[:3, :3]
    t = m[:3, 3]
    inv = np.eye(4)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv
