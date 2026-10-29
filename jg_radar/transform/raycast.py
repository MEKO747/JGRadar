"""射线求交：地面平面与三角网格。**纯 numpy。**

两种用途：
  - **地面平面模式**（默认）：与 z = const 求交。消除了"把高处装甲板中心
    投到地面后方"的系统偏差。
  - **网格模式**：射线-三角形求交（Möller–Trumbore），用于高地/坡道。

命中障碍物子网格时**标记为遮挡，绝不静默定位**——把一个撞在立柱上的射线
当成地面交点，是这类系统最隐蔽的错误。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PARALLEL_EPS = 1e-12

#: 求交过程中每对（射线 × 三角形）大约要开的中间数组字节数。
#: ``intersect_triangles`` 同时持有 pvec / tvec / qvec 三个 ``(N,M,3)`` 和
#: 几个 ``(N,M)``，量级就是这么多。用它把峰值内存压到一个可配的上限以内。
_BYTES_PER_PAIR = 112.0


@dataclass(frozen=True)
class PlaneHit:
    """射线与水平面的交点。"""

    points_cm: np.ndarray
    """(N,3) 交点，未命中处为 NaN。"""
    t: np.ndarray
    """沿射线的参数距离。"""
    valid: np.ndarray
    """(N,) 是否命中（方向不平行于平面且交点在射线正方向）。"""


def intersect_plane(
    origins: np.ndarray, directions: np.ndarray, plane_z: float
) -> PlaneHit:
    """射线与水平面 z = plane_z 求交。

    只接受 ``t > 0`` 的交点：交点落在相机背后是没有意义的。
    """
    o = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    d = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    if len(o) != len(d):
        raise ValueError(f"origin 与 direction 数量不一致：{len(o)} vs {len(d)}")

    dz = d[:, 2]
    parallel = np.abs(dz) < PARALLEL_EPS
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (float(plane_z) - o[:, 2]) / dz
    valid = (~parallel) & np.isfinite(t) & (t > 0.0)

    pts = np.full((len(o), 3), np.nan, dtype=np.float64)
    pts[valid] = o[valid] + t[valid, None] * d[valid]
    t = np.where(valid, t, np.nan)
    return PlaneHit(points_cm=pts, t=t, valid=valid)


@dataclass(frozen=True)
class MeshHit:
    """射线与三角网格的最近交点。"""

    points_cm: np.ndarray
    t: np.ndarray
    face_index: np.ndarray
    valid: np.ndarray


def intersect_triangles(
    origins: np.ndarray,
    directions: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> np.ndarray:
    """Möller–Trumbore：每条射线对每个三角形求交。

    输入：``(N,3)`` 射线、``(M,3)`` 三角形顶点。
    返回：``(N,M)`` 参数距离，未命中为 NaN。背面剔除**关闭**——网格朝向
    未必一致，剔除会造成漏检。
    """
    o = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    d = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    a = np.asarray(v0, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(v1, dtype=np.float64).reshape(-1, 3)
    c = np.asarray(v2, dtype=np.float64).reshape(-1, 3)

    e1 = b - a                       # (M,3)
    e2 = c - a                       # (M,3)
    n = len(o)

    pvec = np.cross(d[:, None, :], e2[None, :, :])          # (N,M,3)
    det = np.einsum("mk,nmk->nm", e1, pvec)                 # (N,M)
    ok = np.abs(det) > PARALLEL_EPS

    inv_det = np.where(ok, 1.0 / np.where(ok, det, 1.0), np.nan)
    tvec = o[:, None, :] - a[None, :, :]                    # (N,M,3)
    u = np.einsum("nmk,nmk->nm", tvec, pvec) * inv_det
    ok &= (u >= -1e-9) & (u <= 1.0 + 1e-9)

    qvec = np.cross(tvec, e1[None, :, :])                   # (N,M,3)
    v = np.einsum("nk,nmk->nm", d, qvec) * inv_det
    ok &= (v >= -1e-9) & (u + v <= 1.0 + 1e-9)

    t = np.einsum("mk,nmk->nm", e2, qvec) * inv_det
    ok &= np.isfinite(t) & (t > 0.0)

    out = np.where(ok, t, np.nan)
    if n == 0:
        out = np.zeros((0, len(a)), dtype=np.float64)
    return out


def _nearest_hit(
    o: np.ndarray, d: np.ndarray, verts: np.ndarray, tris: np.ndarray,
    *, max_block_mb: float,
) -> tuple[np.ndarray, np.ndarray]:
    """分块求最近命中。返回 ``(t (N,), face_index (N,))``，未命中为 NaN / -1。

    **一次算完是这里唯一会崩的地方。** ``intersect_triangles`` 的中间数组是
    ``(N, M, 3)``：1200 万像素 × 1 万个三角形，光一个 pvec 就是 2.9 TB。
    所以按内存预算两级分块（射线块 × 三角形块），块内取最小、块间滚动比较。

    这是**纯记账**上的改动，不改数学：块大小取到能一次装下时，结果与不分块
    逐位相同；分块之后每个块各取一次 ``argmin`` 再滚动比较，取到的仍是全局
    最近的那个面。分块边界的唯一影响是**并列命中**时取哪一个面——那要求两个
    面到相机的距离严格相等，也就是两块重合的几何，本就无所谓取谁。
    """
    n_rays, n_tris = len(o), len(tris)
    best_t = np.full(n_rays, np.inf, dtype=np.float64)
    best_i = np.full(n_rays, -1, dtype=np.int64)

    if n_rays and n_tris:
        max_pairs = max(1.0, max(1.0, float(max_block_mb)) * 1e6 / _BYTES_PER_PAIR)
        # 两个方向都要够大：块太小的话 python 循环的固定开销会盖过计算本身，
        # 而循环次数 = 总配对数 / 每块的配对数，跟块怎么切无关。所以从预算
        # 反推一个几何平均边长，两个方向都按它起步。
        side = max(1, int(np.sqrt(max_pairs)))
        tri_chunk = min(n_tris, side)
        ray_chunk = min(n_rays, max(1, int(max_pairs / tri_chunk)))

        for t0 in range(0, n_tris, tri_chunk):
            block = tris[t0:t0 + tri_chunk]
            v0, v1, v2 = verts[block[:, 0]], verts[block[:, 1]], verts[block[:, 2]]
            for r0 in range(0, n_rays, ray_chunk):
                r1 = min(r0 + ray_chunk, n_rays)
                t_all = intersect_triangles(o[r0:r1], d[r0:r1], v0, v1, v2)
                # 整行全 NaN 会让 nanargmin 抛异常，而"这条射线一个面都没碰到"
                # 是完全正常的。换成 inf 再取最小，命中与否最后交给 isfinite。
                idx = np.where(np.isnan(t_all), np.inf, t_all).argmin(axis=1)
                local = t_all[np.arange(r1 - r0), idx]
                better = local < best_t[r0:r1]  # NaN 比较为假，天然被排除
                best_t[r0:r1] = np.where(better, local, best_t[r0:r1])
                best_i[r0:r1] = np.where(better, t0 + idx, best_i[r0:r1])

    valid_hit = np.isfinite(best_t)
    return (
        np.where(valid_hit, best_t, np.nan),
        np.where(valid_hit, best_i, -1).astype(np.int64),
    )


def intersect_mesh(
    origins: np.ndarray,
    directions: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    max_block_mb: float = 192.0,
) -> MeshHit:
    """射线与网格求交，取最近的命中面。

    ``mask`` 可选，``(M,)`` 布尔数组，用于只对子网格（如 ground）求交。

    ``max_block_mb`` 是求交过程中中间数组的内存预算，见 ``_nearest_hit``。
    它的存在是为了让"1200 万像素 × 全场网格"这种调用**跑得慢但跑得完**，
    而不是在开数组那一步就被 OOM 打死——查找表的网格模式正是这么大。
    """
    o = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    d = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tris = np.asarray(faces, dtype=np.int64).reshape(-1, 3)

    if mask is not None:
        tris = tris[np.asarray(mask, dtype=bool).reshape(-1)]

    t, face_index = _nearest_hit(o, d, verts, tris, max_block_mb=max_block_mb)
    valid = np.isfinite(t)

    pts = np.full((len(o), 3), np.nan, dtype=np.float64)
    pts[valid] = o[valid] + t[valid, None] * d[valid]
    return MeshHit(points_cm=pts, t=t, face_index=face_index, valid=valid)


