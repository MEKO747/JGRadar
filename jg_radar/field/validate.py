"""对齐体检：反投影残差 + 射线往返残差。

这是本项目的**回归门**。四家开源方案都没有对变换链做回归测试，导致所有
关于精度的讨论都无法验证；本模块把"标定到底准不准"变成一个每次改动都能
重跑的数字对。

两个指标必须同时看：
  - **反投影残差（px）**：场地点投到图像上，与人工点的像素差多少。
    它检验 R|t 与 K 是否自洽。
  - **往返残差（cm）**：图像点 → 射线 → 与地面/网格求交 → 回到场地系，
    与真值差多少。它检验**运行时真正走的那条路径**，包括畸变与求交。

只测反投影是不够的：反投影用的是针孔模型，而运行时要经过去畸变和射线求交，
后者才是真正会产生坐标的那条路。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from jg_radar.field.keypoints import KeypointSet
from jg_radar.transform.camera_model import CameraModel
from jg_radar.transform.extrinsics import Pose
from jg_radar.transform.raycast import intersect_plane


@dataclass(frozen=True)
class Residual:
    """单个关键点的残差。"""

    name: str
    reproj_px: float | None
    roundtrip_cm: float | None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "reproj_px": None if self.reproj_px is None else round(self.reproj_px, 4),
            "roundtrip_cm": (
                None if self.roundtrip_cm is None else round(self.roundtrip_cm, 4)
            ),
        }


@dataclass(frozen=True)
class AlignmentReport:
    """一次对齐体检的完整结果。"""

    residuals: tuple[Residual, ...]
    rms_px: float
    max_px: float
    rms_roundtrip_cm: float
    max_roundtrip_cm: float
    num_points: int
    num_calibratable: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def acceptable(
        self,
        *,
        rms_max_px: float = 3.0,
        max_max_px: float = 8.0,
        roundtrip_max_cm: float = 2.0,
    ) -> bool:
        """三个门限全过才算通过。"""
        return (
            self.rms_px <= rms_max_px
            and self.max_px <= max_max_px
            and self.max_roundtrip_cm <= roundtrip_max_cm
        )

    def failures(
        self,
        *,
        rms_max_px: float = 3.0,
        max_max_px: float = 8.0,
        roundtrip_max_cm: float = 2.0,
    ) -> list[str]:
        """逐条列出没过门限的项，便于直接打印。"""
        out: list[str] = []
        if self.rms_px > rms_max_px:
            out.append(f"反投影 RMS {self.rms_px:.3f}px > {rms_max_px}px")
        if self.max_px > max_max_px:
            out.append(f"反投影最大 {self.max_px:.3f}px > {max_max_px}px")
        if self.max_roundtrip_cm > roundtrip_max_cm:
            out.append(f"往返最大 {self.max_roundtrip_cm:.3f}cm > {roundtrip_max_cm}cm")
        return out

    def worst(self, n: int = 5) -> list[Residual]:
        """残差最大的前 n 个点——排查时先看这些。"""
        return sorted(
            self.residuals,
            key=lambda r: (
                -(r.roundtrip_cm or 0.0),
                -(r.reproj_px or 0.0),
            ),
        )[:n]

    def summary(self) -> str:
        lines = [
            f"对齐体检：{self.num_points} 个关键点（{self.num_calibratable} 个带像素）",
            f"  反投影   RMS {self.rms_px:8.3f} px   最大 {self.max_px:8.3f} px",
            f"  往返     RMS {self.rms_roundtrip_cm:8.3f} cm   最大 "
            f"{self.max_roundtrip_cm:8.3f} cm",
        ]
        bad = self.failures()
        lines.append("  结论：" + ("通过" if not bad else "未通过 —— " + "；".join(bad)))
        for r in self.worst(3):
            if r.roundtrip_cm is not None and r.roundtrip_cm > 0:
                lines.append(
                    f"    最差之一 {r.name}: 反投影 {r.reproj_px:.3f}px / "
                    f"往返 {r.roundtrip_cm:.3f}cm"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "rms_px": round(self.rms_px, 5),
            "max_px": round(self.max_px, 5),
            "rms_roundtrip_cm": round(self.rms_roundtrip_cm, 5),
            "max_roundtrip_cm": round(self.max_roundtrip_cm, 5),
            "num_points": self.num_points,
            "num_calibratable": self.num_calibratable,
            "residuals": [r.to_dict() for r in self.residuals],
            "diagnostics": self.diagnostics,
        }


def reprojection_residuals(
    camera: CameraModel, pose: Pose, kps: KeypointSet
) -> dict[str, float]:
    """逐点反投影残差（px）。只含有像素的点。"""
    usable = kps.calibratable
    if not usable:
        return {}
    world = np.asarray([p.position_cm for p in usable], dtype=np.float64)
    pixels = np.asarray([p.pixel for p in usable], dtype=np.float64)
    proj, valid = camera.project_world(world, pose.R, pose.t)

    out: dict[str, float] = {}
    for i, p in enumerate(usable):
        out[p.name] = (
            float(np.linalg.norm(proj[i] - pixels[i])) if valid[i] else float("inf")
        )
    return out


def roundtrip_residuals(
    camera: CameraModel,
    pose: Pose,
    kps: KeypointSet,
    *,
    plane_z: float = 0.0,
    ground_only: bool = True,
) -> dict[str, float]:
    """逐点"像素 → 射线 → 地面 → 场地系"的往返残差（cm）。

    默认只对**位于地面高度**的关键点计算：射线只能与地面求交，把高处点的
    真值拿来比是错的，会把报告污染成假阳性。
    """
    usable = [
        p
        for p in kps.calibratable
        if (not ground_only) or abs(p.position_cm[2] - plane_z) <= 1.0
    ]
    if not usable:
        return {}

    pixels = np.asarray([p.pixel for p in usable], dtype=np.float64)
    origins, dirs = camera.rays_world(pixels, pose.R, pose.t)
    hit = intersect_plane(origins, dirs, plane_z)

    out: dict[str, float] = {}
    for i, p in enumerate(usable):
        if not hit.valid[i]:
            out[p.name] = float("inf")
            continue
        d = hit.points_cm[i, :2] - np.asarray(p.position_cm[:2], dtype=np.float64)
        out[p.name] = float(np.linalg.norm(d))
    return out


def _rms(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        return float("inf")
    return float(np.sqrt((arr ** 2).mean())) if len(arr) else float("inf")


def _max(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        return float("inf")
    return float(arr.max()) if len(arr) else float("inf")


def validate_alignment(
    camera: CameraModel,
    pose: Pose,
    kps: KeypointSet,
    *,
    plane_z: float = 0.0,
    extra_diagnostics: dict[str, Any] | None = None,
) -> AlignmentReport:
    """跑一次完整对齐体检。"""
    reproj = reprojection_residuals(camera, pose, kps)
    trip = roundtrip_residuals(camera, pose, kps, plane_z=plane_z)

    names = [p.name for p in kps.points]
    residuals = tuple(
        Residual(n, reproj.get(n), trip.get(n)) for n in names
    )

    return AlignmentReport(
        residuals=residuals,
        rms_px=_rms(list(reproj.values())),
        max_px=_max(list(reproj.values())),
        rms_roundtrip_cm=_rms(list(trip.values())),
        max_roundtrip_cm=_max(list(trip.values())),
        num_points=len(kps),
        num_calibratable=len(kps.calibratable),
        diagnostics=dict(extra_diagnostics or {}),
    )


def coverage_report(
    camera: CameraModel,
    pose: Pose,
    *,
    plane_z: float = 0.0,
    grid_step_cm: float = 100.0,
) -> dict[str, Any]:
    """场地地面采样点中有多少落在画面内。

    用来回答"单机 12mm 到底盖不盖得住全场"这类问题，而不是靠焦距估算。
    """
    from jg_radar.types import FIELD_MAX_X_CM, FIELD_MAX_Y_CM

    xs = np.arange(0.0, FIELD_MAX_X_CM + 1.0, grid_step_cm)
    ys = np.arange(0.0, FIELD_MAX_Y_CM + 1.0, grid_step_cm)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, plane_z)], axis=1)

    pixels, valid = camera.project_world(pts, pose.R, pose.t)
    visible = valid & camera.in_image(pixels)
    return {
        "samples": int(len(pts)),
        "visible": int(visible.sum()),
        "coverage": float(visible.mean()),
        "grid_step_cm": grid_step_cm,
    }
