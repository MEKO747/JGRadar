"""标定与验证用的场地关键点。

一个关键点 = 一个**可命名、可在场地里量出来**的物理特征，带：
  - ``position_cm``：它在场地系里的 3D 坐标（必然有）
  - ``pixel``：在某一帧里点出来的像素坐标（标定/校验时才需要）

关键点的价值在于它把"标定对不对"变成一个可以**证伪**的数字。没有关键点，
外参就只是一组没人能质疑的浮点数。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from jg_radar.field.frame import apply_transform, mirror_to_side


@dataclass(frozen=True)
class Keypoint:
    """单个关键点。"""

    name: str
    position_cm: tuple[float, float, float]
    pixel: tuple[float, float] | None = None
    weight: float = 1.0
    note: str = ""

    def __post_init__(self) -> None:
        if len(self.position_cm) != 3:
            raise ValueError(f"{self.name}: position_cm 必须是 3 个分量")
        if self.pixel is not None and len(self.pixel) != 2:
            raise ValueError(f"{self.name}: pixel 必须是 2 个分量")
        if self.weight <= 0:
            raise ValueError(f"{self.name}: weight 必须为正")

    @property
    def is_calibratable(self) -> bool:
        """有像素才能参与标定。"""
        return self.pixel is not None

    @property
    def xy_cm(self) -> tuple[float, float]:
        return (self.position_cm[0], self.position_cm[1])

    def with_pixel(self, pixel) -> "Keypoint":
        return replace(self, pixel=(float(pixel[0]), float(pixel[1])))

    def mirrored(self, side: str) -> "Keypoint":
        """换到指定阵营视角（蓝方为点对称）。"""
        x, y = mirror_to_side(self.xy_cm, side)
        return replace(self, position_cm=(x, y, self.position_cm[2]))

    def moved_by(self, matrix: np.ndarray) -> "Keypoint":
        """施加网格 → 场地的变换。"""
        p = apply_transform(np.asarray(self.position_cm, dtype=np.float64)[None, :], matrix)[0]
        return replace(self, position_cm=(float(p[0]), float(p[1]), float(p[2])))

    def to_dict(self) -> dict:
        d: dict = {"name": self.name, "position_cm": list(self.position_cm)}
        if self.pixel is not None:
            d["pixel"] = list(self.pixel)
        if self.weight != 1.0:
            d["weight"] = self.weight
        if self.note:
            d["note"] = self.note
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Keypoint":
        if "name" not in d or "position_cm" not in d:
            raise ValueError(f"关键点缺少 name/position_cm：{d}")
        px = d.get("pixel")
        return cls(
            name=str(d["name"]),
            position_cm=tuple(float(v) for v in d["position_cm"]),  # type: ignore[arg-type]
            pixel=(float(px[0]), float(px[1])) if px else None,
            weight=float(d.get("weight", 1.0)),
            note=str(d.get("note", "")),
        )


@dataclass(frozen=True)
class KeypointSet:
    """一组关键点。"""

    points: tuple[Keypoint, ...]
    version: int = 1

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def __getitem__(self, idx):
        return self.points[idx]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.points)

    @property
    def calibratable(self) -> tuple[Keypoint, ...]:
        """有像素、可用于 PnP 的点。"""
        return tuple(p for p in self.points if p.is_calibratable)

    def get(self, name: str) -> Keypoint:
        for p in self.points:
            if p.name == name:
                return p
        raise KeyError(f"没有名为 {name!r} 的关键点；已有 {list(self.names)}")

    def world_array(self, only_calibratable: bool = False) -> np.ndarray:
        pts = self.calibratable if only_calibratable else self.points
        return np.asarray([p.position_cm for p in pts], dtype=np.float64).reshape(-1, 3)

    def pixel_array(self) -> np.ndarray:
        return np.asarray([p.pixel for p in self.calibratable], dtype=np.float64).reshape(-1, 2)

    def weights(self, only_calibratable: bool = False) -> np.ndarray:
        pts = self.calibratable if only_calibratable else self.points
        return np.asarray([p.weight for p in pts], dtype=np.float64)

    def subset(self, names) -> "KeypointSet":
        wanted = set(names)
        return KeypointSet(tuple(p for p in self.points if p.name in wanted), self.version)

    def mirrored(self, side: str) -> "KeypointSet":
        return KeypointSet(tuple(p.mirrored(side) for p in self.points), self.version)

    def moved_by(self, matrix: np.ndarray) -> "KeypointSet":
        return KeypointSet(tuple(p.moved_by(matrix) for p in self.points), self.version)

    def describe(self) -> str:
        return (
            f"KeypointSet({len(self.points)} 个点，其中 {len(self.calibratable)} 个带像素)"
        )

    # -- 序列化 ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {"version": self.version, "points": [p.to_dict() for p in self.points]}

    @classmethod
    def from_dict(cls, d: dict) -> "KeypointSet":
        pts = d.get("points")
        if not isinstance(pts, list):
            raise ValueError("关键点文件缺少 points 列表")
        return cls(tuple(Keypoint.from_dict(p) for p in pts), int(d.get("version", 1)))

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> "KeypointSet":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"关键点文件不存在：{p}")
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))


def assign_pixels(kps: KeypointSet, pixels: dict[str, tuple[float, float]]) -> KeypointSet:
    """按名字批量补上像素坐标（点标注工具的输出对接点）。"""
    missing = set(pixels) - set(kps.names)
    if missing:
        raise KeyError(f"像素字典里有不存在的关键点：{sorted(missing)}")
    return KeypointSet(
        tuple(
            p.with_pixel(pixels[p.name]) if p.name in pixels else p for p in kps.points
        ),
        kps.version,
    )
