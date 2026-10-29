#!/usr/bin/env python3
"""生成合成 fixture。改动生成参数后重跑本文件即可复现。

    python3 tests/fixtures/_generate.py

产出：
  field_mesh_simple.ply   28×15m 地面 + 3cm 台阶 + 一面立柱，**Z 向上**
  field_mesh_yup.ply      同一场地但 **Y 向上**，用于测轴向检测
  keypoints_synth.json    10 个地面关键点 + 投影出的像素 + 生成时用的相机/位姿

关键点像素是用**下面这套相机与位姿**投影出来的，并且把相机与位姿一起写进
JSON，这样 fixture 是自描述的：测试不需要在代码里重复一遍常量。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jg_radar.field import Keypoint, KeypointSet, RawMesh, save_ply  # noqa: E402
from jg_radar.transform import CameraModel, Intrinsics, look_at  # noqa: E402

HERE = Path(__file__).resolve().parent

FIELD_X, FIELD_Y = 2800.0, 1500.0
STEP_Z = 3.0
WALL_X, WALL_H = 2000.0, 150.0

K = [[3478.0, 0.0, 2012.0], [0.0, 3478.0, 1518.0], [0.0, 0.0, 1.0]]
DIST = [-0.12, 0.03, 1e-4, -1e-4, 0.0]
WIDTH, HEIGHT = 4024, 3036
EYE = [1400.0, -500.0, 360.0]
TARGET = [1400.0, 750.0, 0.0]


def quad(a, b, c, d) -> tuple[list, list[list[int]]]:
    """一个四边形 → 2 个三角形。顶点顺序保证法线一致。"""
    return [a, b, c, d], [[0, 1, 2], [0, 2, 3]]


def build_field() -> RawMesh:
    """地面 + 台阶 + 立柱。Z 向上，单位 cm，已落在场地系。"""
    verts: list[list[float]] = []
    faces: list[list[int]] = []

    def add(pts, idxs):
        base = len(verts)
        verts.extend(pts)
        faces.extend([[i + base for i in t] for t in idxs])

    # 地面：z=0 整场
    p, f = quad([0, 0, 0], [FIELD_X, 0, 0], [FIELD_X, FIELD_Y, 0], [0, FIELD_Y, 0])
    add(p, f)
    # 台阶：z=3cm 的一块高地
    p, f = quad([1000, 400, STEP_Z], [1800, 400, STEP_Z],
                [1800, 1100, STEP_Z], [1000, 1100, STEP_Z])
    add(p, f)
    # 台阶侧面（近垂直 → 应被归为遮挡面）
    p, f = quad([1000, 400, 0], [1800, 400, 0],
                [1800, 400, STEP_Z], [1000, 400, STEP_Z])
    add(p, f)
    # 立柱：一面竖墙，近垂直 → 遮挡面
    p, f = quad([WALL_X, 200, 0], [WALL_X, 500, 0],
                [WALL_X, 500, WALL_H], [WALL_X, 200, WALL_H])
    add(p, f)

    return RawMesh(np.asarray(verts, dtype=np.float64),
                   np.asarray(faces, dtype=np.int64), "synthetic")


def to_y_up(mesh: RawMesh) -> RawMesh:
    """把 Z-up 网格转成 Y-up，模拟 Open3D 导出的那种约定。

    旋转：new = (x, z, -y)，即 Y 轴变成原来的 Z 轴，det = +1。
    """
    v = mesh.vertices
    return RawMesh(np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1), mesh.faces,
                   "synthetic-yup")


def build_keypoints() -> dict:
    """在**地面**上取 10 个可命名特征点，投影成像素。

    全部取 z=0：射线只能与地面求交，高处点的真值没法用平面往返检验。
    这也让这组点是**共面**的——而共面恰恰是最难的一种情况（见 extrinsics
    里 DLT 的零空间讨论），现实里的场地特征点也几乎全是共面的。

    点的位置必须落在相机真正看得见的那块梯形里。12mm 镜头架在端线后
    5m 时，近端只看得见 x∈[1100,1700]、远端才展开到 x∈[325,2475]——
    这不是凑数，是"单机 12mm 盖不全场"的实测体现。
    """
    intr = Intrinsics.from_matrix(K, DIST)
    camera = CameraModel(intr, WIDTH, HEIGHT)
    R, t = look_at(EYE, TARGET)

    specs = [
        ("far_left", (700.0, 1400.0, 0.0)),
        ("far_center", (1400.0, 1400.0, 0.0)),
        ("far_right", (2100.0, 1400.0, 0.0)),
        ("mid_left", (1000.0, 750.0, 0.0)),
        ("center_circle", (1400.0, 750.0, 0.0)),
        ("mid_right", (1800.0, 750.0, 0.0)),
        ("deep_left", (800.0, 1000.0, 0.0)),
        ("deep_right", (2000.0, 1000.0, 0.0)),
        ("near_left", (1200.0, 200.0, 0.0)),
        ("near_right", (1600.0, 200.0, 0.0)),
    ]

    world = np.asarray([s[1] for s in specs], dtype=np.float64)
    pixels, valid = camera.project_world(world, R, t)
    assert valid.all(), "合成关键点必须全部在相机前方"
    # 只断言"在相机前方"是不够的：球面外的一万个像素也满足它。
    # 真正要保证的是**落在画面里**，否则标定工具拿到的是点不出来的点。
    inside = camera.in_image(pixels, margin=50.0)
    assert inside.all(), (
        "有合成关键点落在画面外，像素为 "
        f"{np.round(pixels[~inside], 1).tolist()}；请把它们移进可视梯形内"
    )

    points = [
        Keypoint(name=name, position_cm=pos, pixel=(float(px[0]), float(px[1])))
        for (name, pos), px in zip(specs, pixels)
    ]
    kps = KeypointSet(tuple(points))
    doc = kps.to_dict()
    doc["camera"] = {"K": K, "dist": DIST, "width": WIDTH, "height": HEIGHT}
    doc["pose"] = {"eye": EYE, "target": TARGET}
    doc["note"] = "由 tests/fixtures/_generate.py 生成，请勿手改。"
    return doc


def main() -> None:
    field = build_field()
    save_ply(HERE / "field_mesh_simple.ply", field)
    print(f"✓ field_mesh_simple.ply  {field.num_vertices} 顶点 / {field.num_faces} 面")

    save_ply(HERE / "field_mesh_yup.ply", to_y_up(field))
    print("✓ field_mesh_yup.ply")

    (HERE / "keypoints_synth.json").write_text(
        json.dumps(build_keypoints(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("✓ keypoints_synth.json")


if __name__ == "__main__":
    main()
