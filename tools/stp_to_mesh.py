#!/usr/bin/env python3
"""STP/STEP → 三角网格（离线，非阻塞）。

**这不是本项目的关键路径。** T 家与 E 家仓库已附带三角化好的场地网格，
M2 用它们跑通链路即可；STP 转换是精度升级项，卡住它没有任何收益。

依赖（按优先级自动尝试，都没装就打印安装指引）：
  1. ``gmsh``      —— 自带 OCCT STEP reader，``gmsh -2 field.stp -o field.stl``
  2. ``cadquery`` / ``OCP`` —— ``importStep`` + ``tessellate``

STP 的坑（本文件存在的理由）：
  实测 ``UTF-8__RMUC2026_V2.0.0.stp`` 是 **1.25GB ASCII**、CREO 导出、
  毫米单位、**原点任意**。直接全量网格化既慢又没必要 —— 雷达只需要
  "地面接触点 + 主要障碍体"，不需要每一个螺栓。

用法：
    python3 tools/stp_to_mesh.py --stp field.stp --out field/rmuc2026.ply \
        --tolerance 5.0 --ground-only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jg_radar.field import (  # noqa: E402
    FieldMesh,
    RawMesh,
    detect_up_axis,
    load_mesh,
    save_ply,
    suggest_mesh_to_field,
)


def tessellate_with_gmsh(stp: Path, tolerance: float) -> RawMesh | None:
    """用 gmsh 命令行把 STP 曲面网格化成 STL，再读回来。"""
    if shutil.which("gmsh") is None:
        return None
    with tempfile.TemporaryDirectory() as td:
        stl = Path(td) / "meshed.stl"
        cmd = [
            "gmsh", "-2", str(stp), "-o", str(stl),
            "-format", "stl", "-clmin", str(tolerance * 0.5),
            "-clmax", str(tolerance * 2.0), "-v", "2",
        ]
        print(f"  运行：{' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  gmsh 失败：{proc.stderr.strip()[:400]}", file=sys.stderr)
            return None
        if not stl.exists():
            print("  gmsh 未产出文件", file=sys.stderr)
            return None
        return load_mesh(stl)


def tessellate_with_ocp(stp: Path, tolerance: float) -> RawMesh | None:
    """用 OCP（OCCT 的 Python 绑定）直接细分。"""
    try:
        from OCP.BRepMesh import BRepMesh_IncrementalMesh  # type: ignore
        from OCP.IFSelect import IFSelect_RetDone  # type: ignore
        from OCP.STEPControl import STEPControl_Reader  # type: ignore
        from OCP.TopAbs import TopAbs_FACE  # type: ignore
        from OCP.TopExp import TopExp_Explorer  # type: ignore
        from OCP.TopoDS import TopoDS  # type: ignore
    except ImportError:
        return None

    reader = STEPControl_Reader()
    if reader.ReadFile(str(stp)) != IFSelect_RetDone:
        print("  OCP 读取 STEP 失败", file=sys.stderr)
        return None
    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, tolerance, False, 0.5, True)

    verts: list[np.ndarray] = []
    faces: list[list[int]] = []
    base = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        loc = face.Location()
        tri = _face_triangulation(face)
        if tri is not None:
            pts, idxs = tri
            pts = np.asarray([loc.Transformation().Transforms(p) for p in pts])
            verts.append(pts)
            faces.extend([[i + base for i in t] for t in idxs])
            base += len(pts)
        explorer.Next()

    if not verts:
        return None
    return RawMesh(np.vstack(verts), np.asarray(faces, dtype=np.int64), str(stp))


def _face_triangulation(face):
    """取一个面的三角剖分，返回 ``(点数组, 索引数组)``。"""
    from OCP.BRep import BRep_Tool  # type: ignore
    from OCP.TopLoc import TopLoc_Location  # type: ignore

    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    if tri is None:
        return None
    pts = [tri.Node(i + 1) for i in range(tri.NbNodes())]
    idxs = []
    for i in range(tri.NbTriangles()):
        a, b, c = tri.Triangle(i + 1)
        idxs.append([a - 1, b - 1, c - 1])
    return pts, idxs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="STP/STEP → 三角网格")
    ap.add_argument("--stp", required=True)
    ap.add_argument("--out", required=True, help="输出 .ply")
    ap.add_argument("--tolerance", type=float, default=5.0,
                    help="细分弦差（模型单位）。默认 5mm 量级，够地面用")
    ap.add_argument("--max-faces", type=int, default=200_000,
                    help="抽稀上限；超了就按面积优先保留")
    ap.add_argument("--no-recenter", action="store_true",
                    help="不做场地落位，只输出原始网格")
    args = ap.parse_args(argv)

    stp = Path(args.stp)
    if not stp.exists():
        print(f"✗ 找不到 {stp}", file=sys.stderr)
        return 2
    print(f"读取 {stp}（{stp.stat().st_size / 1e6:.1f} MB）…")

    raw = tessellate_with_gmsh(stp, args.tolerance)
    if raw is None:
        print("  gmsh 不可用，尝试 OCP …")
        raw = tessellate_with_ocp(stp, args.tolerance)
    if raw is None:
        print(
            "\n✗ 没有可用的 STEP 网格化后端。\n"
            "  任选其一：\n"
            "    pip install gmsh          # 推荐，自带 OCCT reader\n"
            "    pip install cadquery      # 带 OCP\n"
            "  或者：直接用仓库里现成的场地 PLY（TCR/EGA 都附带），\n"
            "  本工具属于精度升级项，不是关键路径。",
            file=sys.stderr,
        )
        return 2

    print(f"  网格化得到 {raw.num_vertices} 顶点 / {raw.num_faces} 面")

    if raw.num_faces > args.max_faces:
        raw = _decimate(raw, args.max_faces)
        print(f"  抽稀到 {raw.num_vertices} 顶点 / {raw.num_faces} 面")

    if args.no_recenter:
        save_ply(args.out, raw)
        print(f"✓ 已写出 {args.out}（未落位）")
        return 0

    up, conf = detect_up_axis(raw.vertices, raw.faces)
    placement = suggest_mesh_to_field(raw.vertices, raw.faces, up_axis=up)
    print("  " + placement.describe())
    mesh = FieldMesh.from_raw(
        raw, mesh_to_field=placement.matrix, unit_scale=placement.unit_scale
    )
    print("  " + mesh.summary())
    save_ply(args.out, mesh)
    print(f"✓ 已写出 {args.out}")
    print("  ⚠ 落位只是建议值，务必跑 tools/validate_alignment.py 用关键点确认。")
    return 0


def _decimate(raw: RawMesh, max_faces: int) -> RawMesh:
    """按面积从大到小保留三角形（保大面，丢碎面）。

    不做真正的边坍缩——场地网格里"碎面"本就多是装饰细节，
    丢它们对地面求交的精度没有影响。
    """
    from jg_radar.field.mesh import face_areas

    areas = face_areas(raw.vertices, raw.faces)
    keep = np.argsort(-areas)[:max_faces]
    keep = np.sort(keep)
    faces = raw.faces[keep]
    used, inverse = np.unique(faces, return_inverse=True)
    return RawMesh(raw.vertices[used], inverse.reshape(-1, 3), raw.source)


if __name__ == "__main__":
    raise SystemExit(main())
