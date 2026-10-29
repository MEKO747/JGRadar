"""网格载入与分类测试。

覆盖三种格式 + 三种字节序/编码，因为 T 家仓库的 field/ 目录里 PLY 就有
两种完全不同的约定（Y 向上 / Z 向上），解析器必须有足够的容错面。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jg_radar.field import (
    FieldMesh,
    RawMesh,
    estimate_ground_z,
    face_areas,
    face_normals,
    load_mesh,
    save_ply,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SIMPLE_PLY = FIXTURES / "field_mesh_simple.ply"
YUP_PLY = FIXTURES / "field_mesh_yup.ply"


# ======================================================================
# 载入
# ======================================================================
def test_load_binary_ply():
    raw = load_mesh(SIMPLE_PLY)
    assert raw.num_vertices == 16
    assert raw.num_faces == 8
    lo, hi = raw.bounds()
    assert lo[0] == pytest.approx(0.0, abs=1e-3)
    assert hi[0] == pytest.approx(2800.0, abs=1e-3)
    assert hi[1] == pytest.approx(1500.0, abs=1e-3)
    assert hi[2] == pytest.approx(150.0, abs=1e-3)


def test_load_ascii_ply(tmp_path):
    p = tmp_path / "tri.ply"
    p.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "element face 2\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
        "0 0 0\n10 0 0\n10 10 0\n0 10 0\n"
        "4 0 1 2 3\n"          # 四边形 → 扇形三角化成 2 个
        "3 0 1 2\n",
        encoding="utf-8",
    )
    raw = load_mesh(p)
    assert raw.num_vertices == 4
    assert raw.num_faces == 3  # 1 个四边形拆 2 + 1 个三角形
    assert np.allclose(raw.vertices[2], [10, 10, 0])


def test_load_ply_with_extra_properties(tmp_path):
    """带法线的 PLY（Open3D 导出常见）必须能读。"""
    p = tmp_path / "normals.ply"
    p.write_text(
        "ply\nformat ascii 1.0\n"
        "element vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "element face 1\nproperty list uchar int vertex_indices\n"
        "end_header\n"
        "0 0 0 0 0 1\n1 0 0 0 0 1\n0 1 0 0 0 1\n"
        "3 0 1 2\n",
        encoding="utf-8",
    )
    raw = load_mesh(p)
    assert raw.num_vertices == 3 and raw.num_faces == 1
    assert np.allclose(raw.vertices[1], [1, 0, 0])


def test_load_binary_stl_and_ascii_stl(tmp_path):
    import struct

    tri = np.array([[[0, 0, 0], [10, 0, 0], [0, 10, 0]]], dtype=np.float32)
    header = b"binary stl" + b"\0" * 70
    blob = header + struct.pack("<I", 1)
    blob += struct.pack("<3f", 0, 0, 1)
    for v in tri[0]:
        blob += struct.pack("<3f", *v)
    blob += struct.pack("<H", 0)
    bp = tmp_path / "b.stl"
    bp.write_bytes(blob)
    rb = load_mesh(bp)
    assert rb.num_vertices == 3 and rb.num_faces == 1

    ap = tmp_path / "a.stl"
    ap.write_text(
        "solid s\n facet normal 0 0 1\n  outer loop\n"
        "   vertex 0 0 0\n   vertex 10 0 0\n   vertex 0 10 0\n"
        "  endloop\n endfacet\nendsolid s\n",
        encoding="utf-8",
    )
    ra = load_mesh(ap)
    assert ra.num_vertices == 3 and ra.num_faces == 1


def test_load_obj(tmp_path):
    p = tmp_path / "m.obj"
    p.write_text(
        "v 0 0 0\nv 10 0 0\nv 10 10 0\nv 0 10 0\n"
        "f 1 2 3\nf 1 3 4\n",
        encoding="utf-8",
    )
    raw = load_mesh(p)
    assert raw.num_vertices == 4 and raw.num_faces == 2


def test_unsupported_format(tmp_path):
    p = tmp_path / "x.xyz"
    p.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="不支持"):
        load_mesh(p)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_mesh("/nonexistent/field.ply")


# ======================================================================
# 几何工具
# ======================================================================
def test_face_normals_and_areas():
    v = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=float)
    f = np.array([[0, 1, 2]])
    n = face_normals(v, f)
    assert np.allclose(n[0], [0, 0, 1])
    assert face_areas(v, f)[0] == pytest.approx(50.0)


def test_face_normals_handles_degenerate():
    v = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)  # 共线
    f = np.array([[0, 1, 2]])
    assert np.allclose(face_normals(v, f)[0], 0.0)
    assert face_areas(v, f)[0] == pytest.approx(0.0)


def test_raw_mesh_rejects_bad_indices():
    with pytest.raises(ValueError, match="越界"):
        RawMesh(np.zeros((3, 3)), np.array([[0, 1, 9]]))


def test_raw_mesh_transform():
    raw = RawMesh(np.array([[1.0, 2.0, 3.0]]), np.zeros((0, 3), dtype=int))
    m = np.eye(4)
    m[:3, 3] = [10.0, 20.0, 30.0]
    out = raw.transformed(m, unit_scale=2.0)
    assert np.allclose(out.vertices[0], [12.0, 24.0, 36.0])


def test_estimate_ground_z_ignores_outliers():
    """面积加权中位数应忽略一小块沉下去的杂质面。"""
    v = np.array(
        [[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0],   # 大地板 z=0
         [0, 0, -50], [10, 0, -50], [10, 10, -50]],            # 小杂质 z=-50
        dtype=float,
    )
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])
    assert estimate_ground_z(v, f) == pytest.approx(0.0)


# ======================================================================
# FieldMesh 分类
# ======================================================================
def test_field_mesh_classification():
    mesh = FieldMesh.from_file(SIMPLE_PLY, ground_z=0.0, ground_tol=1.0)
    # 地面 2 面（z=0）+ 台阶顶面 2 面（z=3，超出 tol=1 → 不算 ground）
    assert mesh.ground_mask.sum() == 2
    # 可站立 = 所有近水平面 = 地面 2 + 台阶顶 2
    assert mesh.walkable_mask.sum() == 4
    # 遮挡 = 近垂直面 = 台阶侧面 2 + 立柱 2
    assert mesh.occluder_mask.sum() == 4
    assert mesh.ground_z == pytest.approx(0.0)


def test_field_mesh_ground_tol_includes_step():
    mesh = FieldMesh.from_file(SIMPLE_PLY, ground_z=0.0, ground_tol=5.0)
    assert mesh.ground_mask.sum() == 4  # 3cm 台阶被并入地面


def test_field_mesh_auto_ground_z():
    mesh = FieldMesh.from_file(SIMPLE_PLY)
    assert mesh.ground_z == pytest.approx(0.0, abs=1e-6)


def test_height_levels():
    mesh = FieldMesh.from_file(SIMPLE_PLY, ground_z=0.0)
    z = np.sort(mesh.heights())
    assert z[0] == pytest.approx(0.0)      # 地面
    assert z[-1] == pytest.approx(100.0)   # 立柱：三角形中心 (0+150+150)/3

    # tol=10 把同一物理面上的两个三角形并成一层：地面 0 / 台阶面 50 / 立柱面 100
    levels = mesh.height_levels(tol=10.0)
    assert len(levels) == 3
    assert levels[0] == pytest.approx(0.0)
    assert levels[-1] == pytest.approx(100.0)

    # tol 收细就会把台阶顶面(3)与侧面三角形的两个高度拆开
    assert len(mesh.height_levels(tol=1.0)) > len(levels)


def test_field_mesh_bounds_and_summary():
    mesh = FieldMesh.from_file(SIMPLE_PLY)
    lo, hi = mesh.bounds()
    assert hi[0] == pytest.approx(2800.0, abs=1e-3)
    assert "FieldMesh(" in mesh.summary()


def test_field_mesh_rejects_mask_length_mismatch():
    v = np.zeros((3, 3))
    f = np.array([[0, 1, 2]])
    with pytest.raises(ValueError, match="不符"):
        FieldMesh(v, f, np.zeros(5, dtype=bool), np.zeros(1, dtype=bool),
                  np.zeros(1, dtype=bool), 0.0)


def test_field_mesh_npz_roundtrip(tmp_path):
    mesh = FieldMesh.from_file(SIMPLE_PLY, ground_z=0.0)
    path = mesh.save_npz(tmp_path / "field.npz")
    back = FieldMesh.load_npz(path)
    assert back.num_faces == mesh.num_faces
    assert back.ground_z == pytest.approx(mesh.ground_z)
    assert (back.ground_mask == mesh.ground_mask).all()
    assert (back.occluder_mask == mesh.occluder_mask).all()
    # npz 存 float32，量化误差 1e-4 cm 量级
    assert np.allclose(back.vertices, mesh.vertices, atol=1e-3)


def test_save_ply_roundtrip(tmp_path):
    original = load_mesh(SIMPLE_PLY)
    out = save_ply(tmp_path / "out.ply", original)
    back = load_mesh(out)
    assert back.num_vertices == original.num_vertices
    assert back.num_faces == original.num_faces
    assert np.allclose(back.vertices, original.vertices, atol=1e-3)


def test_save_ply_from_field_mesh(tmp_path):
    mesh = FieldMesh.from_file(SIMPLE_PLY)
    out = save_ply(tmp_path / "field.ply", mesh)
    assert load_mesh(out).num_faces == mesh.num_faces


def test_load_real_repo_plys():
    """真实仓库网格能被解析并判对轴向。

    这几份文件的**轴向约定互不相同**（t 家是 Y 向上、EGA 是 Z 向上），
    正是方案里"CAD 坐标系对齐"这个头号风险的具体形态。
    """
    from jg_radar.field import detect_up_axis
    tcr = Path("/home/meko/radar/TCR-RM2026-Radar-OpenSource-main/ground_vision/field")
    ega = Path("/home/meko/radar/EGA_Radar_Algorithm-main/Main_ws/field/RMUC2026.ply")
    if not tcr.exists() and not ega.exists():
        pytest.skip("参考仓库不在本机")

    expect = {
        tcr / "RMUC2026_National.PLY": "y",
        tcr / "RMUC2025_National.PLY": "y",
        tcr / "RM2026_region_field_v1.0_tri.ply": "z",
        ega: "z",
    }
    checked = 0
    for path, axis in expect.items():
        if not path.exists():
            continue
        raw = load_mesh(path)
        assert raw.num_faces > 0, f"{path.name} 没读出任何面"
        up, conf = detect_up_axis(raw.vertices, raw.faces)
        assert up == axis, f"{path.name} 实测是 {axis} 向上，判成了 {up}"
        checked += 1
    if not checked:
        pytest.skip("参考仓库里没有可用的场地 PLY")


def test_area_only_vote_would_get_the_region_ply_wrong():
    """守住 detect_up_axis 的判据：这个网格里墙的面积比地面还大。

    如果哪天有人把共面条件去掉、退回"哪根轴法线面积最多"，这条会立刻失败。
    """
    from jg_radar.field import detect_up_axis
    from jg_radar.field.mesh import face_areas, face_normals

    path = (
        Path("/home/meko/radar/TCR-RM2026-Radar-OpenSource-main/ground_vision/field")
        / "RM2026_region_field_v1.0_tri.ply"
    )
    if not path.exists():
        pytest.skip("参考仓库不在本机")

    raw = load_mesh(path)
    n = face_normals(raw.vertices, raw.faces)
    a = face_areas(raw.vertices, raw.faces)
    fracs = {ax: float(a[np.abs(n[:, i]) > 0.7].sum() / a.sum())
             for i, ax in enumerate("xyz")}
    naive = max(fracs, key=fracs.get)

    assert naive != "z", "前提失效了：纯面积投票本会选错，这个用例才有意义"
    assert detect_up_axis(raw.vertices, raw.faces)[0] == "z"


def test_ega_ply_has_faces_with_properties_after_the_index_list():
    """VCGLIB 导出的 PLY 在 vertex_indices 之后还有 red/green/blue/alpha。

    解析器若假设"列表属性排最后"，每条记录会错位 4 字节，读到第 N 条时
    以 buffer 越界告终——EGA 那份 206 万面的场地网格就是这么挂掉的。
    """
    path = Path(
        "/home/meko/radar/EGA_Radar_Algorithm-main/Main_ws/field/RMUC2026.ply"
    )
    if not path.exists():
        pytest.skip("EGA 仓库不在本机")

    header = path.read_bytes()[:400]
    if b"element face" not in header:
        pytest.skip("这份 PLY 的头部结构变了，用例需要重写")
    assert header.split(b"element face")[1].count(b"property") > 1, (
        "前提失效：这份 PLY 的 face 元素已不再带后置属性"
    )

    raw = load_mesh(path)
    assert raw.num_faces > 1_000_000
    assert raw.num_vertices > 1_000_000
    # 场地真实尺寸 28m × 15m，单位是米
    lo, hi = raw.bounds()
    assert hi[0] - lo[0] == pytest.approx(28.0, abs=0.5)
    assert hi[1] - lo[1] == pytest.approx(15.0, abs=0.5)
