"""两种投影模式：``ground_plane`` 与 ``mesh``。

这一层存在的理由**只有一个**，而它值得单开一个测试文件：

> 一条撞在墙上的射线，如果被当成"落在地面"，会给出一个看起来完全合法的
> 坐标，偏出去几米，而且**不报错**。

平面模式没有任何办法发现自己错了——它不知道墙存在。所以这里测的不是
"网格模式更准"这种程度问题，而是"平面模式在什么情况下会安静地给出一个
几米外的错误坐标"，以及网格模式有没有把那种情况**变成一条明确的遮挡**。

## 场景

自造一个 28×15m 场地，加一个 200cm 高的台子（``x∈[1100,1500] y∈[300,700]``）。
相机架在场地外上方，于是：

- 台子**顶面**上的点：平面模式把射线延长到 z=0，落点在台子**后面** 4 米处
- 台子**正面墙**上的点：平面模式给出一个"墙后 3.2 米"的正常坐标；
  网格模式标为遮挡、不给坐标

这两个数都是几何算出来的，不是拟合出来的，所以测试断言的是**性质**
（"偏出去 > 3 米"、"落在台子另一侧"），而不是某几个硬编码的像素值。
"""

from __future__ import annotations

import numpy as np
import pytest

from jg_radar.field import FieldMesh, RawMesh
from jg_radar.pipeline.stage_a import StageA
from jg_radar.transform import (
    CameraModel,
    Intrinsics,
    build_lookup_map,
    effective_mesh_hash,
    look_at,
    project_points,
    world_to_camera,
)
from jg_radar.transform.lookup_map import LookupMap, mesh_digest

# 台子（cm）
BOX_X0, BOX_X1 = 1100.0, 1500.0
BOX_Y0, BOX_Y1 = 300.0, 700.0
BOX_TOP_Z = 200.0

# 相机：场地外上方，看向台子。
EYE = (1200.0, -900.0, 900.0)
TARGET = (1300.0, 500.0, 0.0)


# ----------------------------------------------------------------------
# 场景构造
# ----------------------------------------------------------------------
def _quad(a, b, c, d):
    return [(a, b, c), (a, c, d)]


def _box(x0, x1, y0, y1, z):
    """台子：顶面 + 四面侧壁。侧壁是**近垂直**的，会被判成遮挡面。"""
    top = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    base = [(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)]
    tris = _quad(*top)
    for i in range(4):
        j = (i + 1) % 4
        tris += _quad(top[i], top[j], base[j], base[i])
    return tris


def _mesh_from_tris(tris) -> FieldMesh:
    verts = np.unique(np.asarray(tris, dtype=np.float64).reshape(-1, 3), axis=0)
    index = {tuple(v): i for i, v in enumerate(verts)}
    faces = np.asarray(
        [[index[tuple(p)] for p in tri] for tri in tris], dtype=np.int64
    )
    return FieldMesh.from_raw(RawMesh(verts, faces, "synthetic-box"))


@pytest.fixture(scope="module")
def scene():
    """``(camera, R, t, mesh)``。相机无畸变——断言的是几何，不是畸变模型。"""
    tris = _quad((0, 0, 0), (2800, 0, 0), (2800, 1500, 0), (0, 1500, 0))
    tris += _box(BOX_X0, BOX_X1, BOX_Y0, BOX_Y1, BOX_TOP_Z)
    mesh = _mesh_from_tris(tris)

    intr = Intrinsics.from_matrix(
        [[1200.0, 0.0, 320.0], [0.0, 1200.0, 240.0], [0.0, 0.0, 1.0]],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    )
    camera = CameraModel(intr, 640, 480)
    R, t = look_at(EYE, TARGET)
    return camera, R, t, mesh


@pytest.fixture(scope="module")
def tables(scene):
    camera, R, t, mesh = scene
    return (
        build_lookup_map(camera, R, t, 0.0, mode="ground_plane"),
        build_lookup_map(camera, R, t, 0.0, mode="mesh", mesh=mesh),
    )


def _anchor_of(camera, R, t, world_xyz):
    """世界点 → 它落在哪个像素 → 包成一个"底边中点就是它"的框。"""
    pixels, ok = project_points(
        camera.intrinsics, world_to_camera(np.asarray([world_xyz], dtype=np.float64), R, t)
    )
    assert ok[0], f"{world_xyz} 投不到画面上"
    assert camera.in_image(pixels)[0], f"{world_xyz} 落在画面外：{pixels[0]}"
    u, v = pixels[0]
    return np.asarray([[u, v, u, v]], dtype=np.float64)


# ----------------------------------------------------------------------
# 台子顶面：平面模式把高地目标投到它身后
# ----------------------------------------------------------------------
def test_a_target_on_the_highland_lands_four_metres_behind_it_in_plane_mode(scene, tables):
    """台子顶上的目标，平面模式把它投到台子**后面**几米处。

    这是"相机架高、场地不平"时最常见的系统性偏差，也是查找表两种模式
    存在的全部理由。
    """
    camera, R, t, _ = scene
    plane, mesh = tables
    truth = np.asarray([1300.0, 500.0, BOX_TOP_Z])

    anchor = _anchor_of(camera, R, t, truth)
    plane_xy, plane_ok = plane.sample_bbox_ground(anchor)
    mesh_xy, mesh_ok = mesh.sample_bbox_ground(anchor)

    assert plane_ok[0] and mesh_ok[0]
    assert np.linalg.norm(mesh_xy[0] - truth[:2]) < 5.0, (
        f"网格模式把台顶目标投到了 {mesh_xy[0]}，真值 {truth[:2]}"
    )
    error = float(np.linalg.norm(plane_xy[0] - truth[:2]))
    assert error > 300.0, (
        f"平面模式的偏差只有 {error:.0f}cm，这个场景没测出想测的东西"
        "（台子太矮或相机太陡）"
    )
    # 偏差的方向是**背后**，不是前面：射线越过台子继续往下走。
    assert plane_xy[0, 1] > BOX_Y1, (
        f"平面模式的落点 y={plane_xy[0, 1]:.0f} 没在台子后缘 {BOX_Y1:.0f} 之后"
    )


def test_the_mesh_keeps_a_flat_ground_target_exactly_where_it_was(scene, tables):
    """平地上的目标，两种模式必须**给出同一个坐标**。

    网格模式在平地上不该有任何额外偏差——如果这里差了几十厘米，说明
    网格分类或求交写错了，而不是"网格模式更精细"。
    """
    camera, R, t, _ = scene
    plane, mesh = tables
    # 挑一个离台子足够远、且确实落在画面里的平地目标。
    truth = np.asarray([900.0, 700.0, 0.0])

    anchor = _anchor_of(camera, R, t, truth)
    plane_xy, plane_ok = plane.sample_bbox_ground(anchor)
    mesh_xy, mesh_ok = mesh.sample_bbox_ground(anchor)

    assert plane_ok[0] and mesh_ok[0]
    assert np.linalg.norm(plane_xy[0] - mesh_xy[0]) < 2.0
    assert np.linalg.norm(mesh_xy[0] - truth[:2]) < 5.0


# ----------------------------------------------------------------------
# 墙面：平面模式会安静地给出一个几米外的坐标
# ----------------------------------------------------------------------
def test_a_ray_that_hits_a_wall_is_marked_occluded_not_silently_located(scene, tables):
    """撞在墙上的射线：网格模式标遮挡，平面模式给一个偏 3 米的合法坐标。

    这个测试的重点是**右半边**。左边只是"我们记得标记遮挡"，右边才是
    这一层存在的理由：平面模式给出的那个坐标完全合法——在场内、格式对、
    不报错——只是它在墙后三米。
    """
    camera, R, t, _ = scene
    plane, mesh = tables
    on_the_wall = np.asarray([1300.0, BOX_Y0, BOX_TOP_Z - 10.0])

    anchor = _anchor_of(camera, R, t, on_the_wall)
    occluded = mesh.occluded_bbox_ground(anchor)
    mesh_xy, mesh_ok = mesh.sample_bbox_ground(anchor)

    assert occluded[0], "打在墙上的射线没有被标成遮挡"
    assert not mesh_ok[0], "被遮挡的像素不该同时给出坐标"

    plane_xy, plane_ok = plane.sample_bbox_ground(anchor)
    assert plane_ok[0], "平面模式本该照常给出坐标——那正是它危险的地方"
    error = float(np.linalg.norm(plane_xy[0] - on_the_wall[:2]))
    assert error > 200.0, (
        f"平面模式在墙上的偏差只有 {error:.0f}cm，"
        "这个场景没测出'看起来合法的错误坐标'"
    )


def test_occluded_and_valid_never_overlap(scene, tables):
    """两个掩码互斥。构造时就该拦住，这里再确认一遍构建端没写反。"""
    _, mesh = tables
    assert not np.any(mesh.occluded & mesh.valid)
    assert mesh.occluded_coverage > 0.0, "这个场景里应该有被墙挡住的像素"
    assert mesh.coverage < 1.0


def test_plane_mode_never_reports_occlusion(scene, tables):
    """平面模式的表里 ``occluded`` 必须是空的——它压根没有遮挡这个概念。

    这一条是给 ``StageA`` 的 ``_counts_occlusion`` 撑腰的：那里靠
    "表里有没有遮挡"来决定要不要每帧多问一次。
    """
    plane, _ = tables
    assert plane.mode == "ground_plane"
    assert not plane.occluded.any()
    assert plane.occluded_coverage == 0.0


# ----------------------------------------------------------------------
# 分块求交：接入查找表之后仍然与不分块逐位相同
# ----------------------------------------------------------------------
def test_chunking_the_raycast_does_not_change_the_table(scene):
    """内存预算调小一百倍，表必须**逐位相同**。

    ``_nearest_hit`` 的分块是纯记账上的改动。但"纯记账"这种事只有测出来
    才算数：块间滚动比较写错一格，症状是远小一撮像素静默地取到第二个面
    的交点——而那种表看起来一切正常。
    """
    camera, R, t, mesh = scene
    big = build_lookup_map(camera, R, t, 0.0, mode="mesh", mesh=mesh, max_block_mb=64.0)
    small = build_lookup_map(camera, R, t, 0.0, mode="mesh", mesh=mesh, max_block_mb=0.05)

    assert np.array_equal(big.valid, small.valid)
    assert np.array_equal(big.occluded, small.occluded)
    assert np.allclose(big.coords_cm, small.coords_cm, equal_nan=True)
    assert big.signature == small.signature, "分块方式不该进签名"


# ----------------------------------------------------------------------
# 签名
# ----------------------------------------------------------------------
def test_the_signature_separates_the_two_modes(scene):
    camera, R, t, mesh = scene
    plane = build_lookup_map(camera, R, t, 0.0, mode="ground_plane")
    full = build_lookup_map(camera, R, t, 0.0, mode="mesh", mesh=mesh)
    assert plane.signature != full.signature


def test_the_signature_follows_the_mesh_contents(scene):
    """换一张网格，签名就得变——否则"改了网格忘了重算表"查不出来。"""
    camera, R, t, mesh = scene
    a = build_lookup_map(camera, R, t, 0.0, mode="mesh", mesh=mesh)

    moved = _mesh_from_tris(
        _quad((0, 0, 0), (2800, 0, 0), (2800, 1500, 0), (0, 1500, 0))
        + _box(BOX_X0, BOX_X1, BOX_Y0 + 25.0, BOX_Y1, BOX_TOP_Z)
    )
    b = build_lookup_map(camera, R, t, 0.0, mode="mesh", mesh=moved)
    assert a.signature != b.signature
    # 面数一样，只有坐标动了——所以必须靠内容摘要发现，不能靠面数。
    assert mesh.num_faces == moved.num_faces
    assert mesh_digest(mesh.vertices, mesh.faces) != mesh_digest(moved.vertices, moved.faces)


def test_a_mesh_does_not_enter_a_plane_modes_signature(scene):
    """平面模式不吃网格，网格就不该进它的签名。

    否则"顺手多传了一个 mesh 参数"会产出一张签名里带网格摘要、而表本身
    与网格无关的 ``.npz``，加载端按模式不折网格、算出另一个签名，**表被
    自己的校验拒掉**，而两边都没有错。
    """
    camera, R, t, mesh = scene
    bare = build_lookup_map(camera, R, t, 0.0, mode="ground_plane")
    with_mesh = build_lookup_map(camera, R, t, 0.0, mode="ground_plane", mesh=mesh)
    assert bare.signature == with_mesh.signature
    assert effective_mesh_hash("ground_plane", mesh) == ""
    assert effective_mesh_hash("mesh", mesh) == mesh_digest(mesh.vertices, mesh.faces)


# ----------------------------------------------------------------------
# 存盘
# ----------------------------------------------------------------------
def test_save_reports_the_path_it_actually_wrote(tmp_path, tables):
    """``np.savez_compressed`` 会自己补 ``.npz``，``save`` 必须返回补过的那一个。

    照抄入参返回的话，调用方拿到的是一个不存在的路径——而失败要拖到下一次
    启动才显形。
    """
    _, mesh_table = tables
    for asked, expected_suffix in (("lk.npy", ".npy.npz"), ("lk.npz", ".npz")):
        path = mesh_table.save(tmp_path / asked)
        assert path.is_file(), f"save 返回的 {path} 不存在"
        assert path.name == "lk" + expected_suffix
        back = LookupMap.load(path)
        assert back.signature == mesh_table.signature
        assert back.mode == mesh_table.mode
        assert np.array_equal(back.occluded, mesh_table.occluded)
        assert np.allclose(back.coords_cm, mesh_table.coords_cm, equal_nan=True)


# ----------------------------------------------------------------------
# 配置接线
# ----------------------------------------------------------------------
def _write_config(tmp_path, body: str):
    path = tmp_path / "cfg.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_build_lookup_follows_the_configured_projection_mode(tmp_path, scene):
    """``transform.projection_mode`` 真的会改变建出来的那张表。

    在接上之前这个键存在、有注释、有合法取值，而**没有任何一处读它**：
    把它改成 ``mesh`` 什么也不会发生，配置看起来生效了但其实没有。
    """
    from jg_radar.pipeline.build import build_lookup

    camera, R, t, _ = scene
    mesh_path = (tmp_path / "field.ply").resolve()
    from jg_radar.field import save_ply

    save_ply(mesh_path, scene[3])

    body = f"""
camera:
  width: {camera.width}
  height: {camera.height}
  K: {camera.intrinsics.to_matrix().tolist()}
  dist: [0.0, 0.0, 0.0, 0.0, 0.0]
transform:
  R: {np.asarray(R).tolist()}
  t: {np.asarray(t).tolist()}
  projection_mode: {{mode}}
field:
  mesh_path: "{mesh_path}"
  ground_plane_z: 0.0
"""
    from jg_radar.config import load_config

    plane = build_lookup(load_config(_write_config(
        tmp_path, body.format(mode="ground_plane"))), stride=8)
    assert plane.mode == "ground_plane"

    full = build_lookup(load_config(_write_config(
        tmp_path, body.format(mode="mesh"))), stride=8)
    assert full.mode == "mesh"
    assert full.occluded.any(), "网格模式建出来的表里应当有遮挡像素"
    assert full.signature != plane.signature


def test_mesh_mode_without_a_mesh_fails_loudly(tmp_path, scene):
    """选了网格模式却没给网格 —— 必须报错，不能静默退回平面模式。

    静默退回的后果是一张看起来完全正常、实际把高地目标投到几米外的表。
    """
    from jg_radar.config import load_config
    from jg_radar.pipeline.build import build_lookup

    camera, R, t, _ = scene
    path = _write_config(tmp_path, f"""
camera:
  width: {camera.width}
  height: {camera.height}
  K: {camera.intrinsics.to_matrix().tolist()}
  dist: [0.0, 0.0, 0.0, 0.0, 0.0]
transform:
  R: {np.asarray(R).tolist()}
  t: {np.asarray(t).tolist()}
  projection_mode: mesh
field:
  mesh_path: ""
""")
    with pytest.raises(ValueError, match="网格"):
        build_lookup(load_config(path), stride=16)


def test_an_unknown_projection_mode_fails_loudly(tmp_path, scene):
    """``mash`` 这种拼写错误不许静默退回平面模式。"""
    from jg_radar.config import load_config
    from jg_radar.pipeline.build import build_lookup

    camera, R, t, _ = scene
    path = _write_config(tmp_path, f"""
camera:
  width: {camera.width}
  height: {camera.height}
  K: {camera.intrinsics.to_matrix().tolist()}
  dist: [0.0, 0.0, 0.0, 0.0, 0.0]
transform:
  R: {np.asarray(R).tolist()}
  t: {np.asarray(t).tolist()}
  projection_mode: mash
""")
    with pytest.raises(ValueError, match="mash"):
        build_lookup(load_config(path), stride=16)


def test_a_stale_table_is_refused_at_load_time(tmp_path, scene):
    """预计算表的签名对不上 → 拒绝启动。

    ``LookupMap.matches`` 原来是算出来了却没人比对，于是"改了标定忘了重算表"
    照样能跑，症状是每个坐标都偏几十厘米：没有异常、没有告警。
    """
    from jg_radar.config import load_config
    from jg_radar.pipeline.build import build_lookup

    camera, R, t, mesh = scene

    # 网格过一遍文件：加载路径是从 field.mesh_path 现读的，签名里的网格摘要
    # 也就算在**读回来的**那份上。测试里直接拿内存里那份建表的话，两边是
    # 两张网格，签名当然对不上——那是测试自己制造的假失败。
    from jg_radar.field import save_ply

    mesh_path = tmp_path / "field.ply"
    save_ply(mesh_path, mesh)
    saved = build_lookup_map(camera, R, t, 0.0, mode="mesh", mesh=mesh, stride=4).save(
        tmp_path / "lk.npz"
    )

    def config_for(rotation, translation, mode="mesh"):
        # 只写一个片段不行——加载路径要拿完整的相机与位姿现算签名，
        # 所以这里把当前场景那一套照抄进去。
        return load_config(_write_config(tmp_path, f"""
camera:
  width: {camera.width}
  height: {camera.height}
  K: {camera.intrinsics.to_matrix().tolist()}
  dist: [0.0, 0.0, 0.0, 0.0, 0.0]
transform:
  R: {np.asarray(rotation).tolist()}
  t: {np.asarray(translation).tolist()}
  projection_mode: {mode}
  lookup_map_path: "{saved}"
field:
  mesh_path: "{mesh_path}"
"""))

    assert build_lookup(config_for(R, t)).mode == "mesh"

    # 位姿挪了 25cm —— 表还读得进来，但它已经不描述当前几何了。
    shifted = np.asarray(t, dtype=np.float64) + np.array([25.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="签名"):
        build_lookup(config_for(R, shifted))

    # 模式不符：它折在签名里，但报出来该是"模式不对"这句话。
    with pytest.raises(ValueError, match="投影模式"):
        build_lookup(config_for(R, t, mode="ground_plane"))


# ----------------------------------------------------------------------
# StageA：被挡住这件事要被单独数出来
# ----------------------------------------------------------------------
def _one_car_cascade(bbox):
    from jg_radar.detect.factory import CascadeResult
    from jg_radar.detect.protocol import DetectionRecord

    rec = DetectionRecord(class_id=0, confidence=0.9, bbox=tuple(bbox), label="R1")
    return CascadeResult(
        cars=[rec],
        rois=np.asarray([bbox], dtype=np.float64).reshape(-1, 4),
        armors=[],
        digits=[],
    )


def _stage_a_run(table, anchor):
    from jg_radar.io.frame_source import Frame

    stage = StageA(table)
    packet = stage.run(Frame.with_cascade(1, 0.0, _one_car_cascade(anchor[0])))
    return stage, packet


def test_stage_a_counts_occluded_observations_separately(scene, tables):
    """被挡住的观测：没有坐标，但**照样传下去**，而且被单独计数。

    混成一个 ``None`` 的话，"被立柱挡了一下"和"目标开出了场地"在日志里
    长得一模一样，而这两件事该触发完全不同的反应。
    """
    camera, R, t, _ = scene
    plane, mesh = tables
    anchor = _anchor_of(camera, R, t, np.asarray([1300.0, BOX_Y0, BOX_TOP_Z - 10.0]))

    stage, packet = _stage_a_run(mesh, anchor)
    assert packet.observations[0].world_cm is None, "被遮挡的观测不该有世界坐标"
    assert len(packet.observations) == 1, "投影不出来 ≠ 丢掉这条观测"
    assert stage.occluded == 1
    assert stage.diagnostics()["stage_a_occluded"] == 1.0

    # 同一像素在平面模式的表上：给得出坐标，也就数不出遮挡。
    stage_plane, packet_plane = _stage_a_run(plane, anchor)
    assert packet_plane.observations[0].world_cm is not None
    assert stage_plane.occluded == 0
