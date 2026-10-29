"""像素 → 场地坐标预计算查找表。**运行时主路径。**

相机静止、场地静止 ⇒ 每个像素对应的场地点是个常数。于是把整张映射表在标定
时算一次、存成 ``.npz``，运行时只做数组索引：

  - 运行时**零 3D 库依赖**（也就绕开了 open3d 在 Python 3.12 上的 wheel 风险）
  - 查表是纯数组索引：与库版本、BVH 遍历顺序、浮点路径都无关，结果确定
  - 映射表的内容可以被测试逐个像素地验证
  - **失效判定靠签名**：分辨率、内参、畸变、外参、地面高度、投影模式、网格
    内容任一变化，签名就变。签名字符串里带全部输入，因此不可能"悄悄用错表"

## 关于"快"

本模块原来写着"比逐帧射线求交快几个数量级"。单次查询确实如此，但在
``ground_plane`` 模式下**这不是理由**：一帧只投十几个锚点，numpy 向量化的
``intersect_plane`` 也就几十微秒。速度真正成为硬理由的是 ``mesh`` 模式——
那里是 (射线 × 三角形)，全场网格几万个面 × 一千多万像素，几小时起步（见
``tools/build_lookup_map.py`` 的代价估计）。

查找表的正确讲法是**把一次性的构建代价挪到标定阶段，换运行时的零依赖、
可验证与确定性**，而不是"查表更快"。

## 两种投影模式

``ground_plane``（默认）
    与水平面求交。相机架得高、场地平的时候够用。

``mesh``
    与场地三角网格求交，高地、坡道上的目标才落得对。逐像素取**最近的**
    命中面，然后按面分类：

    - 命中**可站立面**（近水平：地面、高地顶、坡道）→ 有效坐标
    - 命中**遮挡面**（近垂直：墙、立柱）→ 记进 ``occluded``，**坐标留空**

    第二条是这一层存在的理由。一条撞在立柱上的射线，如果当成"落在地面"，
    会给出一个看起来完全合法的坐标，偏出去几米，而且**不报错**——这是这类
    系统最隐蔽的一类错误。所以遮挡必须是**另一个**掩码，不能和"射线打空"
    混成同一个 invalid：前者说"这里有东西挡着"，后者说"这里什么都没有"。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jg_radar.transform.camera_model import CameraModel, Intrinsics
from jg_radar.transform.raycast import intersect_mesh, intersect_plane

LOOKUP_VERSION = 2
"""1 → 2：``LookupMap`` 增加了 ``occluded`` 与 ``mode``。

版本号必须跟着形状走。旧的 ``.npz`` 里没有这两样，按 1 读进来会得到一张
"永远没有遮挡、模式永远是平面"的表——而它可能是按网格算的。宁可让它在
``load`` 那一刻就报错。
"""

MODE_GROUND_PLANE = "ground_plane"
MODE_MESH = "mesh"
KNOWN_MODES = (MODE_GROUND_PLANE, MODE_MESH)


def compute_signature(
    intrinsics: Intrinsics,
    R: np.ndarray,
    t: np.ndarray,
    width: int,
    height: int,
    plane_z: float,
    *,
    mesh_hash: str = "",
    mode: str = MODE_GROUND_PLANE,
) -> str:
    """把所有影响映射结果的输入揉进一个短哈希。"""
    payload = {
        "v": LOOKUP_VERSION,
        "mode": mode,
        "fx": round(intrinsics.fx, 6),
        "fy": round(intrinsics.fy, 6),
        "cx": round(intrinsics.cx, 6),
        "cy": round(intrinsics.cy, 6),
        "dist": [round(float(v), 8) for v in intrinsics.dist_vector()],
        "R": np.asarray(R, dtype=np.float64).round(10).tolist(),
        "t": np.asarray(t, dtype=np.float64).round(6).tolist(),
        "w": int(width),
        "h": int(height),
        "z": round(float(plane_z), 6),
        "mesh": mesh_hash,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(blob.encode(), digest_size=16).hexdigest()


@dataclass(frozen=True)
class LookupMap:
    """``(H, W, 2)`` 的像素 → 场地坐标表（cm）。"""

    coords_cm: np.ndarray
    valid: np.ndarray
    signature: str
    plane_z: float = 0.0
    mode: str = MODE_GROUND_PLANE
    occluded: np.ndarray | None = None
    """``(H, W)`` 布尔：这个像素的射线**先撞上了遮挡面**（墙、立柱）。

    与 ``valid`` 互斥，而且是**分开的一维信息**：``valid=False`` 只说"这个
    像素给不出坐标"，
    但"因为那里有东西挡着"和"因为射线指向了天"是两件不同的事——前者说明
    目标可能就在后面，后者说明那儿什么都没有。混成一个 invalid，遮挡就再也
    查不出来了，而那正是本模块最想挡住的那种失败。

    ``None`` 一律解释成"这张表没有遮挡信息"，在构造时铺成全 False。
    """

    def __post_init__(self) -> None:
        if self.coords_cm.ndim != 3 or self.coords_cm.shape[2] != 2:
            raise ValueError(f"coords_cm 必须是 (H,W,2)，得到 {self.coords_cm.shape}")
        if self.coords_cm.shape[:2] != self.valid.shape:
            raise ValueError("coords_cm 与 valid 的形状不匹配")
        if self.mode not in KNOWN_MODES:
            raise ValueError(
                f"未知的投影模式 {self.mode!r}；可用 = {list(KNOWN_MODES)}"
            )

        occluded = (
            np.zeros_like(self.valid)
            if self.occluded is None
            else np.asarray(self.occluded, dtype=bool)
        )
        if occluded.shape != self.valid.shape:
            raise ValueError("occluded 与 valid 的形状不匹配")
        object.__setattr__(self, "occluded", occluded)

        overlap = int(np.count_nonzero(occluded & self.valid))
        if overlap:
            # 两者同时为真 = "这个像素既有坐标又被挡住"，自相矛盾。它一定是
            # 构建时写错了，而错的后果是遮挡信息静默失效。
            raise ValueError(f"occluded 与 valid 有 {overlap} 个像素同时为真")

    @property
    def height(self) -> int:
        return int(self.coords_cm.shape[0])

    @property
    def width(self) -> int:
        return int(self.coords_cm.shape[1])

    @property
    def coverage(self) -> float:
        """有效像素占比。"""
        return float(self.valid.mean())

    @property
    def occluded_coverage(self) -> float:
        """被遮挡的像素占比。``mesh`` 模式下这是"有多少画面被挡住"的度量。"""
        return float(self.occluded.mean())

    # ------------------------------------------------------------------
    def sample(self, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """按像素取世界坐标。返回 ``(world_cm (N,2), valid (N,))``。

        越界像素标为无效，不抛异常——检测框可能压在图像边缘上。
        """
        px = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        u = np.rint(px[:, 0]).astype(np.int64)
        v = np.rint(px[:, 1]).astype(np.int64)
        inb = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        world = np.full((len(px), 2), np.nan, dtype=np.float64)
        ok = inb & self.valid[np.clip(v, 0, self.height - 1), np.clip(u, 0, self.width - 1)]
        world[ok] = self.coords_cm[v[ok], u[ok]]
        return world, ok

    def occluded_at(self, pixels: np.ndarray) -> np.ndarray:
        """这些像素是不是被挡住了。返回 ``(N,)`` 布尔。

        单独一个查询而不是并进 ``sample`` 的返回值：绝大多数调用方只关心
        "坐标是多少"，多返回一个数组会让每一个解包点都要改；而真正需要
        区分遮挡的那一处（``StageA``）会显式地再来问一次。
        """
        px = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        u = np.rint(px[:, 0]).astype(np.int64)
        v = np.rint(px[:, 1]).astype(np.int64)
        inb = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return inb & self.occluded[
            np.clip(v, 0, self.height - 1), np.clip(u, 0, self.width - 1)
        ]

    def sample_bbox_center(self, bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """按**框中心**取世界坐标。``bbox`` 为 ``(N,4)`` 的 x1,y1,x2,y2。

        .. warning::

            **算机器人位置不要用这个。** 框中心在车的一半高度上，穿过它的
            射线打到地面要落在车**后方**——相机越平、距离越远，偏得越多。

            这不是理论顾虑：拿回放 fixture（12MP+12mm 的 1/4 降采样、相机
            架在端线后 5m）实测，这个偏差平均 121cm、最大 161cm。而
            ``switch_jump_cm`` 才 150cm——**光是把锚点选错，就足以让每一次
            换源都被判成"跳变"**，滞回门从此形同虚设。

            要位置就用 :meth:`sample_bbox_ground`。本方法留给图像域用途
            （画叠层、算尺度），那里框中心才是对的。
        """
        return self.sample(self._center_anchors(bbox))

    def sample_bbox_ground(self, bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """按框**底边中点**取世界坐标——**算机器人位置用这个**。

        底边中点是"车与地面的接触点"的最好近似：它是那个框上唯一真的落在
        地面平面附近的点，所以穿过它的射线与地面的交点就是机器人所在处。

        它仍是个近似（车有宽度、相机有俯仰，底边中点比真实接地点略偏），
        但那个残差是**厘米级向下**的，而框中心带来的偏差是**米级**的。
        两个不是一回事，别互相抵消着算。
        """
        return self.sample(self._ground_anchors(bbox))

    def occluded_bbox_ground(self, bbox: np.ndarray) -> np.ndarray:
        """框的底边中点那个像素是不是被挡住了。返回 ``(N,)`` 布尔。

        和 :meth:`sample_bbox_ground` 问的是**同一个像素**，所以两个答案
        拼起来才完整："这个框拿不到坐标，是因为它后面是一堵墙"，还是
        "因为那一带根本没有对应的场地"。
        """
        return self.occluded_at(self._ground_anchors(bbox))

    # ------------------------------------------------------------------
    @staticmethod
    def _center_anchors(bbox: np.ndarray) -> np.ndarray:
        b = np.asarray(bbox, dtype=np.float64).reshape(-1, 4)
        return np.stack([(b[:, 0] + b[:, 2]) * 0.5, (b[:, 1] + b[:, 3]) * 0.5], axis=1)

    @staticmethod
    def _ground_anchors(bbox: np.ndarray) -> np.ndarray:
        b = np.asarray(bbox, dtype=np.float64).reshape(-1, 4)
        return np.stack([(b[:, 0] + b[:, 2]) * 0.5, b[:, 3]], axis=1)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """存盘。坐标按 float32 落盘。**返回真正写出去的那个路径。**

        4024×3036 的全表 float64 是 196MB，float32 减半。代价是量化误差约
        1e-4 cm（1 微米），比标定误差小四个数量级，可以忽略。用
        ``savez_compressed`` 是因为 ``valid`` / ``occluded`` 两个布尔面的
        压缩率极高，而它们正是网格模式全部的信息增量。

        .. warning::

            **扩展名要在这里定死，不能让 numpy 定。** ``np.savez_compressed``
            在文件名不以 ``.npz`` 结尾时会自己**追加**一个：喂给它
            ``lookup.npy``，落盘的是 ``lookup.npy.npz``，而函数返回值如果照抄
            入参，调用方拿到的就是一个**不存在的路径**。那条路径上的失败会
            拖到很久以后才显形——配置里写 ``lookup_map_path: build/lookup.npy``、
            工具报"已写入"、下次启动 FileNotFoundError，中间隔着一整轮部署。
            所以这里自己补后缀，并返回补过的那一个。
        """
        p = Path(path)
        if p.suffix.lower() != ".npz":
            p = p.with_name(p.name + ".npz")
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            coords_cm=self.coords_cm.astype(np.float32),
            valid=self.valid,
            occluded=self.occluded,
            signature=np.array(self.signature),
            plane_z=np.array(float(self.plane_z)),
            mode=np.array(self.mode),
            version=np.array(LOOKUP_VERSION),
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> "LookupMap":
        data = np.load(Path(path), allow_pickle=False)
        if int(data["version"]) != LOOKUP_VERSION:
            raise ValueError(
                f"查找表版本不匹配：文件 {int(data['version'])}，本程序 {LOOKUP_VERSION}"
            )
        return cls(
            coords_cm=np.asarray(data["coords_cm"], dtype=np.float64),
            valid=np.asarray(data["valid"], dtype=bool),
            occluded=np.asarray(data["occluded"], dtype=bool),
            signature=str(data["signature"]),
            plane_z=float(data["plane_z"]),
            mode=str(data["mode"]),
        )

    def matches(self, signature: str) -> bool:
        """签名是否仍然有效。"""
        return self.signature == signature


def build_lookup_map(
    camera: CameraModel,
    R: np.ndarray,
    t: np.ndarray,
    plane_z: float,
    *,
    mode: str = MODE_GROUND_PLANE,
    mesh=None,
    mesh_hash: str = "",
    stride: int = 1,
    max_block_mb: float = 192.0,
) -> LookupMap:
    """逐像素射线求交，生成映射表。

    ``stride > 1`` 时只算 1/stride 的像素，其余留作无效——用于低分辨率预览
    或省时间的粗表。默认 stride=1 即全分辨率。

    ``mode="mesh"`` 需要 ``mesh``（``jg_radar.field.mesh.FieldMesh``），
    并且会用它的 ``walkable_mask`` 把命中面分成"可站立"与"遮挡"。**缺网格
    直接报错**——静默退回平面模式的话，得到的是一张看起来完全正常、实际把
    高地目标投到几米外的表，而那正是查找表机制最该防住的失败。

    ``max_block_mb`` 只影响网格模式求交时的分块大小，见 ``raycast``。
    """
    if stride < 1:
        raise ValueError("stride 必须 ≥ 1")
    if mode not in KNOWN_MODES:
        # 配置里把 mesh 拼成 mash，静默退回平面模式 = 整条投影链跑错而无人知晓
        raise ValueError(f"未知的投影模式 {mode!r}；可用 = {list(KNOWN_MODES)}")
    if mode == MODE_MESH and mesh is None:
        raise ValueError(
            "projection_mode=mesh 但没有给出场地网格。网格模式必须有网格："
            "它要用网格把射线分成'打在地面/高地上'与'打在墙上'两种，"
            "没有网格就只能退回平面模式，而那是**静默地把高地目标投到它身后**。"
            "请在 field.mesh_path 里给出网格，或把 projection_mode 改回 ground_plane。"
        )

    w, h = camera.width, camera.height

    vs = np.arange(0, h, stride)
    us = np.arange(0, w, stride)
    grid_u, grid_v = np.meshgrid(us, vs)
    pixels = np.stack([grid_u.ravel(), grid_v.ravel()], axis=1).astype(np.float64)
    gv, gu = grid_v.ravel(), grid_u.ravel()

    origins, dirs = camera.rays_world(pixels, R, t)
    inb = camera.in_image(pixels)

    if mode == MODE_MESH:
        # 对**整张**网格取最近命中，再看那个面是什么。顺序不能反：先只在
        # 可站立面上求交的话，一堵挡在前面的墙会被完全无视，射线会穿墙打到
        # 墙后的地面——那正是遮挡标记要拦下的情形。
        hit = intersect_mesh(
            origins, dirs, mesh.vertices, mesh.faces, max_block_mb=max_block_mb
        )
        standable = _standable_of(mesh, hit.face_index)
        ok = hit.valid & standable & inb
        blocked = hit.valid & ~standable & inb
    else:
        hit = intersect_plane(origins, dirs, plane_z)
        ok = hit.valid & inb
        blocked = np.zeros(len(pixels), dtype=bool)

    coords = np.full((h, w, 2), np.nan, dtype=np.float64)
    valid = np.zeros((h, w), dtype=bool)
    occluded = np.zeros((h, w), dtype=bool)
    world_xy = hit.points_cm[:, :2]
    coords[gv[ok], gu[ok]] = world_xy[ok]
    valid[gv[ok], gu[ok]] = True
    occluded[gv[blocked], gu[blocked]] = True

    digest = effective_mesh_hash(mode, mesh, mesh_hash)
    return LookupMap(
        coords_cm=coords,
        valid=valid,
        occluded=occluded,
        signature=compute_signature(
            camera.intrinsics, R, t, w, h, plane_z, mesh_hash=digest, mode=mode
        ),
        plane_z=float(plane_z),
        mode=mode,
    )


def effective_mesh_hash(mode: str, mesh, explicit: str = "") -> str:
    """签名里**该出现**的那个网格摘要。构建端与加载端都用这一个。

    规则只有一句：**这个模式下用不到的输入，就不该进签名。**

    平面模式不吃网格，所以它算出来的表和网格内容无关，签名也就不该把网格
    折进去。否则"给平面模式的构建顺手传了个网格"这种事，会产出一张签名
    里带着网格摘要、而表本身与网格无关的 ``.npz``——加载端按模式判断不折
    网格、算出另一个签名，于是**表被自己的签名校验拒掉**，而两边都没有错。
    一个只能靠"记得别多传参数"来维持的约定，不如在这里一次说清。
    """
    if mode == MODE_MESH and mesh is not None:
        return mesh_digest(mesh.vertices, mesh.faces)
    return str(explicit)


def _standable_of(mesh, face_index: np.ndarray) -> np.ndarray:
    """命中面是不是**可站立**的（近水平）。``face_index < 0`` 一律不是。

    网格为空时 ``walkable_mask`` 也是空的，索引会越界——那时的正确答案是
    "什么都没命中"，所以直接返回全 False，而不是抛异常。
    """
    mask = np.asarray(mesh.walkable_mask, dtype=bool).reshape(-1)
    if not len(mask):
        return np.zeros(len(face_index), dtype=bool)
    idx = np.clip(np.asarray(face_index, dtype=np.int64), 0, len(mask) - 1)
    return (np.asarray(face_index, dtype=np.int64) >= 0) & mask[idx]


def mesh_digest(vertices: np.ndarray, faces: np.ndarray) -> str:
    """网格内容摘要，用于查找表签名。

    叫 ``digest`` 不叫 ``hash`` 是为了不和 ``build_lookup_map`` 的同名参数
    打架：那个参数是个字符串，在函数体里会把同名函数整个遮住，
    于是"算一下网格摘要"会变成"调用一个字符串"。
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(np.ascontiguousarray(np.asarray(vertices, dtype=np.float32)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(faces, dtype=np.int32)).tobytes())
    return h.hexdigest()
