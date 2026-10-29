"""场地三角网格：载入、单位换算、坐标系对齐、地面/障碍拆分。

支持 PLY（ascii / binary 大小端）、STL（ascii / binary）、OBJ，**纯 numpy**。
不用 open3d：它是 Python 3.12 上的 wheel 风险，而且运行时根本不需要它——
运行时走预计算查找表，网格只在离线构建时用一次。

为什么必须拆 ground / walkable / occluder：
  一条射线打到立柱上，如果被当成"落在地面"，就会得到一个看起来合法、
  实际上偏了几米的坐标，而且**不会报错**。拆出 occluder 之后这种情况
  可以被显式标记为遮挡。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# PLY 属性类型 → numpy dtype 字符
_PLY_TYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


# ======================================================================
# 原始网格
# ======================================================================
@dataclass(frozen=True)
class RawMesh:
    """未经场地对齐的网格，坐标就是文件里的坐标。"""

    vertices: np.ndarray
    """(N,3) float64。"""
    faces: np.ndarray
    """(M,3) int64，三角形顶点索引。"""
    source: str = ""

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        f = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)
        object.__setattr__(self, "vertices", v)
        object.__setattr__(self, "faces", f)
        if len(f) and (f.min() < 0 or f.max() >= len(v)):
            raise ValueError(
                f"面索引越界：范围 [{f.min()}, {f.max()}]，顶点数 {len(v)}"
            )

    @property
    def num_vertices(self) -> int:
        return int(len(self.vertices))

    @property
    def num_faces(self) -> int:
        return int(len(self.faces))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if not len(self.vertices):
            return np.zeros(3), np.zeros(3)
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def extent(self) -> np.ndarray:
        lo, hi = self.bounds()
        return hi - lo

    def transformed(self, matrix: np.ndarray, unit_scale: float = 1.0) -> "RawMesh":
        """施加 4×4 齐次变换（先缩放再变换）。"""
        m = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        v = self.vertices * float(unit_scale)
        vh = np.concatenate([v, np.ones((len(v), 1))], axis=1)
        out = (vh @ m.T)[:, :3]
        return RawMesh(out, self.faces, self.source)


# ======================================================================
# 载入
# ======================================================================
def load_mesh(path: str | Path) -> RawMesh:
    """按扩展名分派。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"网格文件不存在：{p}")
    suffix = p.suffix.lower()
    if suffix == ".ply":
        return load_ply(p)
    if suffix == ".stl":
        return load_stl(p)
    if suffix == ".obj":
        return load_obj(p)
    raise ValueError(f"不支持的网格格式：{suffix}（支持 .ply/.stl/.obj）")


def load_ply(path: str | Path) -> RawMesh:
    """读 PLY。支持 ascii 与 binary little/big endian。"""
    p = Path(path)
    blob = p.read_bytes()

    marker = b"end_header"
    idx = blob.find(marker)
    if idx < 0:
        raise ValueError(f"{p}: 找不到 end_header，不是合法 PLY")
    nl = blob.find(b"\n", idx)
    header_end = nl + 1
    header = blob[:header_end].decode("ascii", errors="replace")

    fmt = "ascii"
    elements: list[tuple[str, int, list[tuple[str, str, str | None]]]] = []
    for line in header.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            elements.append((parts[1], int(parts[2]), []))
        elif parts[0] == "property" and elements:
            if parts[1] == "list":
                elements[-1][2].append((parts[4], parts[3], parts[2]))
            else:
                elements[-1][2].append((parts[2], parts[1], None))

    if fmt == "ascii":
        return _parse_ply_ascii(blob[header_end:].decode("ascii", errors="replace"), elements, str(p))

    endian = "<" if fmt == "binary_little_endian" else ">"
    return _parse_ply_binary(blob, header_end, elements, endian, str(p))


def _element_dtype(props: list[tuple[str, str, str | None]], endian: str) -> np.dtype:
    return np.dtype([(name, endian + _PLY_TYPES[typ]) for name, typ, _ in props])


def _itemsize(typ: str, endian: str) -> int:
    return int(np.dtype(endian + _PLY_TYPES[typ]).itemsize)


def _split_props(
    props: list[tuple[str, str, str | None]],
) -> tuple[int, str, str, int, int]:
    """拆出「列表属性」及其前后固定属性的字节数。

    PLY 不要求列表属性排在最后。VCGLIB（MeshLab 的写出器）导出的
    ``element face`` 就是 ``vertex_indices`` 后面还跟着 red/green/blue/alpha，
    只跳前面的固定属性会让每条记录错位 4 字节，最后在读第 N 条时炸掉。
    """
    list_idx = next(i for i, (_, _, c) in enumerate(props) if c is not None)
    _, ltype, lcount = props[list_idx]
    pre = sum(_itemsize(t, "") for _, t, c in props[:list_idx] if c is None)
    post = sum(_itemsize(t, "") for _, t, c in props[list_idx + 1:] if c is None)
    return list_idx, ltype, lcount or "uchar", pre, post


def _stack_faces(chunks: list[np.ndarray]) -> np.ndarray:
    """把各条记录（每条是 (k,3) 的三角形块）拼成 (M,3)。

    用 concatenate 而不是 asarray：四边形拆出的 (2,3) 与三角形的 (1,3)
    首维不同，直接堆会得到 object 数组。
    """
    if not chunks:
        return np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(chunks, axis=0).astype(np.int64)


def _fan(idxs: np.ndarray, n: int) -> np.ndarray:
    """把一条 n 边形索引扇形三角化成 (n-2, 3)。"""
    if n == 3:
        return idxs.reshape(-1, 3)
    return np.concatenate([idxs[:, [0, k, k + 1]] for k in range(1, n - 1)], axis=0)


def _read_ply_faces(
    blob: bytes,
    offset: int,
    count: int,
    pre: int,
    post: int,
    cdt: np.dtype,
    idt: np.dtype,
    name: str,
    faces: list[np.ndarray],
) -> int:
    """读 ``count`` 条「前置固定字节 + 索引列表 + 后置固定字节」，返回消耗字节数。

    场地网格动辄两百万面（EGA 的 RMUC2026.ply 就是 206 万），逐面 Python
    循环要几十秒。索引条数一致时走结构体 dtype 的向量化快路径，不一致再退回
    逐条解析。
    """
    if offset + pre + cdt.itemsize > len(blob):
        raise ValueError("PLY 数据段比头部声明得短")
    first = int(np.frombuffer(blob, dtype=cdt, count=1, offset=offset + pre)[0])
    # PLY 的列表长度至少是 1；3 以下（点/线）对三角网格没有意义，照样按快路径读掉
    stride = pre + cdt.itemsize + idt.itemsize * first + post
    if first >= 3 and offset + count * stride <= len(blob):
        rec = np.dtype(
            [("a", f"V{pre}"), ("n", cdt), ("i", idt, (first,)), ("b", f"V{post}")]
        )
        block = np.frombuffer(blob, dtype=rec, count=count, offset=offset)
        if bool(np.all(block["n"] == first)):  # 长度确实一致，快路径成立
            if name == "face":
                faces.append(_fan(block["i"].astype(np.int64), first))
            return int(count * stride)

    pos = offset
    for _ in range(count):
        pos += pre
        n = int(np.frombuffer(blob, dtype=cdt, count=1, offset=pos)[0])
        pos += cdt.itemsize
        if pos + idt.itemsize * n > len(blob):
            raise ValueError("PLY 数据段在一条记录中间就结束了")
        idxs = np.frombuffer(blob, dtype=idt, count=n, offset=pos).astype(np.int64)
        pos += idt.itemsize * n + post
        if name == "face" and n >= 3:
            faces.append(_fan(idxs.reshape(1, -1), n))
    return int(pos - offset)


def _parse_ply_binary(
    blob: bytes,
    offset: int,
    elements: list[tuple[str, int, list[tuple[str, str, str | None]]]],
    endian: str,
    source: str,
) -> RawMesh:
    vertices: np.ndarray | None = None
    faces: list[np.ndarray] = []

    for name, count, props in elements:
        has_list = any(lst is not None for _, _, lst in props)
        if not has_list:
            dt = _element_dtype(props, endian)
            arr = np.frombuffer(blob, dtype=dt, count=count, offset=offset)
            offset += arr.nbytes
            if name == "vertex":
                cols = [c for c in ("x", "y", "z") if c in arr.dtype.names]
                if len(cols) != 3:
                    raise ValueError(f"{source}: vertex 元素缺少 x/y/z")
                vertices = np.stack([arr[c] for c in cols], axis=1).astype(np.float64)
            continue

        # 含 list 属性（典型是 face）
        _, ltype, lcount, pre, post = _split_props(props)
        cdt = np.dtype(endian + _PLY_TYPES[lcount])
        idt = np.dtype(endian + _PLY_TYPES[ltype])
        consumed = _read_ply_faces(blob, offset, count, pre, post, cdt, idt, name, faces)
        offset += consumed

    if vertices is None:
        raise ValueError(f"{source}: 没有 vertex 元素")
    return RawMesh(vertices, _stack_faces(faces), source)


def _parse_ply_ascii(
    text: str,
    elements: list[tuple[str, int, list[tuple[str, str, str | None]]]],
    source: str,
) -> RawMesh:
    tokens = text.split()
    cursor = 0
    vertices: np.ndarray | None = None
    faces: list[np.ndarray] = []

    for name, count, props in elements:
        if any(lst is not None for _, _, lst in props):
            # 列表属性未必排在最后，其后的固定属性也要吃进来
            list_idx, _, _, _, _ = _split_props(props)
            tail = len(props) - list_idx - 1
            for _ in range(count):
                cursor += list_idx
                n = int(tokens[cursor])
                idxs = [int(t) for t in tokens[cursor + 1:cursor + 1 + n]]
                cursor += 1 + n + tail
                if name == "face" and n >= 3:
                    faces.append(_fan(np.asarray(idxs, dtype=np.int64).reshape(1, -1), n))
        else:
            ncol = len(props)
            block = np.array(tokens[cursor:cursor + count * ncol], dtype=np.float64)
            cursor += count * ncol
            block = block.reshape(count, ncol)
            if name == "vertex":
                names = [n for n, _, _ in props]
                cols = [names.index(c) for c in ("x", "y", "z")]
                vertices = block[:, cols]

    if vertices is None:
        raise ValueError(f"{source}: 没有 vertex 元素")
    return RawMesh(vertices, _stack_faces(faces), source)


def load_stl(path: str | Path) -> RawMesh:
    """读 STL，自动区分 ascii / binary。"""
    p = Path(path)
    blob = p.read_bytes()
    if blob[:5].lower().startswith(b"solid") and b"facet" in blob[:512].lower():
        return _parse_stl_ascii(blob.decode("ascii", errors="replace"), str(p))
    return _parse_stl_binary(blob, str(p))


def _parse_stl_binary(blob: bytes, source: str) -> RawMesh:
    if len(blob) < 84:
        raise ValueError(f"{source}: 文件太短，不是合法 STL")
    (count,) = struct.unpack_from("<I", blob, 80)
    rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    if len(blob) < 84 + count * rec.itemsize:
        raise ValueError(f"{source}: STL 长度与声明的三角形数不符")
    data = np.frombuffer(blob, dtype=rec, count=count, offset=84)
    tris = data["v"].astype(np.float64)          # (M,3,3)
    vertices, faces = _dedup_triangles(tris)
    return RawMesh(vertices, faces, source)


def _parse_stl_ascii(text: str, source: str) -> RawMesh:
    verts = [
        [float(x) for x in line.split()[1:4]]
        for line in text.splitlines()
        if line.strip().lower().startswith("vertex")
    ]
    tris = np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)
    vertices, faces = _dedup_triangles(tris)
    return RawMesh(vertices, faces, source)


def _dedup_triangles(tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把 (M,3,3) 的独立三角形去重成共享顶点。"""
    flat = tris.reshape(-1, 3)
    vertices, inverse = np.unique(np.round(flat, 9), axis=0, return_inverse=True)
    return vertices, inverse.reshape(-1, 3).astype(np.int64)


def load_obj(path: str | Path) -> RawMesh:
    """读 OBJ（只取 v / f，忽略法线与纹理）。"""
    p = Path(path)
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v" and len(parts) >= 4:
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f" and len(parts) >= 4:
            idx = []
            for tok in parts[1:]:
                raw = int(tok.split("/")[0])
                idx.append(raw - 1 if raw > 0 else len(verts) + raw)
            for k in range(1, len(idx) - 1):
                faces.append([idx[0], idx[k], idx[k + 1]])
    if not verts:
        raise ValueError(f"{p}: 没有顶点")
    return RawMesh(
        np.asarray(verts, dtype=np.float64),
        np.asarray(faces, dtype=np.int64).reshape(-1, 3),
        str(p),
    )


# ======================================================================
# 场地网格
# ======================================================================
def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """单位面法线 (M,3)。退化三角形返回零向量。"""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(f):
        return np.zeros((0, 3))
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, norm, out=np.zeros_like(n), where=norm > 1e-15)


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """三角形面积 (M,)。"""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(f):
        return np.zeros(0)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


@dataclass(frozen=True)
class FieldMesh:
    """已对齐到场地坐标系的网格，并按可站立性拆好。"""

    vertices: np.ndarray
    """(N,3) float64，场地坐标系，单位 cm。"""
    faces: np.ndarray
    """(M,3) int64。"""
    ground_mask: np.ndarray
    """近水平且位于地面高度。"""
    walkable_mask: np.ndarray
    """近水平的可站立面（含高地/坡顶）。"""
    occluder_mask: np.ndarray
    """近垂直的面（墙、立柱）。射线打到这些面就是遮挡。"""
    ground_z: float
    source: str = ""

    def __post_init__(self) -> None:
        f = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)
        object.__setattr__(self, "vertices", np.asarray(self.vertices, dtype=np.float64))
        object.__setattr__(self, "faces", f)
        for attr in ("ground_mask", "walkable_mask", "occluder_mask"):
            m = np.asarray(getattr(self, attr), dtype=bool).reshape(-1)
            if len(m) != len(f):
                raise ValueError(f"{attr} 长度 {len(m)} 与面数 {len(f)} 不符")
            object.__setattr__(self, attr, m)

    # -- 构造 -------------------------------------------------------------
    @classmethod
    def from_raw(
        cls,
        raw: RawMesh,
        *,
        mesh_to_field: np.ndarray | None = None,
        unit_scale: float = 1.0,
        ground_z: float | None = None,
        horizontal_cos: float = 0.5,
        ground_tol: float = 5.0,
    ) -> "FieldMesh":
        """把原始网格搬到场地系并分类。

        ``horizontal_cos``：面法线 z 分量绝对值超过它才算"近水平"。
        0.5 ≈ 与水平面夹角小于 60°。
        """
        grid = raw
        if mesh_to_field is not None:
            grid = raw.transformed(mesh_to_field, unit_scale)
        elif unit_scale != 1.0:
            grid = RawMesh(raw.vertices * unit_scale, raw.faces, raw.source)

        v, f = grid.vertices, grid.faces
        normals = face_normals(v, f)
        areas = face_areas(v, f)
        if ground_z is None:
            ground_z = estimate_ground_z(v, f)
        ground_z = float(ground_z)

        if not len(f):
            empty = np.zeros(0, dtype=bool)
            return cls(v, f, empty, empty, empty, ground_z, raw.source)

        horizontal = np.abs(normals[:, 2]) > horizontal_cos
        centers_z = v[f].mean(axis=1)[:, 2]
        ground = horizontal & (np.abs(centers_z - ground_z) <= ground_tol)
        occluder = ~horizontal

        # 退化面两边都不算：面积为零或法线为零
        degenerate = (areas <= 1e-12) | (np.linalg.norm(normals, axis=1) < 1e-12)
        ground &= ~degenerate
        occluder &= ~degenerate
        walkable = horizontal & ~degenerate

        return cls(v, f, ground, walkable, occluder, ground_z, raw.source)

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "FieldMesh":
        return cls.from_raw(load_mesh(path), **kwargs)

    # -- 查询 -------------------------------------------------------------
    @property
    def num_faces(self) -> int:
        return int(len(self.faces))

    @property
    def num_vertices(self) -> int:
        return int(len(self.vertices))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if not len(self.vertices):
            return np.zeros(3), np.zeros(3)
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def heights(self) -> np.ndarray:
        """各三角形中心的高度（cm）。"""
        if not len(self.faces):
            return np.zeros(0)
        return self.vertices[self.faces].mean(axis=1)[:, 2]

    def height_levels(self, tol: float = 5.0) -> np.ndarray:
        """把高度归并成若干层，返回升序的层高。"""
        if not len(self.faces):
            return np.zeros(0)
        z = np.sort(self.heights())
        levels = [z[0]]
        for value in z[1:]:
            if value - levels[-1] > tol:
                levels.append(value)
        return np.asarray(levels)

    def summary(self) -> str:
        lo, hi = self.bounds()
        levels = self.height_levels()
        return (
            f"FieldMesh({self.num_vertices} 顶点 / {self.num_faces} 面 | "
            f"地面 z={self.ground_z:.1f}cm | "
            f"地面面 {int(self.ground_mask.sum())} / "
            f"可站立 {int(self.walkable_mask.sum())} / "
            f"遮挡面 {int(self.occluder_mask.sum())} | "
            f"包围盒 x[{lo[0]:.0f},{hi[0]:.0f}] y[{lo[1]:.0f},{hi[1]:.0f}] "
            f"z[{lo[2]:.0f},{hi[2]:.0f}] | 高度层 {np.round(levels, 1).tolist()})"
        )

    def save_npz(self, path: str | Path) -> Path:
        """存成 npz，避免运行时再解析 PLY。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            vertices=self.vertices.astype(np.float32),
            faces=self.faces.astype(np.int32),
            ground_mask=self.ground_mask,
            walkable_mask=self.walkable_mask,
            occluder_mask=self.occluder_mask,
            ground_z=np.array(self.ground_z),
            source=np.array(self.source),
        )
        return p

    @classmethod
    def load_npz(cls, path: str | Path) -> "FieldMesh":
        d = np.load(Path(path), allow_pickle=False)
        return cls(
            vertices=np.asarray(d["vertices"], dtype=np.float64),
            faces=np.asarray(d["faces"], dtype=np.int64),
            ground_mask=np.asarray(d["ground_mask"], dtype=bool),
            walkable_mask=np.asarray(d["walkable_mask"], dtype=bool),
            occluder_mask=np.asarray(d["occluder_mask"], dtype=bool),
            ground_z=float(d["ground_z"]),
            source=str(d["source"]),
        )


def save_ply(path: str | Path, mesh: RawMesh | FieldMesh) -> Path:
    """写 binary_little_endian PLY（float32 顶点 + uint32 面）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.uint32)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Created by JGRadar\n"
        f"element vertex {len(v)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {len(f)}\n"
        "property list uchar uint vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    face_rec = np.dtype([("n", "u1"), ("i", "<u4", 3)])
    faces = np.empty(len(f), dtype=face_rec)
    faces["n"] = 3
    faces["i"] = f
    with p.open("wb") as fh:
        fh.write(header)
        fh.write(v.tobytes())
        fh.write(faces.tobytes())
    return p


def estimate_ground_z(
    vertices: np.ndarray, faces: np.ndarray, *, horizontal_cos: float = 0.5
) -> float:
    """估计地面高度：取近水平面里面积加权的中位高度。

    比"所有顶点的最小 z"稳：地板网格里常混着几块下沉的排水沟或浮空杂质。
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(f):
        return 0.0
    normals = face_normals(v, f)
    areas = face_areas(v, f)
    z = v[f].mean(axis=1)[:, 2]

    horizontal = np.abs(normals[:, 2]) > horizontal_cos
    if horizontal.sum() == 0:
        return float(np.median(z))
    z_h, a_h = z[horizontal], areas[horizontal]
    order = np.argsort(z_h)
    cw = np.cumsum(a_h[order])
    if cw[-1] <= 0:
        return float(np.median(z_h))
    idx = int(np.searchsorted(cw, cw[-1] * 0.5))
    return float(z_h[order[min(idx, len(order) - 1)]])
