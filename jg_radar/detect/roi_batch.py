"""ROI 裁切、letterbox 缩放和坐标回映射。

正逆变换共享实际重采样后的 scale/offset，记录 ROI 格子与有效图像区域。
画布是一种坐标组织方式；定尺寸后端通过 infer_roi_batch 读取独立瓦片，
支持拼图的后端可以读取整幅画布，两种输入布局不能混用。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator, Sequence

import numpy as np

__all__ = [
    "LetterboxTransform",
    "RoiBatch",
    "RoiBatcher",
    "RoiTile",
    "letterbox",
    "infer_roi_batch",
]


def infer_roi_batch(backend, batch, *, infer=None):
    """独立瓦片后端按 N 维批量推理，再恢复到现有拼图坐标协议。"""
    run = infer if infer is not None else backend.infer
    if getattr(backend, "supports_tiling", True):
        return run([batch.canvas])[0]
    images = []
    for tile in batch.tiles:
        x1, y1, x2, y2 = map(int, tile.canvas_box)
        images.append(batch.canvas[y1:y2, x1:x2])
    results = run(images)
    if len(results) != len(images):
        raise ValueError("后端批量结果数与有效 ROI 数不一致")
    records = []
    for tile, detections in zip(batch.tiles, results):
        ox, oy, _, _ = tile.canvas_box
        for rec in detections:
            cx, cy = rec.center
            if not (0 <= cx < batch.tile_size and 0 <= cy < batch.tile_size):
                continue
            x1, y1, x2, y2 = rec.bbox
            records.append(replace(rec, bbox=(x1 + ox, y1 + oy, x2 + ox, y2 + oy)))
    return records


# ======================================================================
# letterbox
# ======================================================================
@dataclass(frozen=True)
class LetterboxTransform:
    """一次 letterbox 的完整记录。

    坐标取**像素角点**约定：像素 ``i`` 覆盖 ``[i, i+1)``。于是映射就是

        ``dst = src * scale + offset``

    两个轴各一份，因为等比缩放到整数尺寸后两轴的实际比例会差一点点。
    """

    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float
    src_w: int
    src_h: int
    dst_w: int
    dst_h: int

    @property
    def scale(self) -> float:
        """两轴比例的较小者。等比方框缩放时 ``scale_x == scale_y``。"""
        return min(self.scale_x, self.scale_y)

    def forward_points(self, pts: np.ndarray) -> np.ndarray:
        """原图坐标 (N,2) → 画布坐标。"""
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return p * np.array([self.scale_x, self.scale_y]) + np.array(
            [self.offset_x, self.offset_y]
        )

    def inverse_points(self, pts: np.ndarray) -> np.ndarray:
        """画布坐标 (N,2) → 原图坐标。``forward_points`` 的严格逆。"""
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return (p - np.array([self.offset_x, self.offset_y])) / np.array(
            [self.scale_x, self.scale_y]
        )

    def forward_box(self, box: Sequence[float]) -> tuple[float, float, float, float]:
        """``(x1,y1,x2,y2)`` 原图 → 画布。"""
        x1, y1, x2, y2 = (float(v) for v in box)
        return (
            x1 * self.scale_x + self.offset_x,
            y1 * self.scale_y + self.offset_y,
            x2 * self.scale_x + self.offset_x,
            y2 * self.scale_y + self.offset_y,
        )

    def inverse_box(self, box: Sequence[float]) -> tuple[float, float, float, float]:
        """``(x1,y1,x2,y2)`` 画布 → 原图。

        不裁剪：越界框是**信息**（说明检测器看到了灰边或画面外），裁掉就把它
        丢了。需要裁剪的地方显式调 ``clip_box``。
        """
        x1, y1, x2, y2 = (float(v) for v in box)
        return (
            (x1 - self.offset_x) / self.scale_x,
            (y1 - self.offset_y) / self.scale_y,
            (x2 - self.offset_x) / self.scale_x,
            (y2 - self.offset_y) / self.scale_y,
        )

    def clip_box(self, box: Sequence[float]) -> tuple[float, float, float, float]:
        """把原图坐标的框裁到源图范围内。"""
        x1, y1, x2, y2 = self.inverse_box(box)
        return (
            min(max(x1, 0.0), float(self.src_w)),
            min(max(y1, 0.0), float(self.src_h)),
            min(max(x2, 0.0), float(self.src_w)),
            min(max(y2, 0.0), float(self.src_h)),
        )

    def content_region(self) -> tuple[float, float, float, float]:
        """画布上真正承载图像的那块区域，其余是填充。"""
        return (
            self.offset_x,
            self.offset_y,
            self.offset_x + self.src_w * self.scale_x,
            self.offset_y + self.src_h * self.scale_y,
        )

    def contains_content(self, pts: np.ndarray, *, margin: float = 0.0) -> np.ndarray:
        """画布坐标的点是否落在**有图像**的区域里。

        检测器在灰边上的输出是纯噪声，映射回原图后坐标看着正常、指向的地方
        什么都没有——这是最难查的一类假检测，必须在画布坐标系里就拦掉
        （只有在这里，"灰边"才是个有明确定义的区域）。
        """
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        x1, y1, x2, y2 = self.content_region()
        return (
            (p[:, 0] >= x1 - margin) & (p[:, 0] <= x2 + margin)
            & (p[:, 1] >= y1 - margin) & (p[:, 1] <= y2 + margin)
        )


def letterbox(
    image: np.ndarray,
    dst_w: int,
    dst_h: int,
    *,
    pad_value: int = 114,
    return_transform: bool = False,
):
    """等比缩放并居中填充到 ``dst_w × dst_h``。

    ``pad_value=114`` 是 YOLO 系列的惯例灰（不是 0）：纯黑边会让网络在画面
    边缘学出一圈虚假的强梯度。

    默认只返回图像；传 ``return_transform=True`` 时返回 ``(image, transform)``。
    """
    img = np.asarray(image)
    if img.ndim not in (2, 3):
        raise ValueError(f"图像应为 (H,W) 或 (H,W,C)，得到 {img.shape}")
    src_h, src_w = img.shape[:2]
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"源尺寸非法：{src_w}×{src_h}")
    if dst_w <= 0 or dst_h <= 0:
        raise ValueError(f"目标尺寸非法：{dst_w}×{dst_h}")

    ratio = min(dst_w / float(src_w), dst_h / float(src_h))
    new_w = max(1, min(dst_w, int(round(src_w * ratio))))
    new_h = max(1, min(dst_h, int(round(src_h * ratio))))
    off_x = (dst_w - new_w) // 2
    off_y = (dst_h - new_h) // 2

    canvas = np.full((dst_h, dst_w) + img.shape[2:], pad_value, dtype=img.dtype)
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = _resize_nearest(img, new_w, new_h)

    t = LetterboxTransform(
        scale_x=new_w / float(src_w),
        scale_y=new_h / float(src_h),
        offset_x=float(off_x),
        offset_y=float(off_y),
        src_w=src_w, src_h=src_h, dst_w=dst_w, dst_h=dst_h,
    )
    return (canvas, t) if return_transform else canvas


def _resize_nearest(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """最近邻缩放。

    刻意不用双线性：这里要的是**坐标可复现**，不是画质。最近邻在合成 fixture
    上是精确可逆的，测试才能断言像素级的几何正确性；真实推理路径上也一样——
    检测网络对这点重采样差异不敏感，而坐标一致性是刚需。
    """
    src_h, src_w = img.shape[:2]
    if (src_w, src_h) == (w, h):
        return img
    ys = np.minimum((np.arange(h) + 0.5) * src_h / h, src_h - 1).astype(np.intp)
    xs = np.minimum((np.arange(w) + 0.5) * src_w / w, src_w - 1).astype(np.intp)
    return img[ys][:, xs]


# ======================================================================
# ROI 拼图
# ======================================================================
@dataclass(frozen=True)
class RoiTile:
    """画布上的一块 ROI 及其坐标映射。

    同时记着**两个**方框，它们回答两个不同的问题，混用会安静地出错：

    * ``canvas_box``——这块 ROI 在画布上占了哪一格。格子之间互不重叠，
      所以"这个框属于哪块瓦片"用它判，答案是唯一的。
    * ``valid_box``——格子里面真正有图像的那部分（其余是 letterbox 的灰边）。
      "这个框是真的看到了东西，还是网络在灰边上瞎编"用它判。

    而 ``transform.content_region()`` 对瓦片**没有意义**：它描述的是"整张
    原图经过这块瓦片的变换后落在画布的哪里"，因为 ROI 通常被放大好几倍，
    那个区域会远远超出格子、盖住几乎整张画布。拿它当瓦片的有效范围，每块
    瓦片都会认领所有检测。
    """

    index: int
    """对应输入 ROI 列表里的下标。"""
    row: int
    col: int
    source_box: tuple[float, float, float, float]
    """该 ROI 在原图里的位置（已裁到图内）。"""
    canvas_box: tuple[float, float, float, float]
    """该瓦片在画布上占的格子，互不重叠。"""
    valid_box: tuple[float, float, float, float]
    """格子内有图像的部分（``canvas_box`` 减去 letterbox 灰边）。"""
    transform: LetterboxTransform
    """**原图 → 画布**的变换，已把裁剪、缩放、拼图三步合成一次。"""

    def owns(self, x: float, y: float) -> bool:
        """点是否落在这块瓦片的格子里（画布坐标）。"""
        return (self.canvas_box[0] <= x < self.canvas_box[2]
                and self.canvas_box[1] <= y < self.canvas_box[3])

    def has_content(self, x: float, y: float, *, margin: float = 0.0) -> bool:
        """点是否落在这块瓦片**有图像**的部分里（画布坐标）。"""
        return (self.valid_box[0] - margin <= x <= self.valid_box[2] + margin
                and self.valid_box[1] - margin <= y <= self.valid_box[3] + margin)

    def to_source_box(self, box: Sequence[float]) -> tuple[float, float, float, float]:
        return self.transform.inverse_box(box)


@dataclass
class RoiBatch:
    """一次拼图的结果。"""

    canvas: np.ndarray
    tiles: list[RoiTile]
    tile_size: int
    grid: int
    source_size: tuple[int, int]
    """原图 ``(w, h)``，用于把映射回来的框裁到图内。"""

    def __len__(self) -> int:
        return len(self.tiles)

    @property
    def empty(self) -> bool:
        return not self.tiles

    def iter_tiles(self) -> Iterator[RoiTile]:
        return iter(self.tiles)

    def by_index(self) -> dict[int, RoiTile]:
        return {t.index: t for t in self.tiles}

    def tile_at(self, x: float, y: float) -> RoiTile | None:
        """点（画布坐标）落在哪块瓦片里。格子的划分是完备且不重叠的，
        所以答案唯一；落在瓦片之间的缝里或画布外时返回 ``None``。"""
        for t in self.tiles:
            if t.owns(x, y):
                return t
        return None

    def tile_with_content_at(self, x: float, y: float) -> RoiTile | None:
        """点落在哪块瓦片的**有效画面**里。用于排除灰边上的假检测。"""
        tile = self.tile_at(x, y)
        return tile if tile is not None and tile.has_content(x, y) else None

    def map_back(self, tile: RoiTile, boxes) -> list[tuple[float, float, float, float]]:
        """把某一块上的若干框映射回原图并裁剪。"""
        w, h = float(self.source_size[0]), float(self.source_size[1])
        out = []
        for b in boxes:
            x1, y1, x2, y2 = tile.to_source_box(b)
            out.append(
                (
                    min(max(x1, 0.0), w),
                    min(max(y1, 0.0), h),
                    min(max(x2, 0.0), w),
                    min(max(y2, 0.0), h),
                )
            )
        return out


@dataclass
class RoiBatcher:
    """把若干 ROI 裁切、缩放、拼到一张 ``grid × grid`` 画布上。

    ``grid`` 默认 4 → 一张画布最多 16 块 ROI，与方案 §五 一致。
    """

    tile_size: int = 192
    grid: int = 4
    pad_value: int = 114

    @property
    def capacity(self) -> int:
        return self.grid * self.grid

    @property
    def canvas_size(self) -> int:
        return self.tile_size * self.grid

    def build(self, image: np.ndarray, rois: Sequence[Sequence[float]]) -> RoiBatch:
        """裁切 ``rois`` 并拼图，直到画布填满（``capacity`` 块）。

        **槽位是紧凑的**：退化的 ROI 被跳过时不会在画布上留一个空洞，后面的
        ROI 会顶上来。留洞没有任何好处——那块地方照样要跑一次推理，白白少
        装一块。代价是"哪些 ROI 被丢掉了"取决于内容而不只是数量，所以
        ``RoiTile.index`` 是回填结果的唯一依据，别拿它在列表里的位置当index。
        """
        img = np.asarray(image)
        src_h, src_w = img.shape[:2]
        canvas = np.full(
            (self.canvas_size, self.canvas_size) + img.shape[2:],
            self.pad_value,
            dtype=img.dtype,
        )
        tiles: list[RoiTile] = []

        for index, roi in enumerate(rois):
            if len(tiles) >= self.capacity:
                break
            x1, y1, x2, y2 = (float(v) for v in roi)
            # 先裁到图内：ROI 来自一级检测，越界是常态而不是异常
            x1c = min(max(x1, 0.0), float(src_w))
            x2c = min(max(x2, 0.0), float(src_w))
            y1c = min(max(y1, 0.0), float(src_h))
            y2c = min(max(y2, 0.0), float(src_h))
            if x2c - x1c < 1.0 or y2c - y1c < 1.0:
                continue  # 退化 ROI，跳过比喂给网络强

            ix1, iy1 = int(np.floor(x1c)), int(np.floor(y1c))
            crop = img[iy1:int(np.ceil(y2c)), ix1:int(np.ceil(x2c))]
            if crop.size == 0:
                continue

            row, col = divmod(len(tiles), self.grid)
            oy, ox = row * self.tile_size, col * self.tile_size

            tile_img, inner = letterbox(
                crop, self.tile_size, self.tile_size, pad_value=self.pad_value,
                return_transform=True,
            )
            canvas[oy:oy + self.tile_size, ox:ox + self.tile_size] = tile_img

            # 瓦片内的有效画面（灰边不算），平移到画布坐标
            icx1, icy1, icx2, icy2 = inner.content_region()
            valid_box = (ox + icx1, oy + icy1, ox + icx2, oy + icy2)

            # 原图 → 画布 =（原图 → 裁剪图）→（裁剪图 → 瓦片）→（瓦片 → 画布），
            # 三级都是纯缩放平移，合成成**一次**变换，别每级各逆一次——
            # 那样误差会逐级累积，而 ROI 的 scale 往往很大（小框放大好几倍）。
            tiles.append(
                RoiTile(
                    index=index,
                    row=row,
                    col=col,
                    source_box=(x1c, y1c, x2c, y2c),
                    canvas_box=(float(ox), float(oy),
                                float(ox + self.tile_size), float(oy + self.tile_size)),
                    valid_box=valid_box,
                    transform=LetterboxTransform(
                        scale_x=inner.scale_x,
                        scale_y=inner.scale_y,
                        offset_x=inner.offset_x - ix1 * inner.scale_x + ox,
                        offset_y=inner.offset_y - iy1 * inner.scale_y + oy,
                        src_w=src_w,
                        src_h=src_h,
                        dst_w=self.canvas_size,
                        dst_h=self.canvas_size,
                    ),
                )
            )
        return RoiBatch(canvas, tiles, self.tile_size, self.grid, (src_w, src_h))
