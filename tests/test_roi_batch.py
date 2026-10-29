"""letterbox 与 ROI 拼图的几何测试。

这个模块的代码**错了不会报错**：坐标整体偏半像素、ROI 映射少减一个裁剪
原点，跑起来一切正常，只是场上定位差几十厘米。所以这里不测"函数返回了
东西"，只测**几何是否与真实重采样一致**：

* 正逆变换必须严格互逆（纯数学，容差 1e-9）；
* 记下的变换必须与画布上**实际**的像素位置吻合（这才是会被静默弄错的部分，
  用"把标记物放进去、再去画布上找它在哪"来验，绕开公式本身）；
* 记录的 scale 是重采样实际用的那个，不是请求的那个。
"""

from __future__ import annotations

import numpy as np
import pytest

from jg_radar.detect.roi_batch import (
    LetterboxTransform,
    RoiBatch,
    RoiBatcher,
    letterbox,
)

PAD = 114


# ----------------------------------------------------------------------
# 辅助
# ----------------------------------------------------------------------
def _transform(scale_x=0.5, scale_y=0.5, off_x=10.0, off_y=20.0, **kw):
    return LetterboxTransform(
        scale_x=scale_x, scale_y=scale_y, offset_x=off_x, offset_y=off_y,
        src_w=kw.get("src_w", 640), src_h=kw.get("src_h", 480),
        dst_w=kw.get("dst_w", 640), dst_h=kw.get("dst_h", 480),
    )


def _marker_image(w=640, h=480, *, marker=None, size=40, bg=0):
    """在 ``marker`` 处放一个实心方块，其余为 ``bg``。"""
    img = np.full((h, w), bg, dtype=np.uint8)
    if marker is not None:
        cx, cy = marker
        x1, y1 = int(cx - size // 2), int(cy - size // 2)
        img[y1:y1 + size, x1:x1 + size] = 255
    return img


def _blob_centroid(img: np.ndarray, value=255) -> np.ndarray:
    """某个值的像素重心，返回 ``(x, y)``（注意与 ``np.argwhere`` 的 ``(y, x)`` 相反）。

    加 0.5 是**必须**的：下标 ``i`` 的像素覆盖 ``[i, i+1)``，重心在下标空间里
    算出来天然比坐标空间少半个像素。少了它，每个用例都会稳定偏 0.5px —— 一个
    看起来像"实现有 bug"、其实是测试自己写错的偏差。
    """
    ys, xs = np.nonzero(img == value)
    assert len(xs) > 0, "画布上找不到标记物"
    return np.array([xs.mean(), ys.mean()], dtype=np.float64) + 0.5


def _assert_marker_agrees(t, canvas, true_src, *, atol=0.5):
    """断言变换与画布上**实际**的像素位置一致。

    比较在**画布坐标系**里做，因为标记物的重心受重采样量化影响，误差上限
    是1个画布像素（块的边界各取整一次），这是与缩放无关的常量。换到原图
    坐标系去比，容差就得写成 ``1/scale``，ROI 被缩小时松到没有意义。
    """
    predicted = t.forward_points(np.asarray(true_src, dtype=np.float64).reshape(1, 2))[0]
    actual = _blob_centroid(canvas)
    assert np.allclose(actual, predicted, atol=atol), (
        f"标记物实际落在画布 {actual}，变换预测 {predicted}（差 {actual - predicted}）"
    )
    return t.inverse_points(actual)[0]


# ======================================================================
# LetterboxTransform —— 纯数学
# ======================================================================
def test_points_roundtrip_is_exact():
    t = _transform(scale_x=0.37, scale_y=0.91, off_x=13.0, off_y=-4.5)
    pts = np.array([[0.0, 0.0], [639.0, 479.0], [123.25, 456.75]])
    back = t.inverse_points(t.forward_points(pts))
    assert np.allclose(back, pts, atol=1e-9)


def test_boxes_roundtrip_is_exact():
    t = _transform(scale_x=2.0, scale_y=1.5, off_x=-8.0, off_y=3.0)
    box = (11.5, 22.25, 300.0, 400.75)
    assert np.allclose(t.inverse_box(t.forward_box(box)), box, atol=1e-9)


def test_box_forward_matches_point_forward():
    """框的四个角与点的映射必须同源，否则框会相对点缓慢漂移。"""
    t = _transform()
    x1, y1, x2, y2 = 10.0, 20.0, 110.0, 220.0
    pts = t.forward_points(np.array([[x1, y1], [x2, y2]]))
    fx1, fy1, fx2, fy2 = t.forward_box((x1, y1, x2, y2))
    assert np.allclose(pts, [[fx1, fy1], [fx2, fy2]])


def test_single_point_is_accepted():
    """``(2,)`` 也应被当成一个点，而不是被 reshape 成 ``(1,2)`` 后静默解错。"""
    t = _transform()
    got = t.forward_points(np.array([100.0, 50.0]))
    assert got.shape == (1, 2)
    assert np.allclose(got[0], (100.0 * 0.5 + 10.0, 50.0 * 0.5 + 20.0))


def test_scale_property_is_the_minor_axis():
    t = _transform(scale_x=0.64, scale_y=0.6436)
    assert t.scale == pytest.approx(0.64)


def test_inverse_box_does_not_clip():
    """越界框是信息（检测器看到了灰边），逆变换不该悄悄裁掉它。"""
    t = _transform(scale_x=1.0, scale_y=1.0, off_x=50.0, off_y=50.0)
    assert t.inverse_box((0.0, 0.0, 10.0, 10.0)) == (-50.0, -50.0, -40.0, -40.0)


def test_clip_box_clips_to_source():
    t = _transform(scale_x=1.0, scale_y=1.0, off_x=50.0, off_y=50.0, src_w=640, src_h=480)
    x1, y1, x2, y2 = t.clip_box((0.0, 0.0, 10_000.0, 10_000.0))
    assert (x1, y1, x2, y2) == (0.0, 0.0, 640.0, 480.0)


# ======================================================================
# letterbox —— 真实重采样
# ======================================================================
def test_canvas_shape_dtype_and_pad_value():
    img = np.full((30, 60), 200, dtype=np.uint8)
    out = letterbox(img, 128, 128)
    assert out.shape == (128, 128)
    assert out.dtype == np.uint8
    assert out[0, 0] == PAD


def test_color_image_keeps_channels():
    img = np.zeros((30, 60, 3), dtype=np.uint8)
    assert letterbox(img, 128, 128).shape == (128, 128, 3)


def test_content_is_centred_and_the_padding_is_where_it_should_be():
    """100×50 放进 200×200：等比 2 倍 → 200×100，上下各留 50 行灰边。"""
    img = np.full((50, 100), 200, dtype=np.uint8)
    out, t = letterbox(img, 200, 200, return_transform=True)

    assert t.content_region() == pytest.approx((0.0, 50.0, 200.0, 150.0))
    assert (out[:50] == PAD).all()
    assert (out[150:] == PAD).all()
    assert (out[50:150] == 200).all()


def test_recorded_scale_is_the_effective_one_not_the_requested_one():
    """300×101 → 192：请求比例 0.64，实际 y 被 round 到了 65px（0.64356）。

    这两者只有在整除时才相等。若把**请求的**比例记进变换，误差会在
    ``scale`` 很小（真实场景里就是 1280 整车缩到 640）时被放大成几个像素。
    """
    img = np.zeros((101, 300), dtype=np.uint8)
    _, t = letterbox(img, 192, 192, return_transform=True)
    assert t.scale_y == pytest.approx(65 / 101)
    assert t.scale_y != pytest.approx(0.64), "记成了请求比例，不是实际比例"
    assert t.scale_x == pytest.approx(0.64)
    # 内容高度必须与记下的 scale_y 一致
    assert t.content_region()[3] - t.content_region()[1] == pytest.approx(65.0)


def test_marker_lands_where_forward_points_says_it_does():
    """**核心断言**：变换不是自洽的公式，它必须描述画布上真实发生的事。

    放大 3 倍放置一个标记块，再去画布上找它的重心——重心必须落在正变换
    预测的位置上。这条一旦挂了，整个级联的坐标就全偏了。
    """
    img = _marker_image(120, 90, marker=(60, 45), size=30)
    out, t = letterbox(img, 360, 360, return_transform=True)
    _assert_marker_agrees(t, out, (60.0, 45.0))


def test_marker_maps_back_to_where_it_started():
    img = _marker_image(200, 150, marker=(130, 40), size=24)
    out, t = letterbox(img, 256, 256, return_transform=True)
    back = _assert_marker_agrees(t, out, (130.0, 40.0))
    # 观测值的重心最多偏 1 个画布像素，回到原图就是 1/scale 个原图像素
    assert np.allclose(back, (130.0, 40.0), atol=0.5 / t.scale_x + 0.25)


def test_identity_when_sizes_match():
    """同尺寸时不该有任何偏移——最近邻重采样会走 early-return。"""
    img = _marker_image(64, 64, marker=(20, 40), size=8)
    out, t = letterbox(img, 64, 64, return_transform=True)
    assert np.array_equal(out, img)
    assert (t.scale_x, t.scale_y, t.offset_x, t.offset_y) == (1.0, 1.0, 0.0, 0.0)


def test_letterbox_without_flag_returns_only_the_image():
    out = letterbox(np.zeros((10, 10), dtype=np.uint8), 32, 32)
    assert isinstance(out, np.ndarray)


@pytest.mark.parametrize("bad", [(0, 10), (10, 0), (0, 0)])
def test_empty_source_raises(bad):
    with pytest.raises(ValueError, match="源尺寸非法"):
        letterbox(np.zeros(bad, dtype=np.uint8), 32, 32)


@pytest.mark.parametrize("bad", [(0, 32), (32, 0)])
def test_empty_destination_raises(bad):
    with pytest.raises(ValueError, match="目标尺寸非法"):
        letterbox(np.zeros((10, 10), dtype=np.uint8), *bad)


def test_bad_ndim_raises():
    with pytest.raises(ValueError, match="图像应为"):
        letterbox(np.zeros((4, 4, 3, 1), dtype=np.uint8), 16, 16)


# ======================================================================
# RoiBatcher —— 拼图与坐标合成
# ======================================================================
def test_slots_follow_row_major_order():
    batcher = RoiBatcher(tile_size=32, grid=4)
    img = np.zeros((400, 400), dtype=np.uint8)
    rois = [(i * 20, i * 20, i * 20 + 40, i * 20 + 40) for i in range(16)]
    batch = batcher.build(img, rois)

    assert len(batch) == 16
    for tile in batch.tiles:
        assert (tile.row, tile.col) == divmod(tile.index, 4)
        # 该 ROI 的中心必须映射进它自己那一格，而不是隔壁
        cx = 0.5 * (tile.source_box[0] + tile.source_box[2])
        cy = 0.5 * (tile.source_box[1] + tile.source_box[3])
        px, py = tile.transform.forward_points(np.array([[cx, cy]]))[0]
        oy, ox = tile.row * 32, tile.col * 32
        assert ox <= px <= ox + 32 and oy <= py <= oy + 32, (
            f"ROI {tile.index} 的中心映射到了 ({px:.1f}, {py:.1f})，"
            f"落在格子 (row={tile.row}, col={tile.col}) 之外"
        )


def test_canvas_size_and_capacity():
    batcher = RoiBatcher(tile_size=192, grid=4)
    assert batcher.capacity == 16
    assert batcher.canvas_size == 768


def test_each_roi_lands_in_its_own_tile():
    """每个 ROI 放一个**不同**的灰度值，看它出现在哪一块里。"""
    batcher = RoiBatcher(tile_size=64, grid=2)  # 画布 128×128，容量 4
    img = np.zeros((200, 200), dtype=np.uint8)
    rois = []
    for i in range(4):
        x1, y1 = 10 + i * 45, 20
        img[y1:y1 + 30, x1:x1 + 30] = 40 * (i + 1)
        rois.append((x1, y1, x1 + 30, y1 + 30))

    batch = batcher.build(img, rois)
    assert len(batch) == 4
    for tile in batch.tiles:
        oy, ox = tile.row * 64, tile.col * 64
        block = batch.canvas[oy:oy + 64, ox:ox + 64]
        assert (block == 40 * (tile.index + 1)).any(), f"ROI {tile.index} 不在自己的格子里"
        # 其余格子不该有它
        for other in batch.tiles:
            if other.index == tile.index:
                continue
            oy2, ox2 = other.row * 64, other.col * 64
            assert not (batch.canvas[oy2:oy2 + 64, ox2:ox2 + 64] == 40 * (tile.index + 1)).any()


def test_marker_roundtrips_through_crop_letterbox_and_canvas():
    """端到端的几何验证：原图标记 → 拼图 → 画布上找到 → 映射回原图。

    这条覆盖了裁剪原点、缩放、拼图偏移三级的合成。任何一级漏了原点修正，
    误差都会远超 1 像素。
    """
    batcher = RoiBatcher(tile_size=192, grid=4)
    src = _marker_image(800, 600, marker=(430, 260), size=36)
    roi = (350, 200, 520, 330)  # 120×170 的框，放大到 192

    batch = batcher.build(src, [roi])
    tile = batch.tiles[0]
    back = _assert_marker_agrees(tile.transform, batch.canvas, (430.0, 260.0))

    # 且必须落在该 ROI 的框内——映射对了但落错框，说明裁剪原点没减对
    x1, y1, x2, y2 = tile.source_box
    assert x1 <= back[0] <= x2 and y1 <= back[1] <= y2


def test_marker_roundtrips_when_roi_is_downscaled():
    """ROI 比瓦片大（缩小）时同样要准——真实场景里整车框就是这样。

    缩小是最容易漏掉的一条：``scale < 1`` 会把原图里的偏差压小，看起来
    "误差很小"，但反过来映射回原图时同样的偏差会被放大 4 倍多。
    """
    batcher = RoiBatcher(tile_size=64, grid=4)
    src = _marker_image(1280, 720, marker=(820, 400), size=120)
    roi = (700, 300, 980, 520)

    batch = batcher.build(src, [roi])
    tile = batch.tiles[0]
    assert tile.transform.scale_x < 0.5, "这个用例要的就是缩小路径"
    back = _assert_marker_agrees(tile.transform, batch.canvas, (820.0, 400.0))
    assert np.allclose(back, (820.0, 400.0), atol=0.5 / tile.transform.scale_x + 0.25)


def test_forward_and_inverse_are_exact_for_every_tile():
    """合成出来的变换必须仍是严格可逆的——即使它由三级不同比例拼成。"""
    batcher = RoiBatcher(tile_size=96, grid=3)
    src = np.zeros((500, 700), dtype=np.uint8)
    rois = [(10, 10, 110, 210), (300, 50, 400, 90), (600, 400, 700, 500)]
    batch = batcher.build(src, rois)
    assert len(batch) == 3
    for tile in batch.tiles:
        pts = np.array([[0.0, 0.0], [400.0, 300.0], [123.5, 77.25]])
        assert np.allclose(tile.transform.inverse_points(tile.transform.forward_points(pts)),
                           pts, atol=1e-9)


def test_roi_touching_the_border_is_clipped_not_raised():
    """一级检测的框越界是常态（车贴着画面边缘），不能抛异常。"""
    batcher = RoiBatcher(tile_size=64, grid=2)
    src = np.zeros((200, 300), dtype=np.uint8)
    batch = batcher.build(src, [(-50, -30, 100, 80), (250, 150, 400, 300)])
    assert len(batch) == 2
    assert batch.tiles[0].source_box == (0.0, 0.0, 100.0, 80.0)
    assert batch.tiles[1].source_box == (250.0, 150.0, 300.0, 200.0)


def test_fully_out_of_frame_roi_is_skipped():
    batcher = RoiBatcher(tile_size=32, grid=2)
    batch = batcher.build(np.zeros((100, 100), dtype=np.uint8),
                          [(-500, -500, -400, -400), (200, 200, 300, 300)])
    assert batch.empty


@pytest.mark.parametrize("degenerate", [
    (10.0, 10.0, 10.0, 50.0),   # 零宽
    (10.0, 10.0, 50.0, 10.0),   # 零高
    (10.0, 10.0, 10.5, 50.0),   # 亚像素宽
    (50.0, 50.0, 10.0, 10.0),   # 反了的框（x2 < x1）
])
def test_degenerate_rois_are_skipped(degenerate):
    """退化框喂给网络只会浪费一次推理，跳过并**保留其余框的下标**。"""
    batcher = RoiBatcher(tile_size=32, grid=2)
    src = np.zeros((200, 200), dtype=np.uint8)
    batch = batcher.build(src, [degenerate, (20, 20, 80, 80)])
    assert len(batch) == 1
    assert batch.tiles[0].index == 1, "跳过一个框后，后面框的下标必须还是它本来的下标"


def test_index_tracks_the_input_list_not_the_slot():
    """第一框退化时，第二个框进的是**第 0 格**，但它的 index 仍是 1。

    index 用来回填检测结果，槽位只用来布局——两者混为一谈会让结果串到
    别人的 ROI 上，而且串得毫无征兆。
    """
    batcher = RoiBatcher(tile_size=32, grid=4)
    src = np.zeros((200, 200), dtype=np.uint8)
    batch = batcher.build(src, [(0, 0, 0, 0), (20, 20, 80, 80), (100, 100, 150, 150)])
    assert [t.index for t in batch.tiles] == [1, 2]
    assert [(t.row, t.col) for t in batch.tiles] == [(0, 0), (0, 1)]


def test_capacity_overflow_drops_the_extras():
    batcher = RoiBatcher(tile_size=16, grid=2)  # 容量 4
    src = np.zeros((600, 600), dtype=np.uint8)
    rois = [(i * 50, i * 50, i * 50 + 40, i * 50 + 40) for i in range(9)]
    batch = batcher.build(src, rois)
    assert len(batch) == 4
    assert [t.index for t in batch.tiles] == [0, 1, 2, 3]


def test_empty_roi_list_gives_an_empty_batch_with_a_valid_canvas():
    batcher = RoiBatcher(tile_size=32, grid=2)
    batch = batcher.build(np.zeros((100, 100), dtype=np.uint8), [])
    assert batch.empty
    assert len(batch) == 0
    assert batch.canvas.shape == (64, 64)
    assert (batch.canvas == PAD).all(), "空批次应当是一张纯灰画布，而不是未初始化的内存"


def test_by_index_and_map_back():
    batcher = RoiBatcher(tile_size=64, grid=2)
    src = np.zeros((300, 300), dtype=np.uint8)
    batch = batcher.build(src, [(10, 10, 110, 110), (150, 150, 250, 250)])

    assert set(batch.by_index()) == {0, 1}
    tile = batch.by_index()[1]
    # 一个正好盖住该 ROI 的画布框，映射回来应当就是那个 ROI（已裁到图内）
    box = tile.transform.forward_box(tile.source_box)
    got = batch.map_back(tile, [box])[0]
    assert np.allclose(got, tile.source_box, atol=1e-6)


def test_map_back_clips_to_the_source_image():
    """映射回原图后越界的框要裁到图内，否则下游会拿它去索引数组。"""
    batcher = RoiBatcher(tile_size=64, grid=2)
    src = np.zeros((100, 100), dtype=np.uint8)
    batch = batcher.build(src, [(80, 80, 100, 100)])
    tile = batch.tiles[0]
    # 一个远大于 ROI 的画布框
    got = batch.map_back(tile, [(-10_000.0, -10_000.0, 10_000.0, 10_000.0)])[0]
    assert got == (0.0, 0.0, 100.0, 100.0)


def test_iter_tiles_yields_every_tile():
    batcher = RoiBatcher(tile_size=32, grid=2)
    src = np.zeros((200, 200), dtype=np.uint8)
    batch = batcher.build(src, [(10, 10, 60, 60), (100, 100, 150, 150)])
    assert [t.index for t in batch.iter_tiles()] == [0, 1]


def test_batch_carries_the_source_size():
    batcher = RoiBatcher(tile_size=32, grid=2)
    batch = batcher.build(np.zeros((480, 640), dtype=np.uint8), [])
    assert batch.source_size == (640, 480)
    assert isinstance(batch, RoiBatch)


def test_color_image_through_the_batcher():
    """三通道要走通，且填充色不能退化成黑。

    ROI 特意取成扁的（60×20），这样等比放进 32×32 的瓦片后上下必然留边——
    正方形的 ROI 会把瓦片填满，根本测不到填充色。
    """
    batcher = RoiBatcher(tile_size=32, grid=2)
    src = np.zeros((100, 100, 3), dtype=np.uint8)
    src[40:60, 40:60] = (255, 0, 0)
    batch = batcher.build(src, [(30, 40, 90, 60)])
    assert batch.canvas.shape == (64, 64, 3)
    assert (batch.canvas == np.array([255, 0, 0], dtype=np.uint8)).all(axis=-1).any()

    pad = np.array([PAD, PAD, PAD], dtype=np.uint8)
    assert (batch.canvas == pad).all(axis=-1).any(), "瓦片里应有灰边"
    # 灰边是**填充色**而不是黑：纯黑边会在画面边缘造出一圈假强梯度
    assert (batch.canvas[0, 0] == pad).all()
