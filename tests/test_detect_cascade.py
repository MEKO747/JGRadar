"""级联几何测试 —— 用真实检测验坐标，不靠权重。

``BlobBackend`` 找出画面上真实的色块，于是"放一个方块 → 走完整条级联
→ 断言回来的坐标就是那个方块的位置"成为一条**无需任何模型权重**的端到端
测试。这正是 ``StubBackend`` 做不到的事：它的框是内容哈希出来的随机数，
没有真值可比，只能测流程能不能跑通。

真实权重到位后这些用例依然有价值——它们是级联坐标系的一条不依赖模型的
基准线。回归挂了时，能立刻分清是"坐标错了"还是"模型不行"。
"""

from __future__ import annotations

import numpy as np
import pytest

from jg_radar.detect import (
    UNKNOWN_CLASS,
    ArmorDetector,
    CarDetector,
    ClassMap,
    DetectorSettings,
    DigitClassifier,
    letterbox,
)
from jg_radar.detect.backends import BlobBackend, EmptyBackend, OnnxBackend, StubBackend
from jg_radar.detect.protocol import DetectionRecord

WHITE = (255, 255, 255)
INPUT = 640


# ----------------------------------------------------------------------
# 合成图
# ----------------------------------------------------------------------
def frame(w=1280, h=720, *, cars=(), armour=(), bg=0) -> np.ndarray:
    """画一帧：``cars`` 是白色整车块，``armour`` 是车内的灰色装甲板。

    装甲板用 200 而不是 255，这样 BlobBackend（找 255）不会把装甲板当成车。
    """
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    for x1, y1, x2, y2 in cars:
        img[y1:y2, x1:x2] = WHITE
    for x1, y1, x2, y2 in armour:
        img[y1:y2, x1:x2] = 200
    return img


def blob_backend(**kw) -> BlobBackend:
    kw.setdefault("target", WHITE)
    kw.setdefault("tol", 20)
    kw.setdefault("min_area", 8)
    kw.setdefault("input_size", INPUT)
    b = BlobBackend(**kw)
    b.load()
    return b


def car_detector(backend=None, **kw) -> CarDetector:
    backend = backend if backend is not None else blob_backend(name="car_blob")
    kw.setdefault("settings", DetectorSettings(conf_thresh=0.05, min_box_px=2.0))
    return CarDetector(backend, **kw)


def gray_backend(**kw) -> BlobBackend:
    """找灰色 200 的装甲板。"""
    kw.setdefault("target", (200, 200, 200))
    kw.setdefault("tol", 20)
    kw.setdefault("min_area", 4)
    kw.setdefault("input_size", 192)
    b = BlobBackend(**kw, name="armor_blob")
    b.load()
    return b


# ======================================================================
# 一级：整车
# ======================================================================
def test_car_box_matches_the_rectangle_that_was_drawn():
    """**核心**：画一个方块，检测框必须就是那个方块。"""
    det = car_detector()
    img = frame(cars=[(400, 300, 560, 460)])
    cars = det.detect(img)

    assert len(cars) == 1
    x1, y1, x2, y2 = cars[0].bbox
    assert (x1, y1, x2, y2) == pytest.approx((400, 300, 560, 460), abs=2.0)


def test_multiple_cars_keep_their_own_positions():
    det = car_detector()
    rects = [(100, 100, 220, 220), (600, 80, 760, 240), (900, 400, 1010, 510)]
    cars = det.detect(frame(cars=rects))

    assert len(cars) == 3
    got = sorted(c.bbox for c in cars)
    for g, want in zip(got, sorted(rects)):
        assert g == pytest.approx(want, abs=2.0)


def test_no_cars_gives_an_empty_list_not_an_error():
    assert car_detector().detect(frame()) == []


def test_empty_frame_is_handled():
    assert car_detector().detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_car_at_the_border_is_still_found():
    """贴着画面边缘的车：letterbox 会在旁边补灰边，映射回来必须裁到图内。"""
    det = car_detector()
    cars = det.detect(frame(cars=[(0, 0, 90, 90)]))
    assert len(cars) == 1
    x1, y1, x2, y2 = cars[0].bbox
    assert x1 >= 0 and y1 >= 0 and x2 <= 1280 and y2 <= 720
    assert (x1, y1) == pytest.approx((0, 0), abs=2.0)


def test_detect_rois_matches_detect():
    """``detect_rois`` 必须与 ``detect`` 用同一批框，不能各排一遍。"""
    det = car_detector()
    img = frame(cars=[(100, 100, 220, 220), (600, 80, 760, 240)])
    cars = det.detect(img)
    cars2, rois = det.detect_rois(img)
    assert [c.bbox for c in cars] == [c.bbox for c in cars2]
    assert rois.shape == (len(cars), 4)
    assert np.allclose(rois, [c.bbox for c in cars])


def test_detect_rois_empty_is_a_well_shaped_array():
    _, rois = car_detector().detect_rois(frame())
    assert rois.shape == (0, 4)


# ======================================================================
# 后处理：三类必须被拦掉的检测
# ======================================================================
class _FixedBackend:
    """返回固定的一批检测，用来精确构造各种边界情况。"""

    def __init__(self, records, input_size=INPUT, name="fixed"):
        self.records = list(records)
        self.input_size = input_size
        self.name = name
        self._loaded = False

    @property
    def is_ready(self):
        return self._loaded

    def load(self):
        self._loaded = True

    def warmup(self, n=20):
        pass

    def infer(self, images):
        return [list(self.records) for _ in images]


def test_detection_on_the_letterbox_padding_is_rejected():
    """灰边上的框必须被拦掉。

    这是最难查的一类假检测：它在画布上完全合法，映射回原图后的坐标也
    "看着正常"，但指向的地方根本没有东西——因为那块画面压根不存在。
    """
    # 1280×720 → 640×640：内容只占中间 640×360，上下各 140 行是灰边
    canvas, t = letterbox(frame(), INPUT, INPUT, return_transform=True)
    assert t.content_region() == pytest.approx((0.0, 140.0, 640.0, 500.0))

    on_pad = DetectionRecord(0, 0.9, (250.0, 20.0, 350.0, 100.0))      # 顶部灰边
    in_content = DetectionRecord(0, 0.9, (250.0, 300.0, 350.0, 360.0))  # 有效画面
    det = car_detector(_FixedBackend([on_pad, in_content]))

    box = det.detect(frame())[0].bbox
    # 只剩有效画面里那个
    assert len(det.detect(frame())) == 1
    assert box == pytest.approx(t.inverse_box(in_content.bbox), abs=1.0)


def test_low_confidence_is_rejected():
    low = DetectionRecord(0, 0.05, (250.0, 300.0, 350.0, 360.0))
    det = CarDetector(_FixedBackend([low]), settings=DetectorSettings(conf_thresh=0.25))
    assert det.detect(frame()) == []


def test_tiny_boxes_are_rejected():
    tiny = DetectionRecord(0, 0.9, (250.0, 300.0, 251.0, 301.0))
    det = CarDetector(_FixedBackend([tiny]), settings=DetectorSettings(min_box_px=4.0))
    assert det.detect(frame()) == []


def test_classes_outside_the_map_are_rejected():
    """类别表是适配层存在的理由：表外的 id 是噪声，不是"另一个类"。"""
    recs = [
        DetectionRecord(0, 0.9, (250.0, 300.0, 350.0, 360.0)),
        DetectionRecord(7, 0.9, (260.0, 310.0, 360.0, 370.0)),
    ]
    det = CarDetector(
        _FixedBackend(recs),
        class_map=ClassMap({0: "car"}),
        settings=DetectorSettings(min_box_px=2.0),
    )
    cars = det.detect(frame())
    assert len(cars) == 1
    assert cars[0].class_id == 0
    assert cars[0].label == "car"


def test_no_class_map_means_no_filtering():
    """不给表就不过滤——``ClassMap({})``（空表）会把一切都滤掉，
    表现为"检测器突然什么都不输出"，这个坑不值得留。"""
    recs = [DetectionRecord(7, 0.9, (250.0, 300.0, 350.0, 360.0))]
    det = CarDetector(_FixedBackend(recs), settings=DetectorSettings(min_box_px=2.0))
    assert len(det.detect(frame())) == 1


def test_nms_suppresses_duplicate_boxes():
    dup = [
        DetectionRecord(0, 0.9, (250.0, 300.0, 350.0, 360.0)),
        DetectionRecord(0, 0.7, (252.0, 302.0, 352.0, 362.0)),
        DetectionRecord(0, 0.5, (254.0, 304.0, 354.0, 364.0)),
    ]
    det = CarDetector(_FixedBackend(dup), settings=DetectorSettings(min_box_px=2.0))
    cars = det.detect(frame())
    assert len(cars) == 1
    assert cars[0].confidence == pytest.approx(0.9)


def test_nms_does_not_suppress_across_classes():
    """两个不同数字的框重叠是**歧义**，不是重复检测。

    压掉一个，跟踪层就只会看到一个干净的结论，永远不知道这里本来有歧义。
    """
    recs = [
        DetectionRecord(1, 0.9, (250.0, 300.0, 350.0, 360.0)),
        DetectionRecord(3, 0.8, (252.0, 302.0, 352.0, 362.0)),
    ]
    det = CarDetector(_FixedBackend(recs), settings=DetectorSettings(min_box_px=2.0))
    assert len(det.detect(frame())) == 2


def test_max_dets_truncates_to_the_best():
    recs = [
        DetectionRecord(0, c, (10.0 * i, 300.0, 10.0 * i + 20.0, 360.0))
        for i, c in enumerate([0.3, 0.9, 0.5, 0.7])
    ]
    det = CarDetector(
        _FixedBackend(recs), settings=DetectorSettings(min_box_px=2.0, max_dets=2)
    )
    cars = det.detect(frame())
    assert [c.confidence for c in cars] == pytest.approx([0.9, 0.7])


# ======================================================================
# 二级：装甲板
# ======================================================================
def test_armor_detections_land_on_the_armour_rectangles():
    """一辆车里两块装甲板，两块都要找回来、位置要对、且都挂在这辆车上。"""
    car = (400, 200, 700, 500)
    plates = [(450, 250, 520, 400), (600, 260, 670, 410)]
    img = frame(cars=[car], armour=plates)

    det = ArmorDetector(gray_backend(), settings=DetectorSettings(conf_thresh=0.05, min_box_px=2.0))
    hits = det.detect(img, [car])

    assert len(hits) == 2
    assert {h.car_index for h in hits} == {0}
    got = sorted(h.bbox for h in hits)
    for g, want in zip(got, sorted(plates)):
        assert g == pytest.approx(want, abs=2.5)


def test_armor_carries_the_right_car_index():
    """级联身份必须一路带下去——丢了它，跟踪层就无法知道哪块板属于哪辆车，
    而"哪块板属于哪辆车"是身份投票唯一的输入。"""
    cars = [(100, 100, 350, 350), (800, 150, 1050, 400)]
    plates = [(150, 160, 210, 300), (860, 200, 920, 340)]
    img = frame(cars=cars, armour=plates)

    det = ArmorDetector(gray_backend(), settings=DetectorSettings(conf_thresh=0.05, min_box_px=2.0))
    hits = det.detect(img, cars)

    assert len(hits) == 2
    by_car = {h.car_index: h.bbox for h in hits}
    assert set(by_car) == {0, 1}
    assert by_car[0] == pytest.approx(plates[0], abs=2.5)
    assert by_car[1] == pytest.approx(plates[1], abs=2.5)


def test_armor_of_the_other_car_is_not_attributed_to_this_one():
    """B 车的装甲板绝不能被算成 A 车的——这会把身份直接认错人。"""
    cars = [(100, 100, 300, 300), (700, 100, 900, 300)]
    img = frame(cars=cars, armour=[(850, 150, 890, 270)])

    det = ArmorDetector(gray_backend(), settings=DetectorSettings(conf_thresh=0.05, min_box_px=2.0))
    hits = det.detect(img, cars)

    assert len(hits) == 1
    assert hits[0].car_index == 1


def test_armor_with_no_rois_returns_nothing():
    det = ArmorDetector(gray_backend())
    assert det.detect(frame(), []) == []
    assert det.detect(frame(), np.zeros((0, 4))) == []


def test_armor_handles_more_rois_than_one_canvas():
    """超过一张画布容量（16 块）时自动分批，且 car_index 仍然指向**原始**下标。

    分批最典型的错误是每批重新从 0 计数，于是第 17 辆车被记成第 1 辆。
    """
    n = 20
    cars = [(i * 60 + 10, 400, i * 60 + 50, 520) for i in range(n)]
    # 装甲板在车框中央、四周留空——理由同 test_digit_classifier_order_survives_batching
    plates = [(i * 60 + 20, 450, i * 60 + 40, 470) for i in range(n)]
    img = frame(w=1280, cars=cars, armour=plates)

    det = ArmorDetector(gray_backend(), settings=DetectorSettings(conf_thresh=0.05, min_box_px=2.0))
    assert det.capacity == 16
    hits = det.detect(img, cars)

    assert len(hits) == n, f"20 块装甲板只找回 {len(hits)} 块"
    assert sorted(h.car_index for h in hits) == list(range(n))
    for h in hits:
        assert h.bbox == pytest.approx(plates[h.car_index], abs=3.0)


def test_tile_ownership_is_a_partition():
    """画布的瓦片之间**没有缝**，一个检测框横跨两块瓦片是常态。

    归属按框心落在哪个格子判定。格子必须构成一个**划分**（互不重叠、缝里
    没有归属），否则一块装甲板会被算到两辆车上——而"这块板属于哪辆车"是
    身份投票唯一的输入，重复计数会直接变成一张投错的票。
    """
    from jg_radar.detect.roi_batch import RoiBatcher

    batcher = RoiBatcher(tile_size=64, grid=4)
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    rois = [(i * 40, i * 40, i * 40 + 60, i * 40 + 60) for i in range(7)]
    batch = batcher.build(img, rois)

    # 每块瓦片的格子里，有且只有它自己认领
    for tile in batch.tiles:
        cx1, cy1, cx2, cy2 = tile.canvas_box
        for x in (cx1 + 0.5, (cx1 + cx2) * 0.5, cx2 - 0.5):
            for y in (cy1 + 0.5, (cy1 + cy2) * 0.5, cy2 - 0.5):
                owners = [t for t in batch.tiles if t.owns(x, y)]
                assert owners == [tile], f"画布点 ({x},{y}) 的归属不是唯一的：{owners}"

    # 一个横跨两块瓦片的框，按框心只归其中一块
    t0, t1 = batch.tiles[0], batch.tiles[1]
    spanning = (t0.canvas_box[2] - 8.0, 4.0, t1.canvas_box[0] + 8.0, 40.0)
    center = (0.5 * (spanning[0] + spanning[2]), 0.5 * (spanning[1] + spanning[3]))
    owner = batch.tile_at(*center)
    assert owner is not None
    assert sum(1 for t in batch.tiles if t.owns(*center)) == 1
    # 而框的左右两个端点各自落在不同的瓦片里——确认这个框真的是横跨的
    assert batch.tile_at(spanning[0], spanning[1]) is t0
    assert batch.tile_at(spanning[2], spanning[3]) is t1


def test_points_outside_every_tile_have_no_owner():
    """缝里和画布外必须返回 None，而不是硬塞给最近的一块。"""
    from jg_radar.detect.roi_batch import RoiBatcher

    batcher = RoiBatcher(tile_size=64, grid=4)  # 画布 256
    batch = batcher.build(np.zeros((100, 100, 3), dtype=np.uint8), [(10, 10, 80, 80)])
    assert batch.tile_at(500.0, 500.0) is None
    assert batch.tile_at(-1.0, 10.0) is None
    # 第一块瓦片占 [0,64)×[0,64)，没被占用的格子不该有归属
    assert batch.tile_at(200.0, 200.0) is None


# ======================================================================
# 三级：数字分类
# ======================================================================
def test_digit_classifier_preserves_input_order():
    """返回顺序必须与输入严格一致（失败的也要占位）。

    调用方是拿它和装甲板列表 zip 的；少一个元素会让后面所有装甲板都配上
    错误的数字，而且不会有任何报错。
    """
    boxes = [(100, 100, 160, 200), (300, 100, 360, 200), (500, 100, 560, 200)]
    plates = [(125, 130, 145, 170), (325, 130, 345, 170), (525, 130, 545, 170)]
    img = frame(w=800, h=320, armour=plates)

    clf = DigitClassifier(gray_backend(class_ids=[1, 3, 4]), min_confidence=0.01)
    clf.load()
    votes = clf.classify(img, boxes)

    assert len(votes) == 3
    assert all(v.known for v in votes)
    # 按面积排名分配：三块一样大，所以排名由 find_objects 的顺序决定——只断言
    # 三个类别互不相同，不断言谁是谁（那是后端细节，不是分类器的契约）
    assert len({v.class_id for v in votes}) == 3


def test_digit_classifier_returns_unknown_when_box_is_blank():
    img = frame(w=400, h=300)
    clf = DigitClassifier(gray_backend(), min_confidence=0.01)
    clf.load()
    votes = clf.classify(img, [(100, 100, 140, 180)])
    assert len(votes) == 1
    assert votes[0].class_id == UNKNOWN_CLASS
    assert not votes[0].known


def test_digit_classifier_unknown_is_not_a_real_class():
    """"读不出数字"必须用一个真实类别之外的 id 表示。

    用 0 表示"不知道"，下游的身份投票会把它当成一票（"数字 0"）投出去，
    票数还记在自己头上。
    """
    assert UNKNOWN_CLASS not in (0, 1, 2, 3)


def test_digit_classifier_rejects_classes_outside_the_map():
    plates = [(100, 100, 140, 180)]
    img = frame(w=400, h=300, armour=plates)

    backend = gray_backend(class_ids=[9])
    clf = DigitClassifier(backend, class_map=ClassMap({1: "1", 3: "3"}), min_confidence=0.01)
    clf.load()
    votes = clf.classify(img, plates)
    assert votes[0].class_id == UNKNOWN_CLASS


def test_digit_classifier_keeps_classes_inside_the_map():
    plates = [(100, 100, 140, 180)]
    img = frame(w=400, h=300, armour=plates)

    clf = DigitClassifier(gray_backend(class_ids=[3]), class_map=ClassMap({3: "3"}),
                          min_confidence=0.01)
    clf.load()
    votes = clf.classify(img, plates)
    assert votes[0].class_id == 3
    assert votes[0].label == "3"


def test_digit_classifier_empty_input():
    clf = DigitClassifier(gray_backend())
    assert clf.classify(frame(), []) == []


def test_digit_classifier_order_survives_batching():
    """跨批次时下标不能重新计数——第 17 个框的结论必须回到第 17 个位置。

    数字块**小于**它所在的框，四周留出背景。这是 `BlobBackend` 的硬性要求，
    而不是随便的构图：letterbox 必然让内容顶满瓦片的某两条边，而画布上的
    瓦片是紧挨着的、中间没有缝。物体一旦顶到瓦片边，相邻瓦片的物体就连成
    一片，连通域分析会把整行认成同一个块。真实情形下这个框就是车 ROI，
    里面的装甲板本来就比框小，所以留边也正是现实的样子。
    """
    n = 20
    boxes = [(i * 40, 100, i * 40 + 30, 150) for i in range(n)]
    plates = [(i * 40 + 9, 115, i * 40 + 21, 135) for i in range(n)]
    img = frame(w=800, h=300, armour=plates)

    clf = DigitClassifier(gray_backend(class_ids=[2]), min_confidence=0.01)
    clf.load()
    assert clf.batcher.capacity < n, "这个用例要的就是跨批次"
    votes = clf.classify(img, boxes)

    assert len(votes) == n
    assert all(v.known for v in votes), (
        f"第 {[i for i, v in enumerate(votes) if not v.known]} 个没读出数字"
    )
    assert all(v.class_id == 2 for v in votes)


def test_digit_classifier_returns_unknown_when_backend_is_not_loaded():
    """后端没装好时给 unknown，而不是抛异常或给一个假数字。"""
    backend = BlobBackend(target=(200, 200, 200), input_size=192)  # 故意不 load()
    assert not backend.is_ready
    clf = DigitClassifier(backend, min_confidence=0.01)
    votes = clf.classify(frame(w=400, h=300, armour=[(100, 100, 140, 180)]),
                         [(100, 100, 140, 180)])
    assert len(votes) == 1
    assert votes[0].class_id == UNKNOWN_CLASS


# ======================================================================
# 后端缺位时的降级
# ======================================================================
def test_missing_backend_degrades_to_no_detections():
    onnx = OnnxBackend("config/does_not_exist.onnx", input_size=INPUT)
    onnx.load()
    assert not onnx.is_ready
    det = CarDetector(onnx)
    assert not det.is_ready
    assert det.detect(frame(cars=[(100, 100, 300, 300)])) == []
    # 原因要能说清楚是缺库还是缺权重
    assert onnx.reason


def test_empty_backend_never_detects():
    b = EmptyBackend(input_size=INPUT)
    b.load()
    det = CarDetector(b)
    assert det.is_ready
    assert det.detect(frame(cars=[(100, 100, 300, 300)])) == []


def test_stub_backend_is_deterministic_across_runs():
    """回放两次必须逐字节相同——否则金标准比对无从谈起。"""
    img = frame(cars=[(100, 100, 300, 300)])

    def run():
        b = StubBackend(input_size=INPUT)
        b.load()
        return CarDetector(b).detect(img)

    a1, a2 = run(), run()
    assert [(r.class_id, r.confidence, r.bbox) for r in a1] == \
           [(r.class_id, r.confidence, r.bbox) for r in a2]


def test_swapping_the_backend_changes_nothing_for_the_caller():
    """后端可换是这一层的存在理由：调用方只认 ``DetectorBackend`` 协议。"""
    det = CarDetector(StubBackend(input_size=INPUT))
    det.load()
    assert det.name == "car:stub"
    assert det.input_size == INPUT
    assert det.is_ready
