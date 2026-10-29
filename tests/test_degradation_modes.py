"""M3 验收门：检测层的降级路径与可复现性。

方案 §九.8 要求"空检测 → 槽位按序降级；垃圾检测被拒；后端可换"。状态机
本身是 M4 的事，这一层能保证的是**上游**：检测层在模型缺失、推理报错、
输出畸形、画面上什么都没有这四种情况下，都必须给出一份**形状正确的空
结果**，而不是抛异常或给出一个假的坐标。

这份保证是整条流水线"连续 3 秒不丢"的前提：视觉掉线的表现必须是
"这一帧没有观测"（跟踪层据此走 occlusion_hold），而不是"进程崩了"。

另一半是**可复现**：同样的输入回放两次必须逐字节相同。没有这条，
金标准比对、回归定位、赛后复盘全都无从谈起。
"""

from __future__ import annotations

import numpy as np
import pytest

from jg_radar.detect import (
    ArmorDetector,
    CarDetector,
    ClassMap,
    DetectorSettings,
    DigitClassifier,
)
from jg_radar.detect.backends import BlobBackend, EmptyBackend, OnnxBackend, StubBackend
from jg_radar.detect.protocol import DetectionRecord

INPUT = 640


def frame(w=1280, h=720, *, cars=(), armour=()) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for x1, y1, x2, y2 in cars:
        img[y1:y2, x1:x2] = (255, 255, 255)
    for x1, y1, x2, y2 in armour:
        img[y1:y2, x1:x2] = 200
    return img


CARS = [(200, 150, 400, 350), (800, 300, 1000, 500)]
ARMOUR = [(240, 200, 300, 320), (860, 340, 920, 460)]
SCENE = frame(cars=CARS, armour=ARMOUR)


def blob(**kw) -> BlobBackend:
    kw.setdefault("target", (255, 255, 255))
    kw.setdefault("tol", 20)
    kw.setdefault("min_area", 8)
    kw.setdefault("input_size", INPUT)
    b = BlobBackend(**kw)
    b.load()
    return b


# ======================================================================
# 后端可换：调用方代码不变
# ======================================================================
@pytest.mark.parametrize("factory", [
    lambda: blob(),
    lambda: StubBackend(input_size=INPUT),
    lambda: EmptyBackend(input_size=INPUT),
], ids=["blob", "stub", "empty"])
def test_any_backend_satisfies_the_contract(factory):
    """换后端，调用方一行都不用改——这正是这一层的存在理由。"""
    backend = factory()
    backend.load()
    det = CarDetector(backend, settings=DetectorSettings(min_box_px=2.0))
    cars, rois = det.detect_rois(SCENE)

    assert isinstance(cars, list)
    assert rois.shape == (len(cars), 4)
    for r in cars:
        assert len(r.bbox) == 4
        assert 0.0 <= r.confidence <= 1.0
        x1, y1, x2, y2 = r.bbox
        assert x2 >= x1 and y2 >= y1
        assert 0.0 <= x1 and x2 <= 1280.0 and 0.0 <= y1 and y2 <= 720.0


def test_backend_swap_keeps_the_armor_stage_working():
    """一级换了后端，二级也必须照常工作（它们只通过 ROI 列表耦合）。"""
    det = CarDetector(blob(), settings=DetectorSettings(min_box_px=2.0))
    _, rois = det.detect_rois(SCENE)
    assert len(rois) == 2

    armor = ArmorDetector(
        BlobBackend(target=(200, 200, 200), tol=20, min_area=4, input_size=192),
        settings=DetectorSettings(min_box_px=2.0),
    )
    armor.load()
    hits = armor.detect(SCENE, rois)
    assert {h.car_index for h in hits} == {0, 1}


# ======================================================================
# 模型缺失 / 推理失败
# ======================================================================
def test_missing_model_reports_why_and_stays_silent():
    """没装 onnxruntime 或没有权重时：``is_ready`` 为假、检测为空、
    不抛异常，且**能说清缺的是什么**。

    "检测器什么都不输出"是现场最难查的故障之一，原因是缺库还是权重路径
    写错，直接决定了要不要连夜重装环境。
    """
    onnx = OnnxBackend("config/definitely_not_here.onnx", input_size=INPUT)
    onnx.load()
    assert not onnx.is_ready
    assert onnx.reason, "必须给出没就绪的原因"

    det = CarDetector(onnx)
    assert not det.is_ready
    assert det.detect(SCENE) == []
    assert det.detect_rois(SCENE)[1].shape == (0, 4)


class _ExplodingBackend:
    """每次推理都炸。模拟模型文件损坏、显存不足、CUDA context 丢失。"""

    name = "exploding"
    input_size = INPUT

    def __init__(self):
        self._loaded = False

    @property
    def is_ready(self):
        return self._loaded

    def load(self):
        self._loaded = True

    def warmup(self, n=20):
        pass

    def infer(self, images):
        raise RuntimeError("CUDA error: out of memory")


def test_inference_failure_degrades_to_no_detections():
    """推理炸了要退化成"这一帧没有观测"，而不是把异常抛进流水线。

    一次显存抖动不该让雷达站停摆——跟踪层本来就有处理"没有观测"的能力
    （走 occlusion_hold），但那只有在异常没被抛出去的前提下才轮得到它。
    """
    det = CarDetector(_ExplodingBackend())
    assert det.detect(SCENE) == []


class _WrongLengthBackend:
    """返回的批长度和输入对不上。写错批处理代码时很常见。"""

    name = "wronglen"
    input_size = INPUT

    def __init__(self, n=0):
        self.n = n
        self._loaded = False

    @property
    def is_ready(self):
        return self._loaded

    def load(self):
        self._loaded = True

    def warmup(self, n=20):
        pass

    def infer(self, images):
        return [[] for _ in range(self.n)]


@pytest.mark.parametrize("n", [0, 1, 5])
def test_wrong_batch_length_is_treated_as_no_detections(n):
    """批长度对不上时按"这一帧没有观测"处理。

    1 个输入却回了 0 个或 5 个结果，说明批处理代码写错了；此时把结果硬
    zip 上去会让检测框配到错误的图像上——那比没有任何检测危险得多。
    """
    det = CarDetector(_WrongLengthBackend(n))
    det.load()
    assert det.detect(SCENE) == []


def test_backend_that_is_never_loaded_returns_nothing():
    """忘了 load() 时也该是空结果，而不是 RuntimeError 掀翻整条流水线。"""
    backend = BlobBackend(target=(255, 255, 255), input_size=INPUT)  # 故意不 load()
    det = CarDetector(backend)
    assert not det.is_ready
    assert det.detect(SCENE) == []


# ======================================================================
# 垃圾检测被拒
# ======================================================================
class _FixedBackend:
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


def _accept(records, **kw):
    kw.setdefault("settings", DetectorSettings(conf_thresh=0.25, min_box_px=4.0))
    det = CarDetector(_FixedBackend(records), **kw)
    det.load()
    return det.detect(SCENE)


GOOD = (320.0, 200.0, 420.0, 300.0)  # 稳落在有效画面里


@pytest.mark.parametrize("bad_bbox", [
    (float("nan"), 200.0, 420.0, 300.0),
    (320.0, float("nan"), 420.0, 300.0),
    (320.0, 200.0, float("nan"), 300.0),
    (320.0, 200.0, 420.0, float("nan")),
    (float("inf"), 200.0, 420.0, 300.0),
    (320.0, 200.0, float("-inf"), 300.0),
    (0.0, 0.0, 0.0, 0.0),                 # 零面积
    (400.0, 300.0, 320.0, 200.0),          # 反了的框
    (320.0, 200.0, 321.0, 201.0),          # 太小
])
def test_garbage_boxes_are_rejected(bad_bbox):
    assert _accept([DetectionRecord(0, 0.9, bad_bbox)]) == []


def test_nan_never_reaches_the_caller():
    """NaN 一旦进入跟踪层，协方差会被污染且无法恢复——必须在入口挡住。"""
    recs = [DetectionRecord(0, 0.9, b) for b in [
        (float("nan"),) * 4,
        (320.0, 200.0, float("inf"), float("inf")),
    ]]
    assert _accept(recs) == []


def test_a_good_box_survives_the_same_path():
    """上面那一堆"被拒"必须有对照，否则可能是整条路都废了而测试看不出来。"""
    got = _accept([DetectionRecord(0, 0.9, GOOD)])
    assert len(got) == 1


def test_extreme_confidence_values_are_handled():
    assert _accept([DetectionRecord(0, -1.0, GOOD)]) == []
    assert len(_accept([DetectionRecord(0, 1.0, GOOD)])) == 1
    assert len(_accept([DetectionRecord(0, 1e9, GOOD)])) == 1


def test_negative_class_id_is_not_silently_accepted_into_a_map():
    """类别表之外的 id 一律拒绝，负号也不例外。"""
    det = CarDetector(
        _FixedBackend([DetectionRecord(-1, 0.9, GOOD)]),
        class_map=ClassMap({0: "car", 1: "car"}),
        settings=DetectorSettings(conf_thresh=0.25, min_box_px=4.0),
    )
    det.load()
    assert det.detect(SCENE) == []


# ======================================================================
# 空画面
# ======================================================================
def test_blank_scene_yields_nothing_at_every_stage():
    blank = frame()
    det = CarDetector(blob(), settings=DetectorSettings(min_box_px=2.0))
    cars, rois = det.detect_rois(blank)
    assert cars == [] and rois.shape == (0, 4)

    armor = ArmorDetector(
        BlobBackend(target=(200, 200, 200), tol=20, min_area=4, input_size=192)
    )
    armor.load()
    assert armor.detect(blank, rois) == []

    clf = DigitClassifier(
        BlobBackend(target=(200, 200, 200), tol=20, min_area=4, input_size=192)
    )
    clf.load()
    assert clf.classify(blank, np.zeros((0, 4))) == []


@pytest.mark.parametrize("shape", [(0, 0, 3), (0, 100, 3), (100, 0, 3)])
def test_empty_frames_do_not_crash(shape):
    det = CarDetector(blob(), settings=DetectorSettings(min_box_px=2.0))
    assert det.detect(np.zeros(shape, dtype=np.uint8)) == []


def test_empty_backend_reproduces_the_full_degradation_chain():
    """``EmptyBackend`` 是降级链的驱动器：它让"画面里没有车"这件事可以被
    稳定复现，M4 的 ``MEASURED→HOLD→BLIND→MISSING`` 就靠它来测。

    这一层能保证的是它**确实**什么都不给，而不是偶尔漏一个。
    """
    det = CarDetector(EmptyBackend(input_size=INPUT))
    det.load()
    assert det.is_ready
    for _ in range(5):
        assert det.detect(SCENE) == []


# ======================================================================
# 回放可复现
# ======================================================================
def _run_cascade(backend_factory, frames):
    det = CarDetector(backend_factory(), settings=DetectorSettings(min_box_px=2.0))
    det.load()
    out = []
    for f in frames:
        cars, rois = det.detect_rois(f)
        out.append([(c.class_id, round(c.confidence, 12), c.bbox) for c in cars])
    return out


def test_replay_is_byte_identical():
    """同样的输入回放两次，结果必须逐字节相同。

    没有这条，金标准比对、回归定位、赛后复盘全都不成立——而"偶尔不一样"
    的 bug 恰恰是最难查的那种。
    """
    frames = [SCENE, frame(), frame(cars=[(100, 100, 260, 260)]), SCENE]
    a = _run_cascade(blob, frames)
    b = _run_cascade(blob, frames)
    assert a == b


def test_replay_is_identical_across_backend_instances():
    """换一个**实例**（不是换配置）也应当完全一致——种子必须来自内容，
    不能来自对象生命周期或全局随机状态。"""
    frames = [SCENE, frame(cars=[(50, 50, 200, 200)])]
    a = _run_cascade(blob, frames)

    def fresh():
        b = BlobBackend(target=(255, 255, 255), tol=20, min_area=8, input_size=INPUT)
        b.load()
        return b

    assert a == _run_cascade(fresh, frames)


def test_stub_backend_replay_is_reproducible():
    frames = [SCENE, frame(cars=[(100, 100, 260, 260)])]
    assert _run_cascade(lambda: StubBackend(input_size=INPUT, seed=7), frames) == \
           _run_cascade(lambda: StubBackend(input_size=INPUT, seed=7), frames)


def test_stub_backend_seed_actually_changes_the_output():
    """上面那条"可复现"必须有对照：如果种子根本没起作用，它也会通过。"""
    frames = [SCENE]
    a = _run_cascade(lambda: StubBackend(input_size=INPUT, seed=1), frames)
    b = _run_cascade(lambda: StubBackend(input_size=INPUT, seed=2), frames)
    assert a != b


def test_different_frames_produce_different_stub_detections():
    """确定性来自**图像内容**哈希，而不是"第几次调用"——否则回放时
    跳帧、重放、并行都会让结果漂移。"""
    a = _run_cascade(lambda: StubBackend(input_size=INPUT), [SCENE])
    b = _run_cascade(lambda: StubBackend(input_size=INPUT),
                     [frame(cars=[(700, 100, 900, 400)])])
    assert a != b
