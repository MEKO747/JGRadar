"""检测层与配置的接线：换权重只改 YAML，不动代码。

这是方案 §五 对"用开源权重起步"这件事的正面回应。三家的类别定义完全不同，
把任何一家写死进代码，都意味着换权重要改代码、要重新 review 检测层——而
换权重本来应该只是改一行配置。

这里测的就是那条界线：**类别语义只能来自配置**，以及配置写错时必须
拒绝启动，而不是悄悄退化成"什么都检测不到"。
"""

from __future__ import annotations

import numpy as np
import pytest

from jg_radar.config import load_config
from jg_radar.config.schema import ConfigError, DetectConfig
from jg_radar.detect.factory import build_backend, build_cascade
from jg_radar.detect.labels import ClassMap


# ======================================================================
# 类别表：翻译 + 过滤
# ======================================================================
def test_class_map_from_a_yaml_style_dict_with_int_keys():
    m = ClassMap.from_spec({0: "B1", 5: "R1"})
    assert m.label(0) == "B1" and m.label(5) == "R1"
    assert m.has(0) and not m.has(1)


def test_class_map_accepts_string_keys():
    """YAML 把 ``0:`` 解析成 int，但 ``"0":`` 或从 JSON 来的是字符串。
    两种都必须能用——否则同一份表换个来源就静默失效。"""
    m = ClassMap.from_spec({"0": "B1", "5": "R1"})
    assert m.label(0) == "B1" and m.has(5)


def test_class_map_from_a_list_uses_the_index():
    m = ClassMap.from_spec(["1", "2", "3"])
    assert [m.label(i) for i in range(3)] == ["1", "2", "3"]


def test_none_spec_means_no_mapping():
    """``None`` 是"不过滤"，``{}`` 是"过滤掉一切"——两者不能混。"""
    assert len(ClassMap.from_spec(None, num_classes=4)) == 4
    assert len(ClassMap.from_spec({})) == 0


def test_ids_and_subset():
    m = ClassMap.from_spec({0: "B1", 1: "B2", 2: "R1"})
    assert m.ids("R1") == [2]
    assert m.ids("B1", "R1") == [0, 2]
    assert m.ids("nope") == []          # 找不到的标签跳过，不报错
    assert set(m.subset("B1", "B2").names) == {0, 1}


def test_describe_lists_every_class():
    text = ClassMap.from_spec({0: "B1", 1: "R1"}).describe()
    assert "0→B1" in text and "1→R1" in text


# ======================================================================
# 从 default.yaml 构造
# ======================================================================
def test_default_config_builds_a_working_cascade():
    cfg = load_config("config/default.yaml")
    cascade = build_cascade(cfg.detect)
    assert cascade.car.name == "car:stub"
    assert cascade.armor.capacity == 16   # 与方案 §五 的 4×4 一致
    assert cascade.digit.name == "digit:stub"


def test_default_config_car_class_table_is_wired_through():
    """配置里的车类别名必须真的进到检测器里，而不是只躺在配置对象里。"""
    cfg = load_config("config/default.yaml")
    cascade = build_cascade(cfg.detect)
    labels = set(cascade.car.class_map.names.values())
    assert {"B1", "R5"} <= labels
    assert len(cascade.car.class_map) == 10


def test_empty_class_table_means_no_filtering():
    """``{}`` 是"这一级不过滤"，而不是"什么都不要"。"""
    cfg = load_config("config/default.yaml")
    cascade = build_cascade(cfg.detect)
    assert cascade.armor.class_map is None


def test_roi_batch_config_drives_the_grid():
    cfg = DetectConfig(backend="stub", roi_batch=9)
    assert build_cascade(cfg).armor.capacity == 9
    cfg = DetectConfig(backend="stub", roi_batch=17)
    assert build_cascade(cfg).armor.capacity == 25   # 向上取整到方阵
    cfg = DetectConfig(backend="stub", roi_batch=1)
    assert build_cascade(cfg).armor.capacity == 1


def test_confidence_thresholds_flow_into_the_detectors():
    cfg = DetectConfig(backend="stub", car_conf=0.77, armor_conf=0.66, iou=0.33)
    cascade = build_cascade(cfg)
    assert cascade.car.settings.conf_thresh == pytest.approx(0.77)
    assert cascade.armor.settings.conf_thresh == pytest.approx(0.66)
    assert cascade.car.settings.iou_thresh == pytest.approx(0.33)


def test_img_sizes_flow_into_backends_and_tiles():
    cfg = DetectConfig(backend="stub", car_img_size=960, armor_img_size=256,
                       digit_img_size=64)
    cascade = build_cascade(cfg)
    assert cascade.car.input_size == 960
    assert cascade.armor.batcher.tile_size == 256
    assert cascade.digit.batcher.tile_size == 64


# ======================================================================
# 配错时必须拒绝启动
# ======================================================================
def test_unimplemented_backend_refuses_to_start():
    """``trt`` 还没写，配了要**报错**，绝不能悄悄退回 stub。

    静默降级的后果不是"慢一点"，而是雷达站一路跑下去、按时上报"没看到
    敌人"。那比启动失败危险得多。
    """
    for name in ("trt", "tensorrt", "torch"):
        with pytest.raises(ConfigError, match="尚未实现"):
            build_backend(name, input_size=640)


def test_unknown_backend_is_rejected():
    with pytest.raises(ConfigError, match="未知的 detect.backend"):
        build_backend("yolov99", input_size=640)


def test_backend_names_are_case_and_space_insensitive():
    """YAML 里手写的 ``ONNX `` 不该因为大小写而报"未知后端"。"""
    for name in ("STUB", " stub ", "Empty"):
        assert build_backend(name, input_size=64) is not None


def test_onnx_backend_from_config_is_not_ready_without_a_model():
    """没有权重时后端不就绪，但**构造本身不能炸** —— 缺权重是常态，
    要在 ``is_ready`` 上体现，而不是在 import 或构造阶段抛异常。"""
    cfg = DetectConfig(backend="onnx", car_model="", car_img_size=640)
    backend = build_backend("onnx", model_path="", input_size=640)
    assert not backend.is_ready
    cascade = build_cascade(cfg)
    assert not cascade.is_ready
    assert cascade.run(np.zeros((480, 640, 3), dtype=np.uint8)).empty


def test_config_rejects_unknown_detect_keys():
    """写错键名（比如用了旧的 ``class_mapping``）要报错，而不是被忽略。"""
    from jg_radar.config.schema import RadarConfig

    with pytest.raises(ConfigError, match="未知配置项"):
        RadarConfig.from_dict({"detect": {"class_mapping": {}}})


# ======================================================================
# Cascade.run 的短路行为
# ======================================================================
def test_cascade_short_circuits_when_no_car_is_found():
    """整车没框出来，装甲板就不该凭空出现——空的结果必须一路空到底。"""
    cfg = DetectConfig(backend="empty", car_img_size=320, armor_img_size=64,
                       digit_img_size=32)
    cascade = build_cascade(cfg)
    cascade.load()
    result = cascade.run(np.zeros((240, 320, 3), dtype=np.uint8))
    assert result.empty
    assert result.armors == [] and result.digits == []
    assert result.rois.shape == (0, 4)


def test_armor_for_filters_by_car_index():
    from jg_radar.detect.armor_detector import ArmorHit
    from jg_radar.detect.factory import CascadeResult
    from jg_radar.detect.protocol import DetectionRecord

    res = CascadeResult(
        cars=[], rois=np.zeros((0, 4)),
        armors=[
            ArmorHit(0, DetectionRecord(1, 0.9, (0.0, 0.0, 10.0, 10.0))),
            ArmorHit(1, DetectionRecord(1, 0.9, (0.0, 0.0, 10.0, 10.0))),
            ArmorHit(0, DetectionRecord(2, 0.9, (0.0, 0.0, 10.0, 10.0))),
        ],
        digits=[],
    )
    assert len(res.armor_for(0)) == 2
    assert len(res.armor_for(1)) == 1
    assert res.armor_for(9) == []
