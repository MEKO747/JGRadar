"""从配置装配整条检测级联。

这一层的存在意义就一句话：**换一份权重、换一个推理后端，只改 YAML**。
类别表、输入尺寸、置信度门限、批大小全在 ``detect`` 组里，代码里没有
任何一处硬编码的类别定义。

``trt`` / ``torch`` 目前**直接报错**而不是退回 stub。一个雷达站如果因为
权重路径写错就静默退化成"什么都检测不到"，它会一路跑下去、按时上报
"没看到敌人"——这比启动失败危险得多。宁可拒绝启动。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jg_radar.config.schema import ConfigError, DetectConfig
from jg_radar.detect.armor_detector import ArmorDetector, ArmorHit
from jg_radar.detect.base import DetectorSettings
from jg_radar.detect.car_detector import CarDetector
from jg_radar.detect.digit_classifier import DigitClassifier, DigitVote
from jg_radar.detect.labels import ClassMap
from jg_radar.detect.protocol import DetectionRecord, DetectorBackend
from jg_radar.detect.roi_batch import RoiBatcher

__all__ = ["Cascade", "CascadeResult", "build_backend", "build_cascade"]


def build_backend(
    name: str,
    *,
    model_path: str = "",
    input_size: int,
    num_classes: int = 0,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_dets: int = 32,
) -> DetectorBackend:
    """按名字构造一个后端。"""
    key = (name or "").strip().lower()

    if key == "mobilenet":
        from jg_radar.detect.backends.mobilenet_backend import MobileNetBackend
        return MobileNetBackend(model_path, input_size=input_size, num_classes=num_classes)

    if key == "stub":
        from jg_radar.detect.backends import StubBackend

        # 类别数取类别表的大小；表为空时给一个够用的上界
        return StubBackend(
            num_classes=max(num_classes, 1), input_size=input_size,
        )
    if key == "empty":
        from jg_radar.detect.backends import EmptyBackend

        return EmptyBackend(input_size=input_size, num_classes=max(num_classes, 1))
    if key == "onnx":
        from jg_radar.detect.backends import OnnxBackend

        return OnnxBackend(
            model_path or None,
            input_size=input_size,
            num_classes=max(num_classes, 1),
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            max_dets=max_dets,
        )
    if key in ("trt", "tensorrt", "torch"):
        raise ConfigError(
            f"detect.backend={name!r} 尚未实现。可用的有 stub / empty / onnx。"
            "这里刻意不退回 stub —— 静默降级会让雷达站按时上报“没看到敌人”，"
            "比启动失败危险得多。"
        )
    raise ConfigError(f"未知的 detect.backend={name!r}；可用：stub / empty / onnx")


@dataclass
class CascadeResult:
    """一帧走完三级级联的全部产物。"""

    cars: list[DetectionRecord]
    rois: np.ndarray
    armors: list[ArmorHit]
    digits: list[DigitVote]

    @property
    def empty(self) -> bool:
        return not self.cars

    def armor_for(self, car_index: int) -> list[ArmorHit]:
        return [h for h in self.armors if h.car_index == car_index]


class Cascade:
    """整车 → 装甲板 → 数字。事件循环调用的入口。"""

    def __init__(
        self, car: CarDetector, armor: ArmorDetector, digit: DigitClassifier
    ) -> None:
        self.car = car
        self.armor = armor
        self.digit = digit

    @property
    def is_ready(self) -> bool:
        return self.car.is_ready

    def load(self) -> None:
        self.car.load()
        self.armor.load()
        self.digit.load()

    def warmup(self, n: int = 20) -> None:
        self.car.warmup(n)
        self.armor.warmup(n)
        self.digit.warmup(n)

    def run(self, frame: np.ndarray) -> CascadeResult:
        """跑一帧。任何一级出问题都只影响它自己的输出，不影响其它级。

        一级空 → 二级没有 ROI 可裁、三级没有框可读，自然也都是空的。
        这是**设计如此**：整车没框出来，它的装甲板就不该凭空出现。
        """
        cars, rois = self.car.detect_rois(frame)
        if not cars:
            return CascadeResult(cars=[], rois=rois, armors=[], digits=[])
        armors = self.armor.detect(frame, rois)
        digits = (
            self.digit.classify(frame, [h.bbox for h in armors]) if armors else []
        )
        return CascadeResult(cars=cars, rois=rois, armors=armors, digits=digits)


def build_cascade(cfg: DetectConfig) -> Cascade:
    """按 ``detect`` 配置装配级联。

    **类别表就是过滤器**：表里没有的类别整级丢弃。留空表示这一级不过滤
    （适配期用，正式配置里应当写明）。
    """
    car_classes = ClassMap.from_spec(cfg.car_classes or None)
    armor_classes = ClassMap.from_spec(cfg.armor_classes or None)
    digit_classes = ClassMap.from_spec(cfg.digit_classes or None)

    car_backend = build_backend(
        cfg.backend, model_path=cfg.car_model, input_size=cfg.car_img_size,
        num_classes=len(car_classes), conf_thresh=cfg.car_conf,
        iou_thresh=cfg.iou, max_dets=cfg.max_det,
    )
    armor_backend = build_backend(
        cfg.backend, model_path=cfg.armor_model, input_size=cfg.armor_img_size,
        num_classes=len(armor_classes), conf_thresh=cfg.armor_conf,
        iou_thresh=cfg.iou, max_dets=cfg.max_det,
    )
    digit_backend = build_backend(
        cfg.digit_backend or cfg.backend, model_path=cfg.digit_model, input_size=cfg.digit_img_size,
        num_classes=len(digit_classes), conf_thresh=cfg.digit_conf,
        iou_thresh=cfg.iou, max_dets=cfg.max_det,
    )

    car = CarDetector(
        car_backend,
        class_map=car_classes if cfg.car_classes else None,
        settings=DetectorSettings(
            conf_thresh=cfg.car_conf, iou_thresh=cfg.iou,
            max_dets=cfg.max_det, min_box_px=cfg.min_box_px,
        ),
    )
    armor = ArmorDetector(
        armor_backend,
        class_map=armor_classes if cfg.armor_classes else None,
        settings=DetectorSettings(
            conf_thresh=cfg.armor_conf, iou_thresh=cfg.iou,
            max_dets=cfg.max_det, min_box_px=cfg.min_box_px,
        ),
        batcher=RoiBatcher(tile_size=cfg.armor_img_size, grid=_grid_for(cfg.roi_batch)),
    )
    digit = DigitClassifier(
        digit_backend,
        class_map=digit_classes if cfg.digit_classes else None,
        min_confidence=cfg.digit_conf,
        batcher=RoiBatcher(tile_size=cfg.digit_img_size, grid=_grid_for(cfg.roi_batch)),
    )
    return Cascade(car, armor, digit)


def _grid_for(capacity: int) -> int:
    """把"一张画布装多少块"折算成方阵边长，向上取整。"""
    n = max(1, int(capacity))
    g = 1
    while g * g < n:
        g += 1
    return g
