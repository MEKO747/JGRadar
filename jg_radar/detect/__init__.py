"""检测层：可插拔推理后端 + 两级级联。

级联的数据流（方案 §五）：

    整帧 ──CarDetector（1280，letterbox）──▶ 整车框
                                              │
         ┌────────────────────────────────────┘
         ▼
    RoiBatcher 裁切拼图（≤16 块 / 画布）
         ▼
    ArmorDetector 一次批量推理 ──▶ 装甲板框（带整车下标）
         ▼
    DigitClassifier 批量推理 ──▶ 数字（身份投票的输入）
"""

from jg_radar.detect.armor_detector import ArmorDetector, ArmorHit
from jg_radar.detect.base import BaseDetector, DetectorSettings
from jg_radar.detect.car_detector import CarDetector
from jg_radar.detect.digit_classifier import UNKNOWN_CLASS, DigitClassifier, DigitVote
from jg_radar.detect.factory import Cascade, CascadeResult, build_backend, build_cascade
from jg_radar.detect.labels import ClassMap
from jg_radar.detect.nms import iou_matrix, nms, nms_records
from jg_radar.detect.protocol import DetectionRecord, DetectorBackend
from jg_radar.detect.roi_batch import (
    LetterboxTransform,
    RoiBatch,
    RoiBatcher,
    RoiTile,
    letterbox,
)

__all__ = [
    "UNKNOWN_CLASS",
    "ArmorDetector",
    "ArmorHit",
    "BaseDetector",
    "CarDetector",
    "Cascade",
    "CascadeResult",
    "build_backend",
    "build_cascade",
    "ClassMap",
    "DetectionRecord",
    "DetectorBackend",
    "DetectorSettings",
    "DigitClassifier",
    "DigitVote",
    "LetterboxTransform",
    "RoiBatch",
    "RoiBatcher",
    "RoiTile",
    "iou_matrix",
    "letterbox",
    "nms",
    "nms_records",
]
