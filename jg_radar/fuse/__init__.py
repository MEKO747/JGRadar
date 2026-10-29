"""融合层：多源仲裁与发布。

数据流（单帧）::

    VisionSource.submit(跟踪器输出)        ─┐
    RadioSource.feed(解码结果)             ─┼─→ LatestValueBuffer ─→ Arbiter.decide
                                           ─┘                          │
                                                                 PublishLayer.build
                                                                        │
                                                                    FrameOutput

仲裁发生在**发布时刻**，不在写入时刻——慢一帧的视觉不该拖慢发送。
"""

from jg_radar.fuse.arbiter import (
    KNOWN_TIERS,
    PREDICTION_SIGMA,
    Arbiter,
    SlotDecision,
    classify,
    position_error_bound_cm,
    radio_tier,
    vision_tier,
)
from jg_radar.fuse.contract import LatestValueBuffer, Source, StampedSample
from jg_radar.fuse.publish import PublishLayer
from jg_radar.fuse.sources import RadioReading, RadioSource, VisionSource

__all__ = [
    "KNOWN_TIERS",
    "PREDICTION_SIGMA",
    "Arbiter",
    "LatestValueBuffer",
    "PublishLayer",
    "RadioReading",
    "RadioSource",
    "SlotDecision",
    "Source",
    "StampedSample",
    "VisionSource",
    "classify",
    "position_error_bound_cm",
    "radio_tier",
    "vision_tier",
]
