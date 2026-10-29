"""事件循环：把检测 → 跟踪 → 仲裁 → 发布串成一条能一直跑的流水线。

分两级（EGA 的设计）：

  Stage A（感知）   帧 → 检测级联 → 观测（**只做像素与世界坐标的转换**）
  Stage B（融合）   观测 → 跟踪 → 仲裁 → 发布

两级之间隔一个 :class:`~jg_radar.pipeline.mailbox.Mailbox`。分开的理由不是
"并行更快"，而是**两者对延迟的容忍度完全不同**：Stage A 慢一点只是这一帧
晚出来，Stage B 慢一点则会让上报的坐标整体落后于现实。隔开之后，Stage B
可以理直气壮地丢掉积压的旧帧（见 ``mailbox`` 的文档），而这在一条
端到端的循环里是做不到的。

两种跑法，**默认是单线程的那一种**：

``EventLoop``          确定性，回放与测试用。同一份输入两次跑出逐字节相同的结果。
``ThreadedEventLoop``  实时部署用。Stage A 在后台线程，Stage B 按节奏发布。

装配入口是 :func:`build_loop`——换相机、换模型、换权重都只改 YAML。
"""

from jg_radar.pipeline.build import (
    build_loop,
    build_lookup,
    build_source,
    build_stage_a,
    build_stage_b,
)
from jg_radar.pipeline.event_loop import (
    EventLoop,
    ThreadedEventLoop,
    default_clock,
    frame_clock,
    wall_clock,
)
from jg_radar.pipeline.mailbox import Mailbox, StageAPacket
from jg_radar.pipeline.stage_a import StageA
from jg_radar.pipeline.stage_b import MAX_PLAUSIBLE_AGE_S, ClockMismatch, StageB
from jg_radar.pipeline.timings import Timings

__all__ = [
    "ClockMismatch",
    "EventLoop",
    "MAX_PLAUSIBLE_AGE_S",
    "Mailbox",
    "StageA",
    "StageAPacket",
    "StageB",
    "ThreadedEventLoop",
    "Timings",
    "build_loop",
    "build_lookup",
    "build_source",
    "build_stage_a",
    "build_stage_b",
    "default_clock",
    "frame_clock",
    "wall_clock",
]
