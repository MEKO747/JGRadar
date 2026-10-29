"""无线电接入点——**接口桩，不含解码**。

信息波解析不归本项目做。这个文件的全部意义是：把"将来插进来"这件事
压缩成**一个类的实现**，而且不需要动视觉侧任何一行代码。

## 接入步骤（三步，全在无线电组那边）

1. 解码器在**空口收到一帧**时，调用::

       source.feed([RadioReading(name="R1", position_cm=(1200, 700),
                                 observed_at=monotonic_now)], now=monotonic_now)

2. 把 ``RadioSource()`` 塞进源列表（或按配置注册）。
3. 完。

仲裁层通过 ``fuse.priority`` 里的 ``radio_fresh`` / ``radio_stale`` 两个档位
认识它；这两个档位名**本来就在优先级阶梯里**，视觉侧一行的改动都不需要。
这就是"零改动赢得仲裁"的实现方式。

## 唯一一个必须说清楚的约定：``observed_at`` 是空口接收时刻

**不是**"解码完成时刻"，**更不是** ``now``。

仲裁层判断"新鲜/陈旧"量的是 ``now - observed_at``。如果解码器图省事填 ``now``，
那么无论链路实际上断了多久，每条样本看起来都是 0 秒新鲜——**断流永远不会被
发现**，雷达会抱着一个不动的位置一直报 ``radio_fresh``。这是这个接口唯一一个
"填错了不会报错、只会让整个降级机制失效"的地方。

时钟必须是**本机的单调时钟**，在收到报文的那一刻取。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from jg_radar.types import RobotState, SourceKind, SourceSample

__all__ = ["RadioReading", "RadioSource"]


@dataclass(frozen=True)
class RadioReading:
    """无线电组需要产出的最小形状。"""

    name: str
    position_cm: tuple[int, int]
    observed_at: float
    """**本机单调时钟**上的空口接收时刻。见模块文档。"""
    confidence: float = 1.0
    state: RobotState = RobotState.MEASURED


class RadioSource:
    """一个被动的源桩：没人 ``feed`` 就什么都产不出。"""

    kind = SourceKind.RADIO

    def __init__(self, name: str = "radio", *, max_future_s: float = 0.0) -> None:
        self._name = str(name)
        self._max_future_s = float(max_future_s)
        self._pending: list[SourceSample] = []
        self._revision = 0
        self.accepted = 0
        self.rejected = 0

    @property
    def name(self) -> str:
        return self._name

    def feed(self, readings: Iterable[RadioReading], *, now: float) -> int:
        """写入一批解码结果。返回实际接受的条数。"""
        now = float(now)
        taken = 0
        for r in readings:
            ts = float(r.observed_at)
            if not math.isfinite(ts):
                self.rejected += 1
                continue
            if ts > now + self._max_future_s:
                # 未来时间戳会让 information_age 被 max(0, ·) 钳成 0，
                # 于是这条样本**永远新鲜**——降级机制对它彻底失效。
                # 这种错必须在源头炸掉，不能钳一下放过去。
                raise ValueError(
                    f"无线电 {r.name}: observed_at={ts} 在未来（now={now}）。"
                    "解码器应在本机收到报文的那一刻取单调时钟。"
                )
            self._revision += 1
            self._pending.append(
                SourceSample(
                    source=SourceKind.RADIO,
                    name=str(r.name),
                    position_cm=(int(r.position_cm[0]), int(r.position_cm[1])),
                    state=r.state,
                    confidence=float(r.confidence),
                    observed_at=ts,
                    revision=self._revision,
                    is_predicted=r.state.is_prediction,
                )
            )
            taken += 1
        self.accepted += taken
        return taken

    def poll(self, now: float) -> Sequence[SourceSample]:
        """发布线程调用：取走并清空。"""
        del now
        out = self._pending
        self._pending = []
        return out

    @property
    def pending(self) -> int:
        return len(self._pending)
