"""融合层的接线契约：**源**与**最新值缓冲**。

无线电不归本项目做，所以这一层的全部意义就是让"将来插进来的那个源"
不需要动其他任何代码。约束只有三条：

1. 源只需要能产出 ``SourceSample``（形状在 ``jg_radar.types`` 里定死了）。
2. 源把样本写进 ``LatestValueBuffer``，**按 ``(来源, 机器人名)`` 索引，每个格子
   只保留最新的一条**。
3. 仲裁在**发布时刻**做，不在写入时刻做。

第 3 条是关键的：慢一帧的视觉不该拖慢发送。生产者各写各的，互不等待；
发布线程该发就发，只是发的时候顺手看一眼"现在谁的数据最新最可信"。

第 2 条里的 ``(来源, 机器人名)`` 是**必须**的两元键，不能只用机器人名。
视觉和无线电会同时报 ``B1``；只按名字索引的话，后写的那条会把先写的**覆盖掉**，
缓冲里永远只剩一个源的数据——于是仲裁层压根看不见第二个候选，"优先级"和
"滞回"全部退化成"谁写得靠后就听谁的"。这个 bug 不会报错、不会崩，
只会让整套仲裁静默失效。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from jg_radar.types import SourceKind, SourceSample

__all__ = ["LatestValueBuffer", "Source", "StampedSample"]


@dataclass(frozen=True)
class StampedSample:
    """一条样本 + **它被写进缓冲的时刻**。

    为什么不能只用 ``SourceSample.observed_at``：视觉的预测态（遮挡保持/盲区）
    刻意让 ``observed_at`` 停在最后一次真观测上，好让下游一眼看出信息有多旧。
    于是它一定会超过视觉看门狗的时间——**如果看门狗拿 observed_at 去量，
    所有预测态都会被判成"视觉死了"**，盲区预测就永远发不出去。

    看门狗要回答的是另一个问题："这个源还活着吗。"那是 ``received_at``。
    两个时间戳量的是两件事，都必须留着。
    """

    sample: SourceSample
    received_at: float

    @property
    def name(self) -> str:
        return self.sample.name

    def information_age(self, now: float) -> float:
        """信息自身的年龄：从**产生**那一刻算起。

        无线电解码要时间，视觉要从采集到推理走完，所以它天然大于 0——
        这正是"这条坐标有多旧"的正确答案。
        """
        return max(0.0, float(now) - float(self.sample.observed_at))

    def liveness_age(self, now: float) -> float:
        """源的存活年龄：从**写进缓冲**那一刻算起。"""
        return max(0.0, float(now) - float(self.received_at))


@runtime_checkable
class Source(Protocol):
    """一个坐标来源。视觉与无线电实现同一个协议。"""

    @property
    def kind(self) -> SourceKind:
        ...

    @property
    def name(self) -> str:
        ...

    def poll(self, now: float) -> Sequence[SourceSample]:
        """取本周期要写入的样本。允许返回空。"""
        ...


_Key = tuple[SourceKind, str]


class LatestValueBuffer:
    """按 ``(来源, 机器人名)`` 索引的最新值缓冲，线程安全。

    键里必须带来源：视觉和无线电会同时报 ``B1``，只按名字索引的话后写的
    会把先写的覆盖掉，仲裁层就再也看不见两个候选了。详见模块文档。

    **每个格子只保留最新的一条**，不做队列。理由：这是实时上报，堆积的历史
    样本没有任何用处——发布时永远只关心"现在最新的那条是什么"。队列反而会
    引入"追着补发旧数据"这种最危险的失效模式（旧坐标看起来像新坐标）。

    没有新样本时**不清空**：视觉在盲区里仍然每帧写，无线电则可能几百毫秒
    才来一次；把没更新的槽位清掉，等于把"上一次听到的"当成"从没听到过"。
    过期的判断交给仲裁层的新鲜度规则，那里有明确的时间阈值。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[_Key, StampedSample] = {}

    @staticmethod
    def _key(sample: SourceSample) -> _Key:
        return (sample.source, sample.name)

    def put(self, sample: SourceSample, *, now: float) -> None:
        with self._lock:
            self._items[self._key(sample)] = StampedSample(
                sample=sample, received_at=float(now)
            )

    def put_many(self, samples: Sequence[SourceSample], *, now: float) -> None:
        with self._lock:
            for s in samples:
                self._items[self._key(s)] = StampedSample(
                    sample=s, received_at=float(now)
                )

    def snapshot(self) -> tuple[StampedSample, ...]:
        """当前全部样本的副本。

        返回的是一个**序列**而不是按名索引的字典：同一个机器人本来就会有
        多条（每个源一条），用字典表达会诱导调用方误以为一个名字只有一个值。
        返回副本是为了让调用方拿到一致的视图，不会在遍历途中被生产者改掉。
        """
        with self._lock:
            return tuple(self._items.values())

    def get(self, source: SourceKind, name: str) -> StampedSample | None:
        with self._lock:
            return self._items.get((source, name))

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
