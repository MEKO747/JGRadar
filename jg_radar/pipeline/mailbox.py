"""A/B 两级之间的信箱。**只留最新的，旧的宁可丢。**

## 为什么是"信箱"而不是"队列"

队列的语义是"不丢数据"：Stage A 产多快，Stage B 就得追多快，追不上就堆积。
放到雷达站上，这个语义是**错的**——而且是危险的那种错。

设想 Stage B 因为一次 GC 卡了 300ms。这期间 Stage A 往里塞了 6 帧。恢复后
Stage B 从队列头开始消费，于是它接下来报出去的是**6 帧之前**的坐标，
每报一帧就又落后一帧。无线电那一路早就报了新位置了，视觉这一路还在补旧账。
下游看到的是一段"延时 300ms 的历史回放"——而它看起来完全正常，只是位置
总在别人后面。这就是方案 §十三 风险 5 说的"帧龄与提交延迟"。

所以这里的语义是**反过来的**：Stage B 永远只取最新的一帧，中间的全部丢掉，
并且把丢掉的数目记下来。丢帧是可观测的坏消息；补旧数据是不可观测的坏消息。
两害相权，选前者。

## 什么时候可以丢

雷达的考核指标是"连续 3 秒正确"，不是"一帧都不能少"。少一帧只让那 3 秒
晚 50ms 凑满；而报一帧旧坐标可能直接把已经凑上的 2.9 秒打断。
**时效性压倒完整性**，这条贯穿整个流水线。
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from jg_radar.track.types import TrackObservation

__all__ = ["Mailbox", "StageAPacket"]


@dataclass(frozen=True)
class StageAPacket:
    """Stage A 的产物。**同一个时刻冻结**。

    ``captured_at`` 是包里**唯一**的时间戳，包里所有观测都来自那一帧。
    这不是为了省字段，是为了堵死一类 bug：如果每个观测各带各的时间戳，
    下游迟早会拿"最新的那个观测的时间"去代表整包，于是包的年龄被低估，
    超龄判断失效，一个早就该丢的包被发了出去。

    EGA 的 ``StageAPacket`` 就是这个形状，这里照搬。
    """

    frame_seq: int
    captured_at: float
    observations: tuple[TrackObservation, ...] = ()

    def age(self, now: float) -> float:
        """这个包在 ``now`` 这一刻有多旧。

        两个时钟必须都来自调用方：``captured_at`` 是采集时刻，
        ``now`` 是判断时刻。这里刻意不缓存"包创建时间"——那会让回放
        （跑得比实时快得多）算出一堆荒唐的帧龄。
        """
        return float(now) - float(self.captured_at)

    def __len__(self) -> int:
        return len(self.observations)


class Mailbox:
    """有界的最新值信箱。线程安全。

    ``put`` 满了就挤掉**最旧的**（不是拒收新的）：在最坏的情况下，唯一
    还活着的那一帧必须是最新的那一帧。反过来做的话，信箱里会永远卡着一
    个谁也没见过的老包。

    ``take`` 取走最新的一帧，并把**其余全部丢弃**——它们的年龄必然更大,
    没有任何理由再处理。被丢掉的数量记在 ``dropped`` 里，供诊断使用。
    """

    def __init__(self, capacity: int = 2) -> None:
        if capacity < 1:
            raise ValueError("信箱容量至少为 1")
        self.capacity = int(capacity)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._items: deque[StageAPacket] = deque()
        self.dropped = 0
        """入了信箱却从没被取走的包数。

        这是排查"上报怎么慢了一拍"的第一个指标：它涨，说明 Stage B 跟不上
        Stage A，系统正在用**丢帧**换取时效性——这正是设计意图，但要看得见。
        """

    # -- 写入 -------------------------------------------------------------
    def put(self, packet: StageAPacket) -> int:
        """放入一个包。返回本次因溢出而挤掉的包数。"""
        with self._cond:
            evicted = 0
            while len(self._items) >= self.capacity:
                self._items.popleft()
                evicted += 1
            self._items.append(packet)
            self.dropped += evicted
            self._cond.notify()
            return evicted

    # -- 读取 -------------------------------------------------------------
    def latest(self) -> StageAPacket | None:
        """看一眼最新的一帧，**不取走**。"""
        with self._lock:
            return self._items[-1] if self._items else None

    def take(self) -> StageAPacket | None:
        """取走最新的一帧，其余全部丢弃。空信箱返回 ``None``。"""
        with self._lock:
            if not self._items:
                return None
            newest = self._items[-1]
            self.dropped += len(self._items) - 1
            self._items.clear()
            return newest

    def wait_latest(self, timeout: float | None = None) -> StageAPacket | None:
        """阻塞到有包可取，然后 ``take``。线程化路径用。

        超时返回 ``None``——**不抛异常**：没有新帧是正常运行状态，
        不是错误。调用方据此继续往下走（大概率是继续发预测态）。
        """
        with self._cond:
            if not self._items:
                self._cond.wait(timeout)
            if not self._items:
                return None
            newest = self._items[-1]
            self.dropped += len(self._items) - 1
            self._items.clear()
            return newest

    def peek_all(self) -> tuple[StageAPacket, ...]:
        """当前全部包（旧→新）。诊断与测试用。"""
        with self._lock:
            return tuple(self._items)

    # -- 杂项 -------------------------------------------------------------
    def clear(self) -> None:
        with self._cond:
            self._items.clear()
            self._cond.notify_all()

    def extend(self, packets: Iterable[StageAPacket]) -> int:
        """批量放入，返回挤掉的总数。"""
        return sum(self.put(p) for p in packets)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def reset_counters(self) -> None:
        with self._lock:
            self.dropped = 0
