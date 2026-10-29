"""视觉源的接线：跟踪器输出 → ``SourceSample``。

这层薄得几乎不像一层，但它承担一个**必须做对**的细节：``MISSING`` 也必须写进
缓冲。

最新值缓冲只保留最新的一条，所以写入本身带有"撤回"的效果。设想 R1 这条轨迹
刚刚越过 ttl：跟踪器这一帧发 ``MISSING``，下一帧起就不再产出它了。如果这一帧的
``MISSING`` 被这层"优化"掉（毕竟它没有坐标、任何档位都不会采纳它），缓冲里
留着的就是 0.1 秒前那条 ``MEASURED`` ——而 0.1 秒远小于 500ms 的看门狗，
于是仲裁层会**继续把一个已经被判定为"不在了"的坐标当成实时测量报出去**。

这正是契约里最不能容忍的失效模式：陈旧数据伪装成新鲜测量。
所以规则很简单：**跟踪器报什么就写什么**，包括 ``MISSING``。
"""

from __future__ import annotations

from typing import Mapping, Sequence
import threading

from jg_radar.types import RobotSlotOutput, SourceKind, SourceSample, missing_slot

__all__ = ["VisionSource"]


class VisionSource:
    """把一个跟踪器的每帧输出转成样本并暂存。

    ``submit`` 在**视觉线程**里调用，``poll`` 在**发布线程**里调用——两者之间
    隔着一次真正的线程切换，所以暂存队列不是多余的：视觉推理要几十毫秒，
    发布不该等它。
    """

    kind = SourceKind.VISION

    def __init__(self, name: str = "vision") -> None:
        self._name = str(name)
        self._pending: list[SourceSample] = []
        self._revision = 0
        self._lock = threading.Lock()
        self._known: set[str] = set()

    @property
    def name(self) -> str:
        return self._name

    def submit(
        self, slots: Mapping[str, RobotSlotOutput], *, now: float
    ) -> list[SourceSample]:
        """视觉线程每帧调用。返回本次产生的样本（同时也已暂存）。"""
        with self._lock:
            return self._submit(slots, now=now)

    def _submit(self, slots, *, now):
        made: list[SourceSample] = []
        removed = self._known - slots.keys()
        values = list(slots.values()) + [missing_slot(name, now) for name in sorted(removed)]
        self._known = set(slots)
        for slot in values:
            self._revision += 1
            made.append(
                SourceSample(
                    source=SourceKind.VISION,
                    name=slot.name,
                    position_cm=slot.position_cm,
                    state=slot.state,
                    confidence=float(slot.confidence),
                    # 跟踪器算好的采集时刻，原样带过来。**不要在这里用 now**：
                    # 预测态的 observed_at 就该停在最后一次真观测上，
                    # 那是设计，不是遗漏。
                    observed_at=float(slot.observed_at),
                    revision=self._revision,
                    is_predicted=bool(slot.is_predicted),
                    track_id=slot.track_id,
                    position_std_cm=slot.position_std_cm,
                    prediction_offset_cm=slot.prediction_offset_cm,
                    estimated_at=float(now),
                )
            )
        # 每个槽位只留最新值，独立 poll 消费者暂时不运行也不会积压历史。
        latest = {sample.name: sample for sample in self._pending}
        latest.update({sample.name: sample for sample in made})
        self._pending = list(latest.values())
        return made

    def poll(self, now: float) -> Sequence[SourceSample]:
        """发布线程调用：取走并清空。``now`` 未使用——样本自带时间戳。"""
        del now
        with self._lock:
            out = self._pending
            self._pending = []
            return out

    def reset(self) -> None:
        with self._lock:
            self._pending.clear()
            self._revision = 0
            self._known.clear()
