"""发布层：补全固定槽位，并保留被采用样本的真实观测时间。

MISSING 的 observed_at 是判定时刻；其余状态保留采集/最后确认观测时刻。
切换来源允许采用更旧的有效样本，不能通过抬高时间戳掩盖信息年龄。
publication_seq 单调递增，published_at 表达发布时刻；frame_seq 仍指向视觉帧。
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from jg_radar.fuse.arbiter import SlotDecision
from jg_radar.types import (
    FrameOutput,
    RobotSlotOutput,
    RobotState,
    SourceKind,
    default_slot_names,
)

__all__ = ["PublishLayer"]


class PublishLayer:
    """保留各槽位的上一次来源，独立记录发布序号，不改写观测时间。"""

    def __init__(self, config=None, *, slot_names: Sequence[str] | None = None) -> None:
        self.cfg = config
        self.slot_names: tuple[str, ...] = (
            tuple(slot_names) if slot_names else default_slot_names()
        )
        self._last_source: dict[str, SourceKind] = {}
        self.frames = 0
        #: 每帧每个槽位落到了哪个档位。档位名按仲裁器的 ``tier`` 记，
        #: 没有任何候选的槽位记成 ``missing``——"预测占多少比例"这种问题
        #: 必须能用一次运行自己的日志回答，而不是靠人肉翻帧。
        self.tier_counts: Counter[str] = Counter()

    def build(
        self,
        decisions: Mapping[str, SlotDecision],
        *,
        frame_seq: int,
        captured_at: float,
        published_at: float,
        diagnostics: Mapping[str, float] | None = None,
    ) -> FrameOutput:
        """落成一帧。**返回的帧一定包含全部 ``slot_names``。**"""
        robots: dict[str, RobotSlotOutput] = {}
        filled = 0
        for name in self.slot_names:
            slot = self._to_slot(name, decisions.get(name), published_at)
            if slot.state is RobotState.MISSING:
                filled += 1
            robots[name] = slot

        # 不属于任何槽位的决策（比如无线电报了一个我们不认识的机器人名）
        # **丢掉但计数**：静默丢弃会让一个配置错位查不出来。
        unknown = sum(1 for name in decisions if name not in robots)

        diag: dict[str, float] = {
            "publish_frames": float(self.frames),
            "filled_missing": float(filled),
            "unknown_slots": float(unknown),
        }
        # 每个槽位的信息年龄和 stale 标记单独摊开：``RobotSlotOutput`` 的
        # 契约已经带了 ``source`` 和 ``observed_at``，年龄可以从两者算出来，
        # 但"算得出来"和"一眼看得见"是两件事——将来按规则记错的 0x0305
        # 发送端要的就是这一眼。
        for name in robots:
            decision = decisions.get(name)
            self.tier_counts["missing" if decision is None or decision.tier is None
                             else decision.tier] += 1

        stale_slots = 0
        for name, decision in decisions.items():
            if name not in robots or decision is None:
                continue
            if decision.age_s is not None:
                diag[f"{name}_age_s"] = float(decision.age_s)
            if decision.stale:
                stale_slots += 1
                diag[f"{name}_stale"] = 1.0
        diag["stale_slots"] = float(stale_slots)
        if diagnostics:
            diag.update({k: float(v) for k, v in diagnostics.items()})

        frame = FrameOutput(
            frame_seq=int(frame_seq),
            captured_at=float(captured_at),
            published_at=float(published_at),
            robots=robots,
            diagnostics=diag,
            publication_seq=self.frames,
        )
        frame.validate()  # 坏数据不出进程
        self.frames += 1
        return frame

    # -- 单槽位 -----------------------------------------------------------
    def _to_slot(
        self, name: str, decision: SlotDecision | None, fallback_at: float
    ) -> RobotSlotOutput:
        sample = None if decision is None else decision.sample

        if sample is None:
            # 没有候选。归因给"上一次在供这个槽位的源"——无线电报的槽位消失了，
            # 说它是 VISION 的 MISSING 会把排查的人引到错误的地方。
            source = self._last_source.get(name, SourceKind.VISION)
            observed = float(fallback_at)
            slot = RobotSlotOutput(
                name=name,
                position_cm=None,
                state=RobotState.MISSING,
                confidence=0.0,
                observed_at=observed,
                source=source,
                is_predicted=False,
            )
        else:
            observed = float(sample.observed_at)
            self._last_source[name] = sample.source
            slot = RobotSlotOutput(
                name=name,
                position_cm=sample.position_cm,
                state=sample.state,
                confidence=float(sample.confidence),
                observed_at=observed,
                source=sample.source,
                is_predicted=bool(sample.is_predicted),
                track_id=sample.track_id,
                position_std_cm=sample.position_std_cm,
                prediction_offset_cm=sample.prediction_offset_cm,
            )

        slot.validate()
        return slot

    def tier_share(self) -> dict[str, float]:
        """各档位占总槽位帧的比例，按占比从高到低。"""
        total = sum(self.tier_counts.values())
        if total <= 0:
            return {}
        return {
            tier: count / total
            for tier, count in sorted(self.tier_counts.items(), key=lambda item: -item[1])
        }

    def reset(self) -> None:
        self._last_source.clear()
        self.frames = 0
        self.tier_counts.clear()
