"""金标准帧序列的构造器。

刻意**不用真实时钟**：所有时间戳都是写死的常数。金标准文件必须能在任何机器、
任何时刻生成出逐字节相同的结果，否则它就失去了"回归门"的意义。
"""

from __future__ import annotations

from jg_radar.types import (
    FrameOutput,
    RobotSlotOutput,
    RobotState,
    SourceKind,
    default_slot_names,
)

ALL_SLOTS = default_slot_names()


def _slot(
    name: str,
    pos: tuple[int, int] | None,
    state: RobotState,
    *,
    conf: float = 0.0,
    at: float,
    track_id: int | None = None,
    std: float | None = None,
    source: SourceKind = SourceKind.VISION,
) -> RobotSlotOutput:
    return RobotSlotOutput(
        name=name,
        position_cm=pos,
        state=state,
        confidence=conf,
        observed_at=at,
        source=source,
        is_predicted=state.is_prediction,
        track_id=track_id,
        position_std_cm=std,
    )


def build_golden_frames() -> list[FrameOutput]:
    """四帧，覆盖四种状态以及状态间的迁移。

    frame 0  t=100.00  只有 R1 被看到，其余全 MISSING（冷启动）
    frame 1  t=100.05  R1 / R2 / B3 被看到
    frame 2  t=100.10  R1 转入遮挡外推，B3 转入盲区预测，R2 仍被看到
    frame 3  t=100.15  R1 重新看到；R2 超时彻底 MISSING；B3 仍在盲区
    """
    frames: list[FrameOutput] = []

    # -- frame 0 ----------------------------------------------------------
    robots = {name: _slot(name, None, RobotState.MISSING, at=100.0) for name in ALL_SLOTS}
    robots["R1"] = _slot("R1", (1200, 700), RobotState.MEASURED, conf=0.91, at=100.0,
                         track_id=1, std=3.2)
    frames.append(FrameOutput(0, 100.00, 100.021, robots, {"fps_ema": 20.0}))

    # -- frame 1 ----------------------------------------------------------
    robots = {name: _slot(name, None, RobotState.MISSING, at=100.05) for name in ALL_SLOTS}
    robots["R1"] = _slot("R1", (1213, 704), RobotState.MEASURED, conf=0.93, at=100.05,
                         track_id=1, std=2.9)
    robots["R2"] = _slot("R2", (860, 402), RobotState.MEASURED, conf=0.77, at=100.05,
                         track_id=2, std=4.1)
    robots["B3"] = _slot("B3", (2100, 1180), RobotState.MEASURED, conf=0.64, at=100.05,
                         track_id=3, std=6.8)
    frames.append(FrameOutput(1, 100.05, 100.071, robots, {"fps_ema": 20.1}))

    # -- frame 2 ----------------------------------------------------------
    robots = {name: _slot(name, None, RobotState.MISSING, at=100.10) for name in ALL_SLOTS}
    # R1 丢失 → 遮挡外推；observed_at 停在最后一次真正观测的时刻
    robots["R1"] = _slot("R1", (1226, 708), RobotState.OCCLUSION_HOLD, conf=0.93,
                         at=100.05, track_id=1, std=7.5)
    robots["R2"] = _slot("R2", (874, 409), RobotState.MEASURED, conf=0.81, at=100.10,
                         track_id=2, std=3.9)
    # B3 丢失较久 → 盲区预测；标准差明显放大
    robots["B3"] = _slot("B3", (2180, 1240), RobotState.BLIND_PREDICTION, conf=0.64,
                         at=100.05, track_id=3, std=45.0)
    frames.append(FrameOutput(2, 100.10, 100.122, robots, {"fps_ema": 19.9}))

    # -- frame 3 ----------------------------------------------------------
    robots = {name: _slot(name, None, RobotState.MISSING, at=100.15) for name in ALL_SLOTS}
    robots["R1"] = _slot("R1", (1240, 713), RobotState.MEASURED, conf=0.90, at=100.15,
                         track_id=1, std=3.0)
    robots["R2"] = _slot("R2", None, RobotState.MISSING, at=100.15, track_id=2)
    robots["B3"] = _slot("B3", (2200, 1260), RobotState.BLIND_PREDICTION, conf=0.60,
                         at=100.05, track_id=3, std=52.0)
    frames.append(FrameOutput(3, 100.15, 100.171, robots, {"fps_ema": 20.0}))

    return frames
