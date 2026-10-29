"""四态状态机 —— 全项目最该被盯住的一段逻辑。

状态决定下游怎么信任坐标，所以这里的失效模式都是"静默的"：不报错、不崩，
只是把猜的当成看到的。三个设计点：

## 1. 降级链是**时间单调**的

一次丢失期间 ``age`` 只增不减，且只在**真的重新观测到**时才清零。于是
``MEASURED → OCCLUSION_HOLD → BLIND_PREDICTION → MISSING`` 的降级不可能
来回跳：任何一级只能往下走一次。抖动只可能来自"检测时有时无"，而那种情况
本来就该回到 ``MEASURED``。

## 2. 从盲区恢复要**连续命中**

``BLIND_PREDICTION → MEASURED`` 需要连续 ``reacquire_hits`` 次命中，单帧
不算数。理由：盲区期间滤波器已经外推了几秒、协方差很大，此时一个孤立的
假检测（反光、对方装甲板的干扰）足以把轨迹猛地拽过去。而 0.8 秒以内的
``OCCLUSION_HOLD`` 不需要这个保护——滤波器还没跑偏，一帧真检测就该认。

没达标的观测**仍然照常喂进滤波器**，只是对外仍报预测态。这是保守方向：
宁可把一次真的观测晚一帧上报，不可把一次假的观测当真的上报。

## 3. 长时丢失**必须有航点**才叫 BLIND_PREDICTION

没有航点候选就是 ``MISSING``。恒速外推 3 秒的位置不是预测，是编造
（见 ``blind.py``）。**没有候选就先说不知道。**
"""

from __future__ import annotations

# 跨帧的重捕获等待由 MultiTracker/Track.reacquiring 锁存；单靠 previous
# 在中途漏检时会退到 HOLD，绕过连续命中要求。

from dataclasses import dataclass

from jg_radar.types import RobotState

__all__ = ["StateThresholds", "resolve_state"]


@dataclass(frozen=True)
class StateThresholds:
    """状态机的三个阈值。默认值与 ``TrackConfig`` 一致。"""

    lost_predict_s: float = 0.8
    """超过它，从"短时遮挡"转为"长时盲区"。"""
    ttl_s: float = 3.0
    """超过它彻底 MISSING。"""
    reacquire_hits: int = 2
    """从 BLIND_PREDICTION 恢复所需的连续命中数。"""

    @classmethod
    def from_config(cls, cfg) -> "StateThresholds":
        return cls(
            lost_predict_s=float(cfg.lost_predict_s),
            ttl_s=float(cfg.ttl_s),
            reacquire_hits=int(cfg.reacquire_hits),
        )

    def __post_init__(self) -> None:
        if self.ttl_s < self.lost_predict_s:
            raise ValueError(
                f"ttl_s({self.ttl_s}) 小于 lost_predict_s({self.lost_predict_s})："
                "那样 BLIND_PREDICTION 这一段永远不可能出现，状态机会直接跳级"
            )
        if self.reacquire_hits < 1:
            raise ValueError("reacquire_hits 至少为 1")


def resolve_state(
    *,
    matched: bool,
    age_s: float,
    previous: RobotState | None,
    consecutive_hits: int,
    has_filter: bool,
    has_waypoint: bool,
    thresholds: StateThresholds,
) -> RobotState:
    """由"这一帧有没有配上"和"丢了多久"推出对外状态。

    纯函数：给定同样的输入永远给同样的输出，所以状态机的每条分支都可以被
    单独测，不用先搭一条完整流水线。
    """
    if matched:
        if (
            previous is RobotState.BLIND_PREDICTION
            and consecutive_hits < thresholds.reacquire_hits
        ):
            # 已在盲区、还没连上足够多次：继续报预测。观测已经进滤波器了，
            # 所以这里的坐标其实是"滤波后的观测"——把测量**保守地**报成预测，
            # 方向是对的。
            return RobotState.BLIND_PREDICTION
        return RobotState.MEASURED

    if not has_filter:
        # 从来没有过世界坐标，没有任何可外推的东西
        return RobotState.MISSING

    if age_s <= thresholds.lost_predict_s:
        return RobotState.OCCLUSION_HOLD

    if age_s <= thresholds.ttl_s and has_waypoint:
        return RobotState.BLIND_PREDICTION

    return RobotState.MISSING
