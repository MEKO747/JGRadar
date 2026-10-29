"""跟踪层的数据结构。

## ``TrackStatus`` 不是 ``RobotState``

计划里这个枚举叫 ``TrackState``，实现时改名了。两个原因：

1. 本模块与 ``jg_radar.types`` 几乎总是一起被导入，而 ``RobotState`` 才是
   对外的"状态"。再来一个 ``TrackState`` 指的是完全不同的东西，review 时
   一定会有人看错。
2. 它们的语义确实不同：``RobotState`` 描述**坐标的可信度**（给裁判看的），
   ``TrackStatus`` 描述**轨迹的生命周期**（给回收逻辑看的）。一条 ACTIVE
   轨迹可以处于 OCCLUSION_HOLD（本帧没配上，但还年轻）；一条 DEAD 轨迹
   则已经没有任何输出。

``TrackStatus`` 只回答一个问题：**这条轨迹还要不要继续养。**
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from jg_radar.track.blind import Waypoint
from jg_radar.track.identity import IdentityVote
from jg_radar.track.kalman import BoxKalman, WorldKalman
from jg_radar.types import (
    RobotSlotOutput,
    RobotState,
    SourceKind,
    clamp_to_field,
)

__all__ = ["Track", "TrackObservation", "TrackStatus"]


class TrackStatus(str, Enum):
    """轨迹生命周期。

    ``TENTATIVE`` 这一级是必需的，不是保险：没有它，一帧孤立假检测就会新建
    一条轨迹、抢下同一个槽位名、把**已经跟了十几帧的那条**在输出字典里顶掉。
    字典按槽位名索引，谁后写谁赢——静默地把真的换成假的。
    """

    TENTATIVE = "tentative"
    """还没攒够连续命中，**不上报**。攒够之前它只是候选。"""
    CONFIRMED = "confirmed"
    """已确认，可以上报。"""
    LOST = "lost"
    """已确认但本帧没观测到，靠滤波器或航点撑着。"""
    DEAD = "dead"
    """本帧之后回收。"""


@dataclass(frozen=True)
class TrackObservation:
    """一帧里的一次观测。跟踪层的输入，与检测层解耦。

    ``bbox`` 是**原图像素**。``world_cm`` 为 ``None`` 表示这一帧没能把它投到
    场地上（被遮挡、投影落在画面外、查表无效）。**这种观测照样要传进来**：
    它仍然是有用的图像域证据，只是暂时不能产生坐标。
    """

    bbox: tuple[float, float, float, float]
    world_cm: tuple[float, float] | None = None
    confidence: float = 0.0
    car_class: str = ""
    """整车类别名，如 ``"B1"``。既用于关联时的类别门，也用于身份投票。"""
    digit: str = ""
    """装甲板数字读数，如 ``"1"``。只用于印证身份，指不出槽位。"""
    car_index: int | None = None
    """来自一级检测的下标，仅用于诊断与归因。"""

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


@dataclass
class Track:
    """一条被持续跟踪的目标。

    可变对象——每帧原地更新。滤波器状态、身份票箱、状态机记忆都在这里。
    """

    track_id: int
    created_at: float

    box: BoxKalman | None = None
    """图像域滤波器。"""
    world: WorldKalman | None = None
    """场地坐标滤波器。投影一直失败的轨迹会长时间为 ``None``——
    此时它可以被关联、可以攒身份，但**不能上报**。"""

    status: TrackStatus = TrackStatus.TENTATIVE
    matched: bool = False
    """本帧是否配上了观测。状态机的输入之一。"""
    state: RobotState | None = None
    """上一次对外发布的状态。"""

    hits: int = 0
    misses: int = 0
    consecutive_hits: int = 0
    """连续命中数。一次 miss 清零。从盲区恢复要连续够这个数。"""

    last_observed_at: float = float("-inf")
    """最后一次**真正观测到**的时刻。外侧看到的 ``observed_at`` 就是它。"""
    filtered_at: float = float("-inf")
    """滤波器状态对应的时刻。预测时用它算 dt，避免同一段时间被积分两次。"""

    last_bbox: tuple[float, float, float, float] | None = None
    last_center: tuple[float, float] | None = None

    confidence: float = 0.0
    label: str = ""
    identity: IdentityVote = field(default_factory=IdentityVote)

    waypoint: Waypoint | None = None
    waypoint_since: float = float("-inf")
    reacquiring: bool = False

    # -- 几何 -------------------------------------------------------------
    def age(self, now: float) -> float:
        """距最后一次真正观测过了多久（秒）。从未观测过时返回 ``inf``。"""
        if not math.isfinite(self.last_observed_at):
            return float("inf")
        return max(0.0, float(now) - self.last_observed_at)

    @property
    def predicted_bbox(self) -> tuple[float, float, float, float] | None:
        """当前（已外推的）图像框。"""
        if self.box is not None:
            return self.box.bbox
        return self.last_bbox

    def predict_to(self, now: float) -> None:
        """把两个滤波器都推到 ``now``。

        ``dt`` 由 ``filtered_at`` 算，所以连着调两次不会把同一个时间段积分
        两遍——这在"有的轨迹本帧配上了、有的没配上"的分支里很容易写错。
        """
        if not math.isfinite(self.filtered_at):
            self.filtered_at = float(now)
            return
        dt = float(now) - self.filtered_at
        if dt <= 0.0:
            return
        if self.box is not None:
            self.box.predict(dt)
        if self.world is not None:
            self.world.predict(dt)
        self.filtered_at = float(now)

    # -- 更新 -------------------------------------------------------------
    def observe(
        self,
        obs: TrackObservation,
        *,
        now: float,
        box_noise: float,
        world_noise: float,
        box_process: float,
        world_process: float,
        min_hits: int = 2,
        confirm_observation: bool = True,
    ) -> None:
        """用一次观测更新轨迹。"""
        if self.box is None:
            self.box = BoxKalman(
                obs.bbox, process_noise=box_process, measurement_noise=box_noise
            )
        else:
            self.box.update_bbox(obs.bbox)

        if obs.world_cm is not None:
            if self.world is None:
                self.world = WorldKalman(
                    obs.world_cm,
                    process_noise=world_process,
                    measurement_noise=world_noise,
                )
            else:
                self.world.update(obs.world_cm)

        self.hits += 1
        self.consecutive_hits += 1
        if confirm_observation:
            self.last_observed_at = float(now)
        self.filtered_at = float(now)
        self.last_bbox = obs.bbox
        self.last_center = obs.center
        self.confidence = float(obs.confidence)
        self.matched = True
        # 用**累计**命中数判确认，且确认是**单向门**。
        # 用连续命中数会出事：一条跟了很久的轨迹丢失后重新被看到，第一帧的
        # 连续命中数是 1，就会被降回 TENTATIVE，于是它在输出的那一帧凭空消失。
        # 一个已经确认过的目标不该因为"刚重新看到"而变成候选。
        if self.hits >= max(int(min_hits), 1):
            self.status = TrackStatus.CONFIRMED
        else:
            self.status = TrackStatus.TENTATIVE

    def miss(self) -> None:
        """本帧没配上。

        从未确认过的轨迹**直接判死**，不给它遮挡保持的机会：它没有任何历史，
        外推出来的东西也没有依据。一帧假检测应该只活一帧。
        """
        self.misses += 1
        self.consecutive_hits = 0
        self.matched = False
        self.status = (
            TrackStatus.DEAD
            if self.status is TrackStatus.TENTATIVE
            else TrackStatus.LOST
        )

    # -- 对外 -------------------------------------------------------------
    def position_for(self, state: RobotState) -> tuple[float, float] | None:
        """按状态取坐标。

        ``BLIND_PREDICTION`` 用**航点**而不是滤波器外推——见 ``blind.py``。
        拿不到航点时退回滤波器位置（调用方本就该在无航点时判 MISSING）。
        """
        if state is RobotState.MISSING:
            return None
        if state is RobotState.BLIND_PREDICTION and self.waypoint is not None:
            return self.waypoint.position_cm
        if self.world is not None:
            return self.world.position_cm
        return None

    def to_slot_output(
        self,
        *,
        name: str,
        state: RobotState,
        now: float,
        source: SourceKind = SourceKind.VISION,
        confidence_half_life_s: float = 1.0,
    ) -> RobotSlotOutput:
        """构造一条对外的槽位输出。

        坐标**钳到场内**。滤波器在遮挡期间会外推，贴着边线的目标很容易算出
        ``x = 2810`` 这种值，而契约 I2 要求坐标在场地内——不钳的话
        ``RobotSlotOutput.validate()`` 会在运行中抛异常，一条正常的边界外推
        就能把整条流水线打断。
        """
        if state is RobotState.MISSING:
            return RobotSlotOutput(
                name=name,
                position_cm=None,
                state=state,
                confidence=0.0,
                observed_at=float(now),
                source=source,
                is_predicted=False,
                track_id=self.track_id,
                position_std_cm=None,
            )

        raw = self.position_for(state)
        if raw is None:
            # 状态说有坐标但拿不出来（世界滤波器缺失）。**不能**在这里
            # 悄悄补一个 (0,0)：那是一个真实的场地坐标，会被当成真位置上报。
            return RobotSlotOutput(
                name=name,
                position_cm=None,
                state=RobotState.MISSING,
                confidence=0.0,
                observed_at=float(now),
                source=source,
                is_predicted=False,
                track_id=self.track_id,
                position_std_cm=None,
            )

        x, y = clamp_to_field(float(raw[0]), float(raw[1]))
        std = (
            float(self.world.position_std_cm)
            if self.world is not None
            else None
        )
        return RobotSlotOutput(
            name=name,
            position_cm=(int(round(x)), int(round(y))),
            state=state,
            confidence=self.decayed_confidence(
                now, half_life_s=confidence_half_life_s
            ),
            # 预测态的 observed_at 停在最后一次真观测上，于是下游能直接从
            # `published_at - observed_at` 看出这条信息有多旧。
            observed_at=float(self.last_observed_at),
            source=source,
            is_predicted=state.is_prediction,
            track_id=self.track_id,
            position_std_cm=std,
            prediction_offset_cm=(
                math.dist((int(round(x)), int(round(y))), self.world.position_cm)
                if state.is_prediction and self.world is not None else 0.0
            ),
        )

    def decayed_confidence(self, now: float, *, half_life_s: float = 1.0) -> float:
        """置信度按丢失时长衰减。

        遮挡中置信度**必须下降**。一个恒定的置信度会让下游以为"和刚看到时
        一样可信"，而它是唯一能横向比较不同来源(视觉/无线电)的标量。
        """
        if not math.isfinite(self.last_observed_at):
            return 0.0
        age = self.age(now)
        raw = self.confidence if age <= 0.0 else (
            self.confidence * 0.5 ** (age / max(float(half_life_s), 1e-6))
        )
        return float(min(max(raw, 0.0), 1.0))
