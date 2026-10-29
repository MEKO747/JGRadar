"""输出契约 —— 全项目唯一真相源。

本模块定义视觉流水线对外的全部数据形状。任何其他模块都不得自造结构。

三条不变式（由 ``tests/test_types_contract.py`` 锁死）：

  I1  ``is_predicted`` ⇔ ``state ∈ {OCCLUSION_HOLD, BLIND_PREDICTION}``
      —— 预测永不伪装成测量。下游必须能一眼分辨"看到的"和"猜的"。
      注意：``MISSING`` 不是预测（它根本没有坐标），故 ``is_predicted=False``。
      方案原文写作 ``is_predicted ⇔ state != MEASURED``，那样会让 MISSING 被标成
      预测，与"没有坐标"自相矛盾；实现时收紧为上述形式。
  I2  ``position_cm`` 非空时必落在 ``[0, FIELD_MAX_X_CM] × [0, FIELD_MAX_Y_CM]``
  I3  保留真实观测时间；发布顺序由 ``publication_seq`` / ``published_at`` 表达。

设计来源：
  - 状态语义取自 shark-radar-vision ``tracking/types.py`` 的 ``TargetState``
  - 遥测字段取自 shark ``vision_telemetry.py``，但补上了 ``source`` / ``track_id``
    / ``position_std_cm``，让下游能判断证据强度
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping
import math

SCHEMA_VERSION = "jg.radar.robot.v1"

# --------------------------------------------------------------------------
# 场地常量（cm）。RMUC 2026 场地 28m × 15m。
# --------------------------------------------------------------------------
FIELD_MAX_X_CM = 2800
FIELD_MAX_Y_CM = 1500

# 雷达上报协议（0x0305 的 map_robot_data_t）覆盖每方 6 个角色：
# 英雄 1 / 工程 2 / 步兵 3 / 步兵 4 / 空中 6 / 哨兵 7。
# 这是默认槽位列表，可由配置覆盖（见 jg_radar.config.schema.RobotsConfig）。
DEFAULT_ROLES: tuple[str, ...] = ("1", "2", "3", "4", "6", "7")
DEFAULT_SIDES: tuple[str, ...] = ("R", "B")


def default_slot_names(
    roles: tuple[str, ...] = DEFAULT_ROLES,
    sides: tuple[str, ...] = DEFAULT_SIDES,
) -> tuple[str, ...]:
    """返回默认槽位名，顺序固定为「先 R 后 B，角色升序」。"""
    return tuple(f"{side}{role}" for side in sides for role in roles)


# --------------------------------------------------------------------------
# 枚举
# --------------------------------------------------------------------------
class RobotState(str, Enum):
    """单个机器人槽位的观测状态。

    这是本项目最重要的一个枚举：它决定了裁判与下游如何信任这个坐标。
    """

    MEASURED = "measured"
    """本帧真实检测到并成功定位。唯一可以喂给滤波器的状态。"""

    OCCLUSION_HOLD = "occlusion_hold"
    """短时丢失，用滤波器外推。位置是估计值，不是观测值。"""

    BLIND_PREDICTION = "blind_prediction"
    """长时丢失，落到盲区航点。位置是猜的。"""

    MISSING = "missing"
    """无任何信息。position_cm 必须为 None。"""

    @property
    def is_measurement(self) -> bool:
        """只有 MEASURED 是观测，其余都不是。"""
        return self is RobotState.MEASURED

    @property
    def is_prediction(self) -> bool:
        """是否有坐标且坐标是推断出来的。MISSING 无坐标，不算预测。"""
        return self in (RobotState.OCCLUSION_HOLD, RobotState.BLIND_PREDICTION)


class SourceKind(str, Enum):
    """坐标的来源。用于融合仲裁与事后归因。"""

    VISION = "vision"
    RADIO = "radio"
    FUSED = "fused"


# --------------------------------------------------------------------------
# 契约数据结构
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceSample:
    """融合层的通用输入。视觉与无线电共用这一个形状。

    无线电接入时只需构造本类型并写入仲裁缓冲，视觉侧零改动。
    """

    source: SourceKind
    name: str
    position_cm: tuple[int, int] | None
    state: RobotState
    confidence: float
    observed_at: float
    """单调时钟，且必须是**采集时刻**而非推理完成时刻。"""
    revision: int
    """同一来源内单调递增，用于判断"这是不是一条新样本"。"""
    is_predicted: bool
    class_id: int | None = None
    track_id: int | None = None
    """产生这条样本的轨迹 id。无线电没有这个概念，留 ``None``。"""
    position_std_cm: float | None = None
    """位置标准差。**这个数本身跨源不可比**：视觉给的是卡尔曼协方差，无线电
    给的是链路自己的估计，两者的口径和含义都不一样，拿来互相比较是错的。

    但仲裁器**确实**会用它——只在 ``is_predicted`` 的样本上，用来算
    :func:`jg_radar.fuse.arbiter.position_error_bound_cm`（取 3σ 当作上界）。
    预测态没有别的可用信息：它的 ``observed_at`` 刻意冻在最后一次真观测上，
    拿它算年龄会得到"9 秒前"这种数，从而把每一条盲区预测都误杀。

    实测态的样本**不看这个字段**，改用"信息年龄 × 最大速度"这个硬上界——
    那个只依赖时间，所以天然跨源可比，正好绕开上面那个口径问题。"""
    prediction_offset_cm: float = 0.0
    """输出点相对滤波预测中心的距离；航点/边界裁剪需计入误差。"""
    estimated_at: float | None = None
    """预测计算时刻，独立于最后真实观测时刻；暂存期间误差继续增长。"""


@dataclass(frozen=True)
class RobotSlotOutput:
    """单个机器人槽位的对外输出。"""

    name: str
    position_cm: tuple[int, int] | None
    state: RobotState
    confidence: float
    observed_at: float
    """该槽位最近一次获得有效信息的时刻（单调时钟）。

    - MEASURED：本次观测的**采集**时刻，不是推理完成时刻
    - OCCLUSION_HOLD / BLIND_PREDICTION：最后一次真正观测的时刻（所以它会停滞）
    - MISSING：判定时刻。"确认它不在了"本身也是一条新信息。
    下游用 ``published_at - observed_at`` 判断信息新鲜度。
    """
    source: SourceKind
    is_predicted: bool
    track_id: int | None = None
    position_std_cm: float | None = None
    """滤波器预测中心的标准差；报告点的偏移另见 prediction_offset_cm。"""
    schema_version: str = SCHEMA_VERSION
    prediction_offset_cm: float = 0.0
    """报告点相对滤波中心的偏移；预测误差需包含该值及暂存时间的影响。"""

    # -- 校验 -------------------------------------------------------------
    def validate(self) -> None:
        """违反任一条不变式即抛 ``ValueError``。

        流水线在每次发布前调用，避免坏数据流出进程。
        """
        if not math.isfinite(self.prediction_offset_cm) or self.prediction_offset_cm < 0:
            raise ValueError(f"{self.name}: 预测点偏移必须是有限非负数")
        if self.is_predicted != self.state.is_prediction:
            raise ValueError(
                f"{self.name}: 违反不变式 I1 —— "
                f"is_predicted={self.is_predicted} 但 state={self.state.value}"
                f"（应为 {self.state.is_prediction}）"
            )

        if self.state is RobotState.MISSING and self.position_cm is not None:
            raise ValueError(f"{self.name}: MISSING 状态必须没有坐标")

        if self.position_cm is not None:
            x, y = self.position_cm
            if not is_in_field(x, y):
                raise ValueError(
                    f"{self.name}: 违反不变式 I2 —— 坐标 ({x}, {y}) 越界"
                )

        if self.position_cm is None and self.state is not RobotState.MISSING:
            raise ValueError(
                f"{self.name}: state={self.state.value} 却没有坐标"
            )

    @property
    def is_valid(self) -> bool:
        """能否用于对外上报。"""
        return self.position_cm is not None and self.state is not RobotState.MISSING

    # -- 序列化 -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "position_cm": list(self.position_cm) if self.position_cm else None,
            "state": self.state.value,
            "confidence": round(float(self.confidence), 4),
            "observed_at": round(float(self.observed_at), 6),
            "source": self.source.value,
            "is_predicted": self.is_predicted,
            "track_id": self.track_id,
            "position_std_cm": (
                round(float(self.position_std_cm), 3)
                if self.position_std_cm is not None
                else None
            ),
            "schema_version": self.schema_version,
        }
        if self.prediction_offset_cm:
            data["prediction_offset_cm"] = round(float(self.prediction_offset_cm), 3)
        return data

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RobotSlotOutput":
        pos = d.get("position_cm")
        return cls(
            name=d["name"],
            position_cm=(int(pos[0]), int(pos[1])) if pos else None,
            state=RobotState(d["state"]),
            confidence=float(d["confidence"]),
            observed_at=float(d["observed_at"]),
            source=SourceKind(d["source"]),
            is_predicted=bool(d["is_predicted"]),
            track_id=d.get("track_id"),
            position_std_cm=d.get("position_std_cm"),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            prediction_offset_cm=float(d.get("prediction_offset_cm", 0.0)),
        )


@dataclass(frozen=True)
class FrameOutput:
    """一帧的完整输出。"""

    frame_seq: int
    captured_at: float
    published_at: float
    robots: Mapping[str, RobotSlotOutput] = field(default_factory=dict)
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    publication_seq: int | None = None

    def validate(self) -> None:
        for slot in self.robots.values():
            slot.validate()

    def to_dict(self) -> dict[str, Any]:
        data = {
            "frame_seq": self.frame_seq,
            "captured_at": round(float(self.captured_at), 6),
            "published_at": round(float(self.published_at), 6),
            "robots": {name: slot.to_dict() for name, slot in self.robots.items()},
            "diagnostics": {
                k: round(float(v), 4) for k, v in self.diagnostics.items()
            },
        }
        if self.publication_seq is not None:
            data["publication_seq"] = self.publication_seq
        return data

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FrameOutput":
        return cls(
            frame_seq=int(d["frame_seq"]),
            captured_at=float(d["captured_at"]),
            published_at=float(d["published_at"]),
            robots={
                name: RobotSlotOutput.from_dict(s)
                for name, s in d.get("robots", {}).items()
            },
            diagnostics={k: float(v) for k, v in d.get("diagnostics", {}).items()},
            publication_seq=d.get("publication_seq"),
        )


# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------
def is_in_field(x: float, y: float) -> bool:
    """坐标是否落在场地范围内（含边界）。"""
    return 0 <= x <= FIELD_MAX_X_CM and 0 <= y <= FIELD_MAX_Y_CM


def clamp_to_field(x: float, y: float) -> tuple[float, float]:
    """把坐标钳到场地内。

    给**外推出来的**坐标用：滤波器在遮挡期间会继续积分，贴着边线的目标很容易
    算出 ``x = 2810``。而此时 ``validate()`` 会抛异常——一条正常的边界外推
    不应该能把整条流水线打断。场地是有界的，钳住是物理事实，不是掩盖。
    """
    return (
        min(max(float(x), 0.0), float(FIELD_MAX_X_CM)),
        min(max(float(y), 0.0), float(FIELD_MAX_Y_CM)),
    )


def missing_slot(name: str, observed_at: float) -> RobotSlotOutput:
    """构造一个 MISSING 槽位，供降级路径使用。"""
    return RobotSlotOutput(
        name=name,
        position_cm=None,
        state=RobotState.MISSING,
        confidence=0.0,
        observed_at=observed_at,
        source=SourceKind.VISION,
        is_predicted=False,
    )
