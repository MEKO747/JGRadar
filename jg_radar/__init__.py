"""JGRadar —— 雷达站视觉算法核心。

设计要点见 /home/meko/.claude/plans/greedy-watching-boole.md
"""

from jg_radar.types import (
    FIELD_MAX_X_CM,
    FIELD_MAX_Y_CM,
    SCHEMA_VERSION,
    FrameOutput,
    RobotSlotOutput,
    RobotState,
    SourceKind,
    SourceSample,
)

__all__ = [
    "FIELD_MAX_X_CM",
    "FIELD_MAX_Y_CM",
    "SCHEMA_VERSION",
    "FrameOutput",
    "RobotSlotOutput",
    "RobotState",
    "SourceKind",
    "SourceSample",
]

__version__ = "0.1.0"
