"""跟踪层：观测 → 轨迹 → 槽位状态。

数据流：

    TrackObservation（像素框 + 可选世界坐标 + 类别/数字）
        │
        ├─ BoxKalman / WorldKalman      预测
        ├─ association                  双闸门 → 代价 → 匈牙利
        ├─ IdentityVote                 身份投票（慢变量）
        ├─ blind.choose_waypoint        长时丢失的落点
        └─ state_machine.resolve_state  四态判定
        ▼
    {槽位名: RobotSlotOutput}

对外只暴露 ``MultiTracker`` 与几个数据结构；上面的每一块都可以单独调用、
单独测试——状态机和航点打分都是纯函数，不需要先搭一条流水线。
"""

from jg_radar.track.association import (
    Associations,
    associate,
    build_cost_matrix,
    gate_mask,
    raw_jump_matrix,
    same_label_matrix,
)
from jg_radar.track.blind import Waypoint, choose_waypoint, parse_waypoints, score_waypoint
from jg_radar.track.identity import IdentityVote, Vote, corroborated_votes
from jg_radar.track.kalman import BoxKalman, WorldKalman, clamp_dt
from jg_radar.track.state_machine import StateThresholds, resolve_state
from jg_radar.track.tracker import MultiTracker
from jg_radar.track.types import Track, TrackObservation, TrackStatus

__all__ = [
    "Associations",
    "BoxKalman",
    "IdentityVote",
    "MultiTracker",
    "StateThresholds",
    "Track",
    "TrackObservation",
    "TrackStatus",
    "Vote",
    "Waypoint",
    "WorldKalman",
    "associate",
    "build_cost_matrix",
    "choose_waypoint",
    "clamp_dt",
    "corroborated_votes",
    "gate_mask",
    "parse_waypoints",
    "raw_jump_matrix",
    "resolve_state",
    "same_label_matrix",
    "score_waypoint",
]
