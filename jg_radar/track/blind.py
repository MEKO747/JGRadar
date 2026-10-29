"""盲区航点：目标消失很久之后，猜它**大概在哪**。

## 为什么不能只靠滤波器外推

``WorldKalman`` 是恒速模型，遮挡 3 秒后它会沿着消失那一刻的方向一路推出去
几米。那不是"预测"，那是外推误差的累积——RM 的机器人会拐弯、会停、会掉头，
恒速假设在 0.8 秒内还算数，到 3 秒就是纯粹的编造。

所以长时丢失（``age > lost_predict_s``）**必须**落到一个**离散的、有先验的**
候选集上：航点。这是在说"它不在我猜的那条直线上，但它多半在基地/高地/某个
关键位置附近"，比编一条直线诚实得多。

## 打分

    score = exp(−距离/decay) + w · (cos(速度, 指向航点) + 1)/2

  - 距离项：越近越可能。用指数而不是反比，是为了让它有界且衰减得干脆。
  - 速度项：正在朝那个方向跑的，更可能到那儿。用 ``(cos+1)/2`` 把余弦映到
    ``[0,1]`` 而不是让它取负——**分数必须非负**，否则下面的重访惩罚乘上去
    会变成"越该罚的越推荐"（负负得正）。

**航点间必须有最小驻留时间**：两个航点分数接近时，逐帧重算会来回跳，输出
位置就会在两地之间反复横跳。跳动比慢一点更糟——裁判侧的"连续 3 秒"是按
位置稳定性判的。

**盲区输出永不重置滤波器**：航点只是**发布什么**，不是**相信什么**。
滤波器继续按它自己的协方差发散，一旦真检测回来，直接用它更新，
不会因为"刚才猜在了高地上"而被污染。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = ["Waypoint", "choose_waypoint", "score_waypoint", "parse_waypoints"]

_EPS = 1e-9


@dataclass(frozen=True)
class Waypoint:
    """一个候选落点。"""

    name: str
    position_cm: tuple[float, float]


def parse_waypoints(spec: Iterable[Sequence]) -> tuple[Waypoint, ...]:
    """把配置里的 ``(名字, x, y)`` 三元组列表转成航点。

    配置写的是三元组而不是 ``{名字: [x, y]}``，这样 YAML 里的每个航点都过一遍
    类型校验：写错一个坐标会在**启动时**报错，而不是等到盲区真的触发、
    在赛场上才发现少了一个航点。
    """
    out: list[Waypoint] = []
    seen: set[str] = set()
    for item in spec or ():
        name, x, y = item
        name = str(name)
        if name in seen:
            raise ValueError(f"航点重名：{name}（名字是重访判定的依据，必须唯一）")
        seen.add(name)
        out.append(Waypoint(name=name, position_cm=(float(x), float(y))))
    return tuple(out)


def score_waypoint(
    wp: Waypoint,
    *,
    position: Sequence[float],
    velocity: Sequence[float],
    decay_cm: float = 400.0,
    speed_cos_weight: float = 1.0,
    previous_name: str | None = None,
    revisit_penalty: float = 0.3,
) -> float:
    """给单个航点打分，越大越可能。返回值恒非负。"""
    pos = np.asarray(position, dtype=np.float64).reshape(2)
    vel = np.asarray(velocity, dtype=np.float64).reshape(2)
    target = np.asarray(wp.position_cm, dtype=np.float64).reshape(2)

    delta = target - pos
    dist = float(np.linalg.norm(delta))

    score = float(np.exp(-dist / max(float(decay_cm), _EPS)))

    speed = float(np.linalg.norm(vel))
    if speed > _EPS and dist > _EPS:
        cos = float(np.dot(vel, delta) / (speed * dist))
        score += float(speed_cos_weight) * 0.5 * (cos + 1.0)

    if previous_name is not None and wp.name == previous_name:
        score *= max(0.0, 1.0 - float(revisit_penalty))
    return max(score, 0.0)


def choose_waypoint(
    waypoints: Sequence[Waypoint],
    *,
    position: Sequence[float],
    velocity: Sequence[float],
    previous: Waypoint | None = None,
    last_change_at: float = float("-inf"),
    now: float = 0.0,
    min_dwell_s: float = 0.5,
    decay_cm: float = 400.0,
    speed_cos_weight: float = 1.0,
    revisit_penalty: float = 0.3,
) -> tuple[Waypoint | None, bool]:
    """选航点。返回 ``(航点, 是否发生了切换)``。

    没配航点返回 ``(None, False)``——上层据此把状态判成 ``MISSING`` 而不是
    ``BLIND_PREDICTION``：**没有候选就先说不知道。**

    平局取配置顺序里的第一个。这不是随便定的：回放两次必须给出同一个航点，
    而平局在场地中心、速度为零时是常态（两个基地左右对称，分数完全相等）。
    """
    if not waypoints:
        return None, False

    if (
        previous is not None
        and (now - last_change_at) < float(min_dwell_s)
        and any(wp.name == previous.name for wp in waypoints)
    ):
        # 驻留期内不换点。previous 可能已经不在候选里了（配置热更新），
        # 那种情况必须重新选，否则会一直输出一个配置里已删除的航点。
        return previous, False

    best = max(
        waypoints,
        key=lambda wp: score_waypoint(
            wp,
            position=position,
            velocity=velocity,
            decay_cm=decay_cm,
            speed_cos_weight=speed_cos_weight,
            previous_name=previous.name if previous is not None else None,
            revisit_penalty=revisit_penalty,
        ),
    )
    changed = previous is None or best.name != previous.name
    return best, changed
