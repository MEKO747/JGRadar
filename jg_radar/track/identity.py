"""带时间衰减和切换门槛的完整身份投票。

完整车类身份优先，数字仅作印证：相符加权，冲突削弱，数字独立不指定阵营。
持续数字冲突不会单独覆盖完整车类；完整身份的新证据可以累计纠正旧判断。
通用车辆模型由 Stage A 将同车装甲颜色与数字组合成完整身份后再投票。
空间关联不使用单帧身份硬门禁，因此纠错证据可以留在同一物理轨迹内。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

__all__ = ["IdentityVote", "Vote", "corroborated_votes"]

_EPS = 1e-12


@dataclass(frozen=True)
class Vote:
    """一次观测折算出的一票。"""

    label: str
    weight: float


def corroborated_votes(
    car_class: str | None,
    digit: str | None,
    *,
    agree_bonus: float = 0.3,
    disagree_penalty: float = 0.6,
    conflict_hint_weight: float = 0.15,
) -> list[Vote]:
    """把一次观测的证据折成票。

    ``car_class`` 形如 ``"B1"``（阵营 + 角色），``digit`` 是数字分类器的结果。
    两者都可能为 ``None``/空（没读到）。
    """
    car = (car_class or "").strip()
    dig = (digit or "").strip()

    if not car:
        # 数字单独出现时指不出槽位：阵营不在数字里。硬猜一半概率是错的，
        # 而身份错一次要好几秒才能纠回来。
        return []
    if len(car) < 2:
        return []

    side, role = car[0], car[1:]
    if not dig:
        return [Vote(car, 1.0)]

    if dig == role:
        return [Vote(car, 1.0 + agree_bonus)]

    # 矛盾：车类别仍然主导（它看的像素多得多），但要挨一记折扣；
    # 同时给数字指出的槽位记一小票，让它反复出现时能积累翻盘。
    weakened = max(0.1, 1.0 - disagree_penalty)
    return [Vote(car, weakened), Vote(f"{side}{dig}", conflict_hint_weight)]


def armor_votes(evidence) -> list[Vote]:
    """把同一帧多个读数组合成一组票，矛盾和小图都会降低单一身份的权重。"""
    scores: dict[str, float] = {}
    for item in evidence:
        side = {"red": "R", "blue": "B"}.get(item.color.lower())
        digit = item.digit.upper()
        if digit == "S":
            digit = "7"
        if not side or not digit:
            continue
        size = min(1.0, math.sqrt(max(item.width_px, 0.0) * max(item.height_px, 0.0)) / 25.0)
        score = min(max(item.armor_confidence, 0.0), 1.0, max(item.digit_confidence, 0.0)) * size
        label = side + digit
        scores[label] = max(scores.get(label, 0.0), score)
    total = sum(scores.values())
    if total <= 0.0:
        return []
    strongest = max(scores.values())
    return [Vote(label, score / total * strongest) for label, score in scores.items()]


@dataclass
class IdentityVote:
    """带指数衰减的标签票箱，**且带切换粘滞**。

    半衰期只在 ``add``/``decay_to`` 被调用时才结算，不做定时任务——
    轨迹本来每帧都会被喂一次，采样点足够密。

    ## 为什么还要粘滞

    假设检测器在 B1 / B2 之间反复横跳（远处目标、颜色相近，这是常态）。
    每帧各投一票的话，票权会变成 ``B1=2.90, B2=1.93`` → 下一帧
    ``B1=2.85, B2=2.90``，**领跑者每帧都换**，身份就跟着每帧翻。

    对外表现是雷达站在 B1 和 B2 之间反复报人，比一开始就不确定还糟——
    裁判侧看到的是一个不断改口的来源。所以换标签要**超出在职者一个幅度**
    (``switch_margin``) 才允许。真发生换人时票权会持续累积、很快越过这个
    幅度；而噪声造成的抖动越不过去。
    """

    half_life_s: float = 2.0
    switch_margin: float = 0.15
    """挑战者要超出在职者这么多比例才允许换标签。"""
    _weights: dict[str, float] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)
    """首次出现的顺序。平局时按它裁决，保证回放可复现。"""
    _current: str | None = None
    _last_ts: float | None = None

    # -- 写入 -------------------------------------------------------------
    def add(self, label: str, weight: float = 1.0, *, now: float) -> None:
        """加一票。weight ≤ 0 的票直接丢弃（负票不在本设计里）。"""
        if weight <= 0.0 or not label:
            return
        self.decay_to(now)
        if label not in self._weights:
            self._weights[label] = 0.0
            self._order.append(label)
        self._weights[label] += float(weight)
        self._retie()

    def add_votes(self, votes, *, now: float) -> None:
        for v in votes:
            self.add(v.label, v.weight, now=now)

    def decay_to(self, now: float) -> None:
        """把所有票权衰减到 ``now`` 时刻。"""
        if self._last_ts is None:
            self._last_ts = float(now)
            return
        dt = float(now) - self._last_ts
        if dt <= 0.0:
            # 时间没走（同刻多次调用）或倒退：不衰减，也不推进基准时刻。
            # 倒退时推进基准会让后面的 dt 变小、衰减偏慢，等于让坏时间戳
            # 悄悄削弱衰减的作用。
            return
        self._last_ts = float(now)
        factor = 0.5 ** (dt / max(float(self.half_life_s), _EPS))
        if factor >= 1.0:
            return
        for k in list(self._weights):
            self._weights[k] *= factor
        self._retie()

    # -- 裁决 -------------------------------------------------------------
    @property
    def leader(self) -> tuple[str, float] | None:
        """**当下**票权最高的标签，不看粘滞。诊断与测试用。

        平局按**首次出现顺序**裁决（不是字典序，也不是集合序）：先看到的先赢，
        符合"证据随时间累积"的直觉，而且完全确定。
        """
        if not self._weights:
            return None
        top = max(self._weights.values())
        if top <= 0.0:
            return None
        for label in self._order:
            if self._weights[label] >= top:
                return label, float(self._weights[label])
        return None  # pragma: no cover - max 保证至少一个达到 top

    @property
    def best(self) -> tuple[str, float] | None:
        """对外使用的标签，**带粘滞**。"""
        if self._current is None or self._current not in self._weights:
            return None
        return self._current, float(self._weights[self._current])

    def _retie(self) -> None:
        """按票权与粘滞规则决定在职标签。"""
        lead = self.leader
        if lead is None:
            return
        if self._current is None or self._current not in self._weights:
            # 首次定身份：直接采纳领跑者。这是唯一不需要越过幅度的时刻，
            # 否则一条新轨迹要等好几帧才敢说自己是谁。
            self._current = lead[0]
            return
        if lead[0] == self._current:
            return
        incumbent = self._weights[self._current]
        if lead[1] > incumbent * (1.0 + max(float(self.switch_margin), 0.0)):
            self._current = lead[0]

    @property
    def total(self) -> float:
        """票权总和。可用作身份置信度的分母。"""
        return float(sum(self._weights.values()))

    def share(self, label: str) -> float:
        """某标签占总票权的比例，``[0,1]``。"""
        total = self.total
        if total <= 0.0:
            return 0.0
        return float(self._weights.get(label, 0.0) / total)

    def snapshot(self) -> dict[str, float]:
        """当前票权快照，供诊断输出。"""
        return {k: float(v) for k, v in self._weights.items()}

    def __len__(self) -> int:
        return len(self._weights)
