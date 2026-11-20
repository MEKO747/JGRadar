"""2027 规则手册 §5.7.4「被标记进度」P 的递推模型。

模型**逐字**来自《RoboMaster 2027 机甲大师超级对抗赛比赛规则手册 V1.0.0》
（20261015）表 5-29，没有任何自拟参数：

| 标记状态 | x 的更新 | P 的更新 |
| --- | --- | --- |
| 准确（d<0.8m） | 上一次准确/半准确 → ``x=x+1.0``；否则 ``x=1.0`` | ``P=P+x`` |
| 半准确（0.8m≤d<1.6m） | 上一次准确/半准确 → ``x=x+0.5``；否则 ``x=0.5`` | ``P=P+x`` |
| 错误（d≥1.6m） | 上一次错误/中断 → ``x=x−0.8``；否则 ``x=−0.8`` | ``P=P+x`` |
| 数据中断 | 裁判系统**每累计 0.5 秒**未收到数据，视为一次「错误」判定 | 同上 |

规则原文的限幅：「若计算结果 P<0，则 P 取 0；若计算结果 P>150，则 P 取 150」。
判定频率由雷达发送频率决定——所以我们 4.7Hz 发，就是 4.7Hz 判。

规则文档里那段 5Hz/100%→0%→断流 1 秒的示例，是这套实现唯一的验收标准：
`tests/test_marking_progress.py` 把示例里每一个 P 值都钉死了。任何"看着差不多"
的改动都会在那条测试上翻车。

这个模块不做平滑、不做加权、不做"更合理"的修正。规则怎么写，这里怎么算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

#: 准确上界：d < 0.8m。
ACCURATE_MAX_M = 0.8
#: 半准确上界：0.8m ≤ d < 1.6m；d ≥ 1.6m 为错误。
HALF_MAX_M = 1.6
#: P 的取值范围（规则原文 0~150）。
P_MIN = 0.0
P_MAX = 150.0
#: 数据中断判定的累计周期，秒。
INTERRUPT_PERIOD_S = 0.5
#: 比较时间戳时的容差。计时是按 ``最后一条数据 + n×0.5`` 累加的，浮点误差
#: 会让"正好 1.0 秒的断流"少算一次中断——规则里 5.0 就是 5.0，不能由 1e-16 决定。
_TIME_EPS = 1e-9


class Judgement(Enum):
    """一次判定的四种结果。

    ``INTERRUPT`` 和 ``ERROR`` 在 x/P 的递推里是同一个分支（都算"错误"），
    但必须分开记：下一次判定要区分"上一次是中断"还是"上一次是错误"没有影响，
    可日志和论文里要能看出这次是**没发**还是**发错了**。
    """

    ACCURATE = "accurate"
    HALF = "half"
    ERROR = "error"
    INTERRUPT = "interrupt"

    @property
    def is_error(self) -> bool:
        return self is Judgement.ERROR or self is Judgement.INTERRUPT


def classify_distance(distance_m: float) -> Judgement:
    """按 d 分级。``distance_m`` 是米，负数或 NaN 直接报错，不做兜底。"""
    if not math.isfinite(distance_m):
        raise ValueError(f"识别误差必须是有限值，收到 {distance_m!r}")
    if distance_m < 0:
        raise ValueError(f"识别误差不能为负，收到 {distance_m!r}")
    if distance_m < ACCURATE_MAX_M:
        return Judgement.ACCURATE
    if distance_m < HALF_MAX_M:
        return Judgement.HALF
    return Judgement.ERROR


@dataclass
class ProgressState:
    """P 与 x 的状态，只实现表 5-29 那一格。

    ``last`` 是上一次判定；``None`` 表示还没有过判定（首帧按"否则"分支走）。
    """

    p: float = P_MIN
    x: float = 0.0
    last: Judgement | None = None

    def apply(self, judgement: Judgement) -> float:
        """更新 x 再更新 P，返回新的 P。顺序按规则原文「先更新 x，后更新 P」。"""
        if judgement is Judgement.ACCURATE:
            self.x = self.x + 1.0 if self.last in (Judgement.ACCURATE, Judgement.HALF) else 1.0
        elif judgement is Judgement.HALF:
            self.x = self.x + 0.5 if self.last in (Judgement.ACCURATE, Judgement.HALF) else 0.5
        else:
            self.x = self.x - 0.8 if self.last is not None and self.last.is_error else -0.8
        self.p = min(max(self.p + self.x, P_MIN), P_MAX)
        self.last = judgement
        return self.p


@dataclass(frozen=True)
class Sample:
    """某一时刻的 P，附带这一时刻发生了什么判定。"""

    t: float
    p: float
    judgement: Judgement | None = None
    distance_m: float | None = None


@dataclass
class MarkingTracker:
    """把时间轴喂进 P 递推的驱动器。

    两件事：
    1. 收到一条坐标 → 算 d、分级、判定；
    2. 没收到坐标 → 每累计 0.5 秒补一次中断判定。

    中断计时以**最后一条数据**为起点，用绝对时间戳推进，不依赖调用频率——
    否则主循环抖动会把 0.5 秒算歪。
    """

    state: ProgressState = field(default_factory=ProgressState)
    interrupt_period_s: float = INTERRUPT_PERIOD_S
    samples: list[Sample] = field(default_factory=list)
    _last_data_t: float | None = None
    _next_interrupt_t: float | None = None

    def data(self, t: float, distance_m: float) -> list[Sample]:
        """收到一条本车坐标并判定。返回这段时间里因中断补出的样本（可能为空）。"""
        emitted = self._advance(t, inclusive=False)
        judgement = classify_distance(distance_m)
        self.state.apply(judgement)
        sample = Sample(t=t, p=self.state.p, judgement=judgement, distance_m=distance_m)
        self.samples.append(sample)
        emitted.append(sample)
        self._last_data_t = t
        self._next_interrupt_t = t + self.interrupt_period_s
        return emitted

    def silence_until(self, t: float) -> list[Sample]:
        """推进到 ``t``，只补中断，不产生数据判定。"""
        return self._advance(t, inclusive=True)

    def _advance(self, t: float, *, inclusive: bool) -> list[Sample]:
        emitted: list[Sample] = []
        if self._next_interrupt_t is None:
            return emitted
        while self._next_interrupt_t < t - _TIME_EPS or (
            inclusive and self._next_interrupt_t <= t + _TIME_EPS
        ):
            self.state.apply(Judgement.INTERRUPT)
            sample = Sample(t=self._next_interrupt_t, p=self.state.p, judgement=Judgement.INTERRUPT)
            self.samples.append(sample)
            emitted.append(sample)
            self._next_interrupt_t += self.interrupt_period_s
        return emitted

    @property
    def p(self) -> float:
        return self.state.p

    def first_time_at(self, threshold: float) -> float | None:
        """P 第一次达到 ``threshold`` 的时刻，没到就返回 None。"""
        for sample in self.samples:
            if sample.p >= threshold:
                return sample.t
        return None
