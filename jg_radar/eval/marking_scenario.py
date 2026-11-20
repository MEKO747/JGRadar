"""断流时「发旧值」还是「发 (0,0)」：同一段输入跑出两条 P(t)。

## 为什么要有这个模块

`0x0305` 里 ``x=y=0`` 等于「未发送此机器人坐标」（协议表 1-36）。于是雷达断流时
发送端有两条路：

- ``drop``：写 ``(0,0)``。服务器看不到数据 → 每累计 0.5 秒记一次「错误」，
  按规则表 5-29，连续的「错误/中断」会让 x 继续往下掉（−0.8 起）。
- ``send_stale``：把最后一次坐标照发。服务器照常按 d 分级判定，5Hz 就是 5Hz 的
  判定频率——**发得越勤，判错的代价也翻倍**。

哪个更好完全取决于"断流期间旧坐标还准不准"，这不是拍脑袋能定的，得算：

- 旧坐标仍然 d<0.8m（目标基本没动）：``send_stale`` 继续 x+1.0 累积，P 还在涨。
- 0.8≤d<1.6m：半准确，x 继续 +0.5 累积，**还在涨**。
- d≥1.6m：每次 −0.8，4.7Hz 下大约每秒掉 3.76，而 ``drop`` 只掉 1.6/秒。

所以分界线在 **1.6m**，不是"新旧哪个看起来更合理"。

## 输入只有一份

``Scenario.measurements()`` 产出**唯一**一份观测序列（目标位置、噪声、断流窗口），
两条策略吃的是同一份——唯一差别是断流那几帧怎么填。这样才叫"同一段录制"。
真实链路的帧日志走 :func:`run_from_frame_times`，接真机时不用改模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import random

from .marking_progress import (
    HALF_MAX_M,
    INTERRUPT_PERIOD_S,
    Judgement,
    MarkingTracker,
    Sample,
)


class StalePolicy(Enum):
    """断流时发送端的行为，对应 ``referee_tx.stale_policy`` 的两个值。"""

    DROP = "drop"
    SEND_STALE = "send_stale"


@dataclass(frozen=True)
class Scenario:
    """一个可复算的断流场景。

    目标沿 x 轴匀速直线运动，速度是**场景假设**（不是规则数据：规则手册只给底盘
    功率上限 50W，没有给平移速度）。所以结论一律按"断流期间漂移了多少米"表述，
    速度只是把漂移算出来的中间量。
    """

    send_hz: float = 4.7
    horizon_s: float = 12.0
    #: 断流起点默认落在**爬坡期**（1.2s 时 P≈21）。放在 3.0s 会被 150 的上限
    #: 顶住——两条策略都撞顶，比出来的差值全是 0，等于什么都没测。
    outage_start_s: float = 1.2
    outage_s: float = 1.0
    speed_mps: float = 2.0
    #: 端到端延迟（采集→融合→串口发出）。4.7Hz 下 200ms 是保守值。
    latency_s: float = 0.2
    noise_m: float = 0.10
    seed: int = 7

    def __post_init__(self) -> None:
        if self.send_hz <= 0:
            raise ValueError("send_hz 必须为正")
        if self.horizon_s <= 0:
            raise ValueError("horizon_s 必须为正")
        if self.outage_s < 0 or self.outage_start_s < 0:
            raise ValueError("断流窗口不能为负")
        if self.speed_mps < 0:
            raise ValueError("速度不能为负")
        if not math.isfinite(self.latency_s) or self.latency_s < 0:
            raise ValueError("latency_s 必须是非负有限值")

    @property
    def period_s(self) -> float:
        return 1.0 / self.send_hz

    @property
    def drift_m(self) -> float:
        """断流期间目标实际走了多远——结论要按这个量说，而不是按速度。"""
        return self.speed_mps * min(self.outage_s, self.horizon_s)

    def truth(self, t: float) -> tuple[float, float]:
        """目标真实平面坐标（米），沿 x 轴匀速。"""
        return (self.speed_mps * max(t, 0.0), 0.0)

    def radio_available(self, t: float) -> bool:
        return not (self.outage_start_s <= t < self.outage_start_s + self.outage_s)

    def frame_times(self) -> list[float]:
        count = int(self.horizon_s * self.send_hz) + 1
        return [index * self.period_s for index in range(count)]

    def measurements(self) -> list[tuple[float, tuple[float, float] | None]]:
        """唯一一份观测序列：``(发送时刻, 雷达估计坐标或 None)``。

        ``None`` 表示这一刻无线电压根没有新数据（断流期间）。噪声用固定种子，
        两条策略拿到的是同一串数字。
        """
        rng = random.Random(self.seed)
        out: list[tuple[float, tuple[float, float] | None]] = []
        for t in self.frame_times():
            if not self.radio_available(t):
                out.append((t, None))
                continue
            delayed = max(t - self.latency_s, 0.0)
            tx, ty = self.truth(delayed)
            out.append((t, (tx + rng.gauss(0.0, self.noise_m), ty + rng.gauss(0.0, self.noise_m))))
        return out


def distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compare_time(scenario: Scenario) -> float:
    """两条策略的对比时刻：断流结束后 0.5 秒。

    不能在"断流结束那一刻"比：``drop`` 在那之前就没样本了，它最后一条还是
    断流前的 P，而 ``send_stale`` 一直有样本——那样比出来的是"谁的样本更新"，
    不是"谁伤得少"。推后 0.5 秒，两边都已恢复数据。

    必须由**同一个场景**算出来：无断流基准的 ``outage_s`` 是 0，用它自己的
    公式会算出更早的时刻，两边的 P 就没法比。
    """
    return scenario.outage_start_s + scenario.outage_s + 0.5


@dataclass
class RunResult:
    """一条策略跑完的结果。"""

    policy: StalePolicy
    scenario: Scenario
    samples: list[Sample] = field(default_factory=list)

    @property
    def p_end(self) -> float:
        return self.samples[-1].p if self.samples else 0.0

    @property
    def p_max(self) -> float:
        return max((sample.p for sample in self.samples), default=0.0)

    @property
    def p_min(self) -> float:
        return min((sample.p for sample in self.samples), default=0.0)

    def p_at(self, t: float) -> float:
        """``t`` 时刻的 P（取不晚于 ``t`` 的最后一个样本）。"""
        value = 0.0
        for sample in self.samples:
            if sample.t > t + 1e-9:
                break
            value = sample.p
        return value

    def first_time_at_after(self, threshold: float, t0: float) -> float | None:
        """从 ``t0`` 之后第一次到达 ``threshold`` 的时刻——断流后的恢复时间。"""
        for sample in self.samples:
            if sample.t >= t0 - 1e-9 and sample.p >= threshold:
                return sample.t
        return None

    def first_time_at(self, threshold: float) -> float | None:
        for sample in self.samples:
            if sample.p >= threshold:
                return sample.t
        return None

    def counts(self) -> dict[str, int]:
        counts = {judgement.value: 0 for judgement in Judgement}
        for sample in self.samples:
            if sample.judgement is not None:
                counts[sample.judgement.value] += 1
        return counts

    def series(self) -> list[tuple[float, float]]:
        return [(sample.t, sample.p) for sample in self.samples]

    @property
    def compare_time(self) -> float:
        return compare_time(self.scenario)

    def p_at_compare(self) -> float:
        return self.p_at(self.compare_time)

    def summary(self) -> dict[str, object]:
        counts = self.counts()
        return {
            "p_at_compare": round(self.p_at_compare(), 3),
            "policy": self.policy.value,
            "outage_s": self.scenario.outage_s,
            "speed_mps": self.scenario.speed_mps,
            "drift_m": round(self.scenario.drift_m, 3),
            "p_end": round(self.p_end, 3),
            "p_max": round(self.p_max, 3),
            "p_min": round(self.p_min, 3),
            "first_100_s": (None if self.first_time_at(100.0) is None
                            else round(self.first_time_at(100.0), 3)),
            "first_120_s": (None if self.first_time_at(120.0) is None
                            else round(self.first_time_at(120.0), 3)),
            "account_js": counts[Judgement.ACCURATE.value],
            "half": counts[Judgement.HALF.value],
            "error": counts[Judgement.ERROR.value],
            "interrupt": counts[Judgement.INTERRUPT.value],
        }


def run(scenario: Scenario, policy: StalePolicy) -> RunResult:
    """按策略把观测序列喂给 P 递推。两条策略只在这里分叉。"""
    tracker = MarkingTracker()
    last_estimate: tuple[float, float] | None = None
    for t, estimate in scenario.measurements():
        if estimate is not None:
            last_estimate = estimate
            tracker.data(t, distance_m(estimate, scenario.truth(t)))
            continue
        if policy is StalePolicy.SEND_STALE and last_estimate is not None:
            tracker.data(t, distance_m(last_estimate, scenario.truth(t)))
        else:
            # 没发送 = 服务器侧「数据中断」，按累计 0.5 秒补错误判定
            tracker.silence_until(t)
    tracker.silence_until(scenario.horizon_s)
    return RunResult(policy=policy, scenario=scenario, samples=list(tracker.samples))


def run_ideal(scenario: Scenario) -> RunResult:
    """无断流基准：同一份观测，没有任何掉帧。"""
    ideal = Scenario(
        send_hz=scenario.send_hz,
        horizon_s=scenario.horizon_s,
        outage_start_s=scenario.outage_start_s,
        outage_s=0.0,
        speed_mps=scenario.speed_mps,
        latency_s=scenario.latency_s,
        noise_m=scenario.noise_m,
        seed=scenario.seed,
    )
    return run(ideal, StalePolicy.SEND_STALE)


def compare(scenario: Scenario) -> dict[StalePolicy, RunResult]:
    return {policy: run(scenario, policy) for policy in StalePolicy}


def first_divergence(result: RunResult, horizon_s: float | None = None) -> float | None:
    """第一条判错/中断出现的时刻。诊断用：看清断流到底从哪一秒开始伤人。"""
    for sample in result.samples:
        if sample.judgement is None:
            continue
        if sample.judgement.is_error:
            return sample.t
    return None


__all__ = [
    "HALF_MAX_M",
    "INTERRUPT_PERIOD_S",
    "RunResult",
    "Scenario",
    "StalePolicy",
    "compare",
    "distance_m",
    "first_divergence",
    "run",
    "run_ideal",
]
