"""源仲裁：优先级 + 滞回 + 跳变抑制。**"丝滑"的全部实现都在这里。**

## 问题

无线电和视觉都在报同一个机器人的坐标。它们会不一致——不是"一个对一个错"，
而是两个都在动、都有噪声、都有延迟。每帧挑"更好的那个"会产生一个致命的
副作用：**坐标在两条轨迹之间来回跳**。

裁判侧的规则是"连续正确识别 3 秒"。一个每秒跳两次的坐标，即使两个源各自
都挺准，也永远凑不满 3 秒——所以**跳变比误差更致命**。这就是方案 §Context
第 2 条说的"稳定性 > 精度"。

## 三条规则

1. **现任保持**：已经在用的源默认继续用。换源本身是有代价的，不是免费的。
2. **挑战要走完驻留**：优先级更高的源必须**连续**满足条件达 ``dwell_s``
   (300ms) 才允许接管。单帧的优先级翻转不构成理由。
3. **跳变抑制**：如果挑战者的位置和现任差得超过 ``switch_jump_cm`` (150cm)，
   说明这一换会是一次肉眼可见的瞬移，于是门槛抬高到 ``reacquire_s`` (800ms)，
   并且要求挑战者在这段时间里的位置**自己也是稳定的**（一致样本）。

第 3 条防的是一类很隐蔽的情况：挑战者自己正在乱跳。如果只看"它优先级高、
它存在了 300ms"，一个正在乱跳的源也能接管，接手之后继续乱跳。

## 例外一：现任**源**还在，只是换了档 —— 不走滞回

滞回管的是**换源**，不是换档。

视觉从 ``MEASURED`` 掉到 ``OCCLUSION_HOLD``、再掉到 ``BLIND_PREDICTION``，
是**同一个源**在降级。这条链已经在跟踪层的状态机里平滑过了（``lost_predict_s``
把"刚丢"和"丢久了"分开），仲裁层再叠一层 300ms 驻留，得到的不是平滑，
而是**每降一级就静默 300ms**：

    MEASURED, MEASURED, [空], [空], [空], [空], [空], [空], HOLD, HOLD ...

对外看就是坐标**先消失、再以预测态回来**——一次本该丝滑的降级变成了一个
断崖。恢复（``HOLD→MEASURED``）同理，更不该让裁判侧多等 300ms 才承认
"又看到了"。

所以判定"现任"用的是**源**（``SourceKind``），不是档位名。只要现任那个源
这一帧还在候选里，它换什么档就立刻采纳什么档。

## 例外二：现任**源**整个消失 —— 也不走滞回

如果现任那一档这一帧根本不在了（无线电断流超过 ``radio_stale_s``、视觉的
数据不再产出），就不该继续抱着它。此时直接采纳当前最优候选，不走滞回：
降级链已经在跟踪层做完了平滑，这里再叠一层延迟没有意义，只会让裁判侧多等
一会儿才知道"没数据了"。

**但有一个必须拦住的情形**：现任源塌掉的同时，缓冲里可能躺着一条**优先级
更高、却从没参与过竞争**的数据。见 ``_decide_one`` 里那段注释——它描述
的正是这里不能无条件采纳的原因。

## 第二条轴：哪个坐标更准

上面那三条规则回答的都是"这份证据是**哪一种**"——无线电的新鲜报文、视觉的
实测、视觉的预测。阶梯本身**不回答"哪个坐标更准"**，而这是两个不同的问题：

- ``radio_stale`` 在阶梯上排在 ``vision_occlusion_hold`` 之上。可是"陈旧"
  的区间是 0.2 秒到 **10 秒**；一条 9 秒前的报文，按 5m/s 算最多可能偏出
  45m——比整个场地（28m × 15m）还长。阶梯说"优先用它"，而它其实一文不值。
- 阶梯的排序编码的是一个**典型**假设（无线电的计量比视觉好），不是一个
  逐样本的判断。

所以加一条与阶梯**正交**的精度轴：每个候选算一个 ``position_error_bound_cm``
（见该函数的文档），回答"这条坐标现在最多可能偏多少"。它用在两处：

1. **可用性否决**：上界超过 ``max_position_error_cm`` 的候选**直接不作数**，
   连挑战者都当不上。默认阈值下这等价于"实测坐标的保质期是 1 秒"。
2. **同档位内的次序**：同档位取上界更小的那个。今天两个源不共档位，所以这
   一条还轮不到；但 ``Source`` 协议本来就是留给"再加一个源"的，到时候它决定
   选谁——而不是"序列里谁写在前面就听谁的"（``min`` 是稳定的，没有这个键
   就等于按入参顺序决胜，而那是个没人设计过的规则）。

**刻意不做**的一件事：不让精度轴去**推翻**阶梯。一条 0.9 秒前的无线电
（上界 450cm）和一条盲区预测（3σ ≈ 80cm）摆在一起时，数字会说盲区更准——
但那是拿一个**硬上界**去比一个**猜测的 3σ**：前者是"最多偏这么多"，一个
关于**误差**的、几乎不可能被超过的界；后者是滤波器对自己外推的自信，
说的是**估计**有多散。用后者赢前者，等于让猜测盖过测量。
阶梯管"是哪一种证据"，精度轴管"这一种证据现在还值不值得报"，各管一头。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from jg_radar.fuse.contract import StampedSample
from jg_radar.types import RobotState, SourceKind, SourceSample

__all__ = [
    "Arbiter",
    "KNOWN_TIERS",
    "PREDICTION_SIGMA",
    "SlotDecision",
    "classify",
    "position_error_bound_cm",
    "radio_tier",
    "vision_tier",
]

#: 把预测态的 ``position_std_cm``（1σ）放大成与实测态同一口径的上界。
#:
#: 实测态的上界是"最多可能偏这么多"（速度 × 时间，一个硬上界）；1σ 不是
#: 同一个口径——它只有 68% 的覆盖。要放在一起比、要共用一个阈值，就得先
#: 把预测态也抬到一个"基本不可能超过"的位置，所以取 3σ（99.7%）。
#:
#: 这个数**不是**标定出来的，也不需要是：它只影响"预测态的上界算出来多大"，
#: 而阈值 ``max_position_error_cm`` 与它之间有 6 倍以上的余量（回放 fixture
#: 实测盲区预测 3σ ≤ 79cm，阈值 500cm）。真正决定成败的是那个阈值，见
#: ``tests/test_arbiter_accuracy.py``。
PREDICTION_SIGMA = 3.0

# 档位名与 ``FuseConfig.priority`` 里的字符串一一对应
TIER_RADIO_FRESH = "radio_fresh"
TIER_RADIO_STALE = "radio_stale"
TIER_VISION_MEASURED = "vision_measured"
TIER_VISION_HOLD = "vision_occlusion_hold"
TIER_VISION_BLIND = "vision_blind"

VISION_TIERS_BY_STATE = {
    RobotState.MEASURED: TIER_VISION_MEASURED,
    RobotState.OCCLUSION_HOLD: TIER_VISION_HOLD,
    RobotState.BLIND_PREDICTION: TIER_VISION_BLIND,
}

KNOWN_TIERS = (
    TIER_RADIO_FRESH,
    TIER_RADIO_STALE,
    TIER_VISION_MEASURED,
    TIER_VISION_HOLD,
    TIER_VISION_BLIND,
)


def vision_tier(stamped: StampedSample, *, now: float, watchdog_s: float) -> str | None:
    """视觉样本落在哪一档。``None`` = 不是合法候选。

    看门狗量的是 ``received_at``（源还活着吗），不是 ``observed_at``——
    预测态的 ``observed_at`` 本来就该很旧，那是**设计**，不是故障。
    """
    if stamped.liveness_age(now) > float(watchdog_s):
        return None
    return VISION_TIERS_BY_STATE.get(stamped.sample.state)


def radio_tier(
    stamped: StampedSample, *, now: float, fresh_s: float, stale_s: float
) -> str | None:
    """无线电样本落在哪一档。

    这里量的就是 ``observed_at``：无线电的"新鲜"指的是**空口上刚收到**，
    不是"我刚从缓冲里读到"。缓冲里躺着一条 5 秒前的无线电报，它依然是陈旧的。
    """
    age = stamped.information_age(now)
    if age <= float(fresh_s):
        return TIER_RADIO_FRESH
    if age <= float(stale_s):
        return TIER_RADIO_STALE
    return None


def position_error_bound_cm(sample: SourceSample, *, now: float, cfg) -> float:
    """这条坐标**现在**最多可能偏多少（cm）。``inf`` = 没有坐标，无从谈起。

    两条分支问的是同一个问题——"我们马上要报出去的这个位置，和机器人真正
    在哪，最远能差多少"——所以它们的数**可以放在一起比**：

    **实测态（``is_predicted=False``）**：``信息年龄 × max_robot_speed_cm_s``。

    信息是 ``observed_at`` 那一刻测到的，从那时到现在机器人最多移动了
    ``速度 × 时间``，所以这是"当前位置"的一个**硬上界**。它**只用时间**，
    因此天然跨源可比——这正好绕开了 ``position_std_cm`` 那个"视觉给协方差、
    无线电给链路估计，两者不同口径"的老问题。

    **预测态（``is_predicted=True``）**：``PREDICTION_SIGMA × position_std_cm``。

    预测态的 ``observed_at`` 刻意冻在最后一次真观测上，拿它算年龄会得到
    "9 秒前"这种数——那是**设计**，不是故障，用它算上界会把所有盲区预测
    一律否掉，而盲区预测存在的理由就是在这段时间里继续报数。
    正确的量是滤波器自己外推出来的协方差：预测步已经把时间折进 ``P`` 里了，
    所以 ``position_std_cm`` 同样是"对**当前**位置的估计"。

    预测态**没有** ``position_std_cm`` 时退回时间上界。那不是保守，是诚实：
    一条声称是预测、却给不出任何不确定度的样本，我们没有任何依据说它准，
    于是它会因为年龄太大被否掉——槽位转为 MISSING，``vetoed`` 计数器涨一格，
    在诊断里看得见。
    """
    if sample.position_cm is None:
        return float("inf")
    if not math.isfinite(sample.observed_at):
        return float("inf")
    if sample.is_predicted and sample.position_std_cm is not None:
        std = float(sample.position_std_cm)
        offset = float(sample.prediction_offset_cm)
        if not math.isfinite(std) or not math.isfinite(offset) or min(std, offset) < 0:
            return float("inf")
        if sample.estimated_at is not None and not math.isfinite(sample.estimated_at):
            return float("inf")
        age = 0.0 if sample.estimated_at is None else max(0.0, now - sample.estimated_at)
        return PREDICTION_SIGMA * std + offset + age * float(cfg.max_robot_speed_cm_s)
    age = max(0.0, float(now) - float(sample.observed_at))
    return age * float(cfg.max_robot_speed_cm_s)


def classify(stamped: StampedSample, *, now: float, cfg) -> str | None:
    """按来源分派到档位。未知来源、MISSING、超新鲜度范围一律 ``None``——**不猜**。

    它只回答"这是**哪一种**证据"，**不回答"它现在还准不准"**。后者是
    :func:`position_error_bound_cm` 的事，闸门开在 ``Arbiter._admit``。

    两道闸门刻意不都放在这里：否决必须**计数**（``Arbiter.vetoed``），
    而这是个自由函数，没有地方放那个计数器；分散在两处则更糟——谁是准入
    的权威会变得说不清，而这里恰恰是最不能含糊的地方。
    """
    kind = stamped.sample.source
    if kind is SourceKind.VISION:
        return vision_tier(stamped, now=now, watchdog_s=cfg.vision_watchdog_s)
    if kind is SourceKind.RADIO:
        return radio_tier(
            stamped, now=now, fresh_s=cfg.radio_fresh_s, stale_s=cfg.radio_stale_s
        )
    return None


@dataclass(frozen=True)
class SlotDecision:
    """一个槽位本帧的仲裁结果。"""

    name: str
    sample: SourceSample | None
    tier: str | None
    """采纳的档位名；``None`` 表示没有可用候选。"""
    switched: bool
    """本帧是否发生了**档位变化**（同一个源内的降级也算）。"""
    previous_tier: str | None = None
    """上一帧的档位。诊断用：``radio_fresh→radio_stale`` 是断流，
    ``radio_*→vision_*`` 才是换源。两者都会置 ``switched``，
    但只有后者可能带来肉眼可见的坐标跳变。"""
    error_bound_cm: float | None = None
    """采纳的这条坐标最多可能偏多少（cm）。见 ``position_error_bound_cm``。

    放进决策里是为了让"哪个坐标更准"这件事**看得见**：日志里一行
    ``R1 vision_occlusion_hold ±78cm`` 和一行 ``R1 vision_occlusion_hold
    ±1200cm`` 说的是完全不同的两件事，而档位名是一样的。``None`` = 本帧
    没有采纳任何东西。

    排在最后、带默认值：``SlotDecision`` 已经被测试按位置构造过，
    往中间插字段会静默地改变那些调用的含义。
    """

    @property
    def empty(self) -> bool:
        return self.sample is None


_Candidate = tuple[str, float, StampedSample]
"""一个候选：``(档位名, 位置误差上界 cm, 样本)``。

上界跟着候选一起走，而不是等要用的时候再算一遍——``_admit`` 每帧已经算过，
选择要用它当第二排序键，采纳时还要写进 ``SlotDecision``。三处都算一次的话，
任何一处的 ``now`` 传错都会让"选出来的"和"报出去的"不是同一个数。
"""


@dataclass
class _SlotState:
    tier: str | None = None
    source: SourceKind | None = None
    """现任是**哪个源**在供数。

    判定"现任还在不在"要用它，不能用档位名。视觉从 ``MEASURED`` 掉到
    ``OCCLUSION_HOLD`` 时档位名变了，但源没有变——拿档位名去认现任，会把
    一次降级误判成"现任没了"，于是每降一级都白等 300ms 驻留。
    """
    position: tuple[float, float] | None = None
    challenger: str | None = None
    challenger_since: float = float("-inf")
    challenger_anchor: tuple[float, float] | None = None


class Arbiter:
    """逐槽位的源仲裁器。**有状态**——滞回必须记得住历史。"""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.priority: tuple[str, ...] = tuple(cfg.priority)
        unknown = sorted(set(self.priority) - set(KNOWN_TIERS))
        if unknown:
            # 档位名写错会**静默地**让某一档永远轮不到，比启动失败危险得多
            raise ValueError(
                f"fuse.priority 里有未知档位 {unknown}；"
                f"可用档位 = {list(KNOWN_TIERS)}"
            )
        self._rank = {tier: i for i, tier in enumerate(self.priority)}
        self._floor = len(self.priority)
        self._slots: dict[str, _SlotState] = {}
        self.switch_count = 0
        self.vetoed = 0
        """因为位置误差上界过大而被否掉的候选数（累加，不是每帧的快照）。

        它是"有一路数据在源源不断地报着，但我们一条都没发出去"的唯一外部
        征兆。没有这个计数，那种情况看起来和"那个源压根没数据"一模一样——
        而两者的处置完全不同（前者要去看标定/延迟，后者要看那个源）。
        """

    # -- 主入口 -----------------------------------------------------------
    def decide(
        self, samples: Iterable[StampedSample], *, now: float
    ) -> dict[str, SlotDecision]:
        """给出每个槽位的仲裁结果。

        只对**有候选或有历史**的槽位产生条目；完全没听说过的槽位不出现在结果里，
        由发布层统一填 MISSING。

        入参是一个**序列**，不是一个按名字索引的字典：同一个机器人本来就会有
        多条样本（每个源一条），用字典表达会诱导调用方以为一个名字只有一个值
        ——那正是"后写的把先写的覆盖掉"这个 bug 的入口。见 ``LatestValueBuffer``。
        """
        grouped: dict[str, list[_Candidate]] = {}
        for st in samples:
            admitted = self._admit(st, now=now)
            if admitted is None:
                continue
            tier, bound = admitted
            grouped.setdefault(st.name, []).append((tier, bound, st))

        return {
            name: self._decide_one(name, grouped.get(name, ()), now=now)
            for name in set(grouped) | set(self._slots)
        }

    def _admit(self, stamped: StampedSample, *, now: float) -> tuple[str, float] | None:
        """**候选的准入闸门。** ``(档位, 误差上界)``；不做候选则 ``None``。

        两道闸门，顺序不能反：先分派得出档位（``classify``），再看位置误差
        上界过不过 ``max_position_error_cm``。

        "不做候选"有多个来源，只有**够格但不够准**那一类计进 ``vetoed``：
        它才是"有一路数据在源源不断地报着，但我们一条都没发出去"。来源不认得、
        看门狗过期、新鲜度掉出范围这些是"它本来就不是候选"，计进去只会让
        真正的信号淹没在噪声里。
        """
        tier = classify(stamped, now=now, cfg=self.cfg)
        if tier is None:
            return None
        bound = position_error_bound_cm(stamped.sample, now=now, cfg=self.cfg)
        if bound > float(self.cfg.max_position_error_cm):
            self.vetoed += 1
            return None
        return tier, bound

    # -- 单槽位 -----------------------------------------------------------
    def _decide_one(
        self, name: str, candidates: Sequence[_Candidate], *, now: float
    ) -> SlotDecision:
        state = self._slots.setdefault(name, _SlotState())
        previous = state.tier

        if not candidates:
            # 一个候选都没有。清掉历史，让下次从零开始——留着旧 tier 会让
            # "现任还在"这个判断在源彻底消失之后依然成立。返回一条空决策，
            # 让调用方看见"这一帧它是真的没了"。
            #
            # "丢掉最后一个源"也是一次档位变化，要计进 ``switch_count``：
            # 它是排查"坐标为什么在跳"的第一手指标，漏计会让最严重的那种
            # 跳变（有→无）在指标上不留痕迹。
            switched = previous is not None
            if switched:
                self.switch_count += 1
            self._slots.pop(name, None)
            return SlotDecision(
                name=name, sample=None, tier=None, previous_tier=previous,
                switched=switched,
            )

        # 先按阶梯，再按**精度**。第二个键只在同档位内起作用——而档位名是由
        # "来源 + 状态"唯一决定的，所以两个不共档位的源永远比不到第二个键
        # （视觉和无线电的档位集不相交）。它今天不生效，但它是"再加一个源"
        # 时的次序规则；没有它，同档位的两个候选只能靠"谁在序列里靠前"取胜，
        # 那是个没人设计过的规则。
        best_tier, best_bound, best = min(
            candidates, key=lambda c: (self._rank.get(c[0], self._floor), c[1])
        )

        incumbent = self._sample_of_source(state.source, candidates)
        if incumbent is None:
            if state.source is None:
                # 第一次见到这个槽位。没有"现任"可言，直接采纳。
                return self._adopt(
                    name, state, best_tier, best, bound=best_bound,
                    previous=previous, now=now,
                )
            # 现任**源**整个塌了：它这一帧一个候选都没产出。
            #
            # 这里**不能**无条件采纳最优候选。设想视觉刚刚不再产出，而缓冲里
            # 躺着一条别的源的报文——它在阶梯上排在视觉之上，于是"啪"地成了
            # 最优。它此前优先级更低，**从没攒过驻留**，也就从没经过跳变检查；
            # 直接采纳可能就是一次半个场地的瞬移。
            #
            # 精度轴**不能**替这道门把关：它只否掉太旧的，而一条 0.2 秒前、
            # 上界才 100cm 的报文照样可以在场地的另一头。所以一律走挑战流程：
            # 本来就在攒驻留的挑战者会因为计时早已满足而立刻接管；刚浮上来的
            # 那个则要老老实实从头攒。
            return self._challenge(
                name, state, best_tier, best, bound=best_bound, incumbent=None,
                previous=previous, now=now,
            )

        if best.sample.source is state.source:
            # **同一个源换了档**：降级（MEASURED→HOLD→BLIND）或恢复（→MEASURED）。
            # 这不是换源，**不走滞回**——见模块文档"例外一"。
            if best_tier == previous:
                # 档位没变，就是一次普通刷新。
                #
                # **必须顺手把驻留计时清掉**：驻留要的是"**连续**满足条件"，而
                # 这一帧最优的又变回现任了，说明原先那个挑战者没能满足条件
                # （无线电从 fresh 掉成了 stale、或者干脆没来）。留着旧计时的
                # 话，它下次恢复时会拿一段早已中断的时间冒充"连续满足"，
                # 一帧就接管——这道门等于不存在。
                state.position = self._position(best.sample)
                state.challenger = None
                return SlotDecision(
                    name=name, sample=best.sample, tier=previous,
                    previous_tier=previous, switched=False,
                    error_bound_cm=best_bound,
                )
            return self._adopt(
                name, state, best_tier, best, bound=best_bound,
                previous=previous, now=now,
            )

        # 现任源还在，但一个**别的**源成了最优。既然现任的档位也在候选里，
        # 最优那个就一定比它优先级更高（否则选不上）——所以这里不需要
        # "挑战者优先级更低"的分支。
        return self._challenge(name, state, best_tier, best, bound=best_bound,
                               incumbent=incumbent, previous=previous, now=now)

    def _challenge(
        self,
        name: str,
        state: _SlotState,
        tier: str,
        stamped: StampedSample,
        *,
        bound: float,
        incumbent: SourceSample | None,
        previous: str | None,
        now: float,
    ) -> SlotDecision:
        """优先级更高的挑战者。要走完驻留、且跳变在允许范围内。

        ``incumbent`` 为 ``None`` 表示现任已经消失——此时"继续用现任"这条退路
        没有了，等待期间只能对外报"不知道"（见下）。
        """
        cfg = self.cfg
        pos = self._position(stamped.sample)
        # 跳变量 = "现在换过去，对外的坐标会瞬移多远"。跟**现任这一帧正在报的
        # 值**比；现任没了就跟它最后报出的位置比（``state.position`` 特意保留，
        # 不随现任消失而清空，否则这个门会在等待期间逐渐失效）。
        incumbent_pos = self._position(incumbent)
        if incumbent_pos is None:
            incumbent_pos = state.position

        if state.challenger != tier:
            state.challenger = tier
            state.challenger_since = float(now)
            state.challenger_anchor = pos
        elif not self._consistent(state.challenger_anchor, pos, cfg.switch_jump_cm):
            # 挑战者自己在这段时间里乱跳："一致样本"不成立，计时重来。
            state.challenger_since = float(now)
            state.challenger_anchor = pos

        jump = self._distance(incumbent_pos, pos)
        required = (
            float(cfg.dwell_s)
            if jump < float(cfg.switch_jump_cm)
            else float(cfg.reacquire_s)
        )

        if (now - state.challenger_since) >= required:
            return self._adopt(
                name, state, tier, stamped, bound=bound, previous=previous, now=now
            )

        if incumbent is None:
            # 现任没了，没有可以继续持有的东西。**诚实地说"不知道"**：
            # 两个源在这个槽位上差了半个场地，而我们刚刚判定原先那个不再可信。
            # 这时候把挑战者抢先报出去，等于绕过自己刚设下的跳变门。
            # ``state.position`` 特意**不清**——下一帧还要拿它当跳变基准。
            return SlotDecision(
                name=name, sample=None, tier=None, previous_tier=previous,
                switched=False,
            )

        # 还没走完驻留：继续用现任。**必须把现任的样本原样发出去**——
        # 返回 None 会被下游读成"这一帧没有信息"，等于每次挑战都制造一次
        # 人为的降级。
        state.position = self._position(incumbent)
        return SlotDecision(
            name=name, sample=incumbent, tier=previous, previous_tier=previous,
            switched=False,
            # 继续用现任，报的就是**现任**的上界。拿挑战者的上界填在这里，
            # 会让"驻留期间"这一段看起来比实际准——而这段正是最该被看见的。
            error_bound_cm=position_error_bound_cm(
                incumbent, now=now, cfg=self.cfg
            ),
        )

    def _adopt(
        self,
        name: str,
        state: _SlotState,
        tier: str,
        stamped: StampedSample,
        *,
        bound: float,
        previous: str | None,
        now: float,
    ) -> SlotDecision:
        state.tier = tier
        state.source = stamped.sample.source
        state.position = self._position(stamped.sample)
        state.challenger = None
        state.challenger_anchor = None
        # 第一次见到这个槽位不算"切换"——它没有从任何东西切过来。
        switched = previous is not None and previous != tier
        if switched:
            self.switch_count += 1
        return SlotDecision(
            name=name, sample=stamped.sample, tier=tier, previous_tier=previous,
            switched=switched, error_bound_cm=bound,
        )

    # -- 工具 -------------------------------------------------------------
    @staticmethod
    def _position(sample: SourceSample | None) -> tuple[float, float] | None:
        if sample is None or sample.position_cm is None:
            return None
        return (float(sample.position_cm[0]), float(sample.position_cm[1]))

    @staticmethod
    def _distance(a, b) -> float:
        """两个位置的差。任一为 ``None`` 时返回 0——**测不出的跳变不算跳变**，
        走常规驻留。把"不知道"当成"跳了很远"会让每次拿不到坐标都多等 800ms。"""
        if a is None or b is None:
            return 0.0
        return float(math.hypot(a[0] - b[0], a[1] - b[1]))

    @staticmethod
    def _consistent(anchor, pos, limit: float) -> bool:
        if anchor is None or pos is None:
            return True
        return Arbiter._distance(anchor, pos) < float(limit)

    @staticmethod
    def _sample_of_source(
        source: SourceKind | None, candidates: Sequence[_Candidate]
    ) -> SourceSample | None:
        """现任**源**这一帧有没有产出候选。

        按源找，不按档位找——档位会在同一个源内变（降级/恢复），源不会。
        """
        if source is None:
            return None
        for _tier, _bound, st in candidates:
            if st.sample.source is source:
                return st.sample
        return None

    def reset(self) -> None:
        self._slots.clear()
        self.switch_count = 0
        self.vetoed = 0
