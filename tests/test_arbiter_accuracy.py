"""融合仲裁的第二条轴：**哪个坐标更准**。

阶梯回答"这是**哪一种**证据"，精度轴回答"这一条现在还值不值得报"。两者正交，
设计理由写在 ``jg_radar.fuse.arbiter`` 的模块文档里。这里把那条轴上的每个结论
钉住，**包括它刻意不做的那件事**（不许推翻阶梯）。

## 为什么这条轴必须存在

``radio_stale`` 在阶梯上高于 ``vision_occlusion_hold``，而它的"陈旧"区间是
0.2s 到 **10s**。一条 9 秒前的报文，9 秒足够机器人跑遍全场——阶梯说"优先用它"，
而它其实一文不值。阶梯的排序编码的是一个**典型**假设（无线电的计量比视觉好），
不是一个逐样本的判断。

## 这条轴最危险的失效模式

不是"没否掉该否的"，而是**把健康数据也否掉了**：症状是每一帧都有数据、
却频繁对外报 MISSING，而所有单元测试都是绿的。所以
``test_the_healthy_replay_is_never_vetoed`` 拿整条 96 帧回放量一遍，
并且钉住余量。
"""

from __future__ import annotations

import pytest

from jg_radar.config import FuseConfig
from jg_radar.fuse import (
    PREDICTION_SIGMA,
    Arbiter,
    SlotDecision,
    StampedSample,
    classify,
    position_error_bound_cm,
)
from jg_radar.types import RobotState, SourceKind, SourceSample

from test_pipeline_replay import _run

CFG = FuseConfig()  # 默认值即方案 §七 的阶梯与阈值
DT = 0.05  # 20fps


def vision(
    name: str = "R1",
    x: float = 1000.0,
    y: float = 700.0,
    *,
    state: RobotState = RobotState.MEASURED,
    observed_at: float = 0.0,
    std: float | None = None,
) -> SourceSample:
    return SourceSample(
        source=SourceKind.VISION,
        name=name,
        position_cm=(int(x), int(y)),
        state=state,
        confidence=0.9,
        observed_at=observed_at,
        revision=1,
        is_predicted=state.is_prediction,
        position_std_cm=std,
    )


def radio(
    name: str = "R1",
    x: float = 1000.0,
    y: float = 700.0,
    *,
    observed_at: float,
    std: float | None = None,
) -> SourceSample:
    return SourceSample(
        source=SourceKind.RADIO,
        name=name,
        position_cm=(int(x), int(y)),
        state=RobotState.MEASURED,
        confidence=0.9,
        observed_at=observed_at,
        revision=1,
        is_predicted=False,
        position_std_cm=std,
    )


def stamped(s: SourceSample, received_at: float) -> StampedSample:
    return StampedSample(sample=s, received_at=received_at)


def decide_one(arb: Arbiter, samples, *, now: float):
    """取 ``R1`` 的决策。

    ``Arbiter.decide`` 对"**完全没听说过**的槽位"不产生条目（留给发布层统一填
    MISSING）。而"所有候选都被否决"的槽位若此前没有历史，看起来和它一模一样
    ——对外也确实是同一件事（都是 MISSING）。这里统一成"空决策"，
    好让每条测试直接断言结果，而不是先处理一个 ``KeyError``。
    """
    return arb.decide(samples, now=now).get("R1", SlotDecision("R1", None, None, False))


# ---------------------------------------------------------------------------
# 上界本身
# ---------------------------------------------------------------------------
def test_a_measured_coordinate_has_a_one_second_shelf_life():
    """实测态的上界是"信息年龄 × 最大速度"，于是**保质期 = 阈值 / 速度**。

    默认值下正好是 1.0 秒——一个能一句话说清的数，也是它被选中当默认值的
    理由：它落在两个已知约束之间。无线电的 ``radio_stale_s``(10s) 远大于它
    （9 秒前的报文必然被否），视觉的看门狗 ``vision_watchdog_s``(0.5s) 远小于
    它（视觉喂进来的实测坐标不会因为这一条被否）。
    """
    cfg = FuseConfig()
    s = vision(observed_at=100.0)

    for age in (0.0, 0.25, 0.5, 0.9):
        assert position_error_bound_cm(s, now=100.0 + age, cfg=cfg) == pytest.approx(
            age * cfg.max_robot_speed_cm_s
        )

    # 保质期就是这两个配置项相除。改了一个忘了另一个，这条会先炸。
    assert cfg.max_position_error_cm / cfg.max_robot_speed_cm_s == pytest.approx(1.0)

    # 阈值是**闭**的：正好 1.0s 还算数，过了就不算
    assert position_error_bound_cm(s, now=101.0, cfg=cfg) == pytest.approx(
        cfg.max_position_error_cm
    )
    assert position_error_bound_cm(s, now=101.05, cfg=cfg) > cfg.max_position_error_cm


def test_the_bound_never_goes_negative_on_a_future_timestamp():
    """时钟抖动可以让 ``observed_at`` 比 ``now`` 晚几微秒。上界是"最多偏多少"，
    负数在这里没有任何意义——钳到 0，而不是让一条本该最准的样本拿到负上界、
    从而在同档位比较里作弊。"""
    s = vision(observed_at=100.0)
    assert position_error_bound_cm(s, now=99.99, cfg=CFG) == 0.0


def test_a_sample_without_a_coordinate_has_no_bound():
    """没有坐标，就无所谓"这条坐标准不准"。``inf`` 而不是 0：0 会被读成
    "绝对准"，而这恰恰是最不该被采纳的那一种。"""
    s = SourceSample(
        source=SourceKind.VISION, name="R1", position_cm=None,
        state=RobotState.MISSING, confidence=0.0, observed_at=0.0,
        revision=1, is_predicted=False,
    )
    assert position_error_bound_cm(s, now=10.0, cfg=CFG) == float("inf")


# ---------------------------------------------------------------------------
# 分派与准入是两件事
# ---------------------------------------------------------------------------
def test_classify_dispatches_a_tier_without_judging_accuracy():
    """``classify`` 只回答"哪一种证据"。一条 9 秒前的报文依然是 ``radio_stale``
    ——它确实陈旧，也确实在阶梯上有一席之地。

    否掉它的是 ``Arbiter``，因为否决**必须计数**（``Arbiter.vetoed``），
    而这是个自由函数、没有地方放计数器；两边各判一次则更糟——谁才是准入的
    权威会变得说不清，而这里恰恰是最不能含糊的地方。
    """
    s = stamped(radio(observed_at=1.0), received_at=10.0)
    assert classify(s, now=10.0, cfg=CFG) == "radio_stale"
    # 而仲裁器在同一时刻、同一条样本上会否掉它
    arb = Arbiter(CFG)
    assert decide_one(arb, [s], now=10.0).empty
    assert arb.vetoed == 1


def test_an_out_of_bounds_report_is_vetoed_and_counted():
    """9 秒前的报文上界 4500cm，是阈值的 9 倍。"""
    arb = Arbiter(CFG)
    d = decide_one(arb, [stamped(radio(observed_at=1.0), received_at=10.0)], now=10.0)

    assert d.empty
    assert d.tier is None
    assert d.error_bound_cm is None, "什么都没采纳，就没有'这条坐标准不准'可言"
    assert arb.vetoed == 1

    # 它一直在报，所以每一帧都该记一笔——这是"有一路数据在源源不断地报着，
    # 但我们一条都没发出去"的唯一外部征兆
    for _ in range(9):
        decide_one(arb, [stamped(radio(observed_at=1.0), received_at=10.0)], now=10.0)
    assert arb.vetoed == 10


def test_the_threshold_is_inclusive():
    """正好卡在上界上的坐标**算数**。判据是 ``>`` 而不是 ``>=``：阈值本身
    就是一个"最多偏这么多还能用"的声明，把它自己排除在外没有道理。"""
    arb = Arbiter(CFG)
    s = stamped(radio(observed_at=9.0), received_at=10.0)  # 年龄 1.0s → 正好 500cm

    assert position_error_bound_cm(s.sample, now=10.0, cfg=CFG) == pytest.approx(
        CFG.max_position_error_cm
    )
    d = decide_one(arb, [s], now=10.0)
    assert not d.empty and d.tier == "radio_stale"
    assert arb.vetoed == 0


def test_tightening_the_threshold_vetoes_more():
    """门槛确实是把尺子，不是个摆设。0.5s 前的报文在默认阈值下上界 250cm
    稳稳通过，收紧到 100cm 就被否。"""
    now = 10.0
    s = stamped(radio(observed_at=now - 0.5), received_at=now)

    loose = Arbiter(FuseConfig())
    assert not decide_one(loose, [s], now=now).empty
    assert loose.vetoed == 0

    strict = Arbiter(FuseConfig(max_position_error_cm=100.0))
    assert decide_one(strict, [s], now=now).empty
    assert strict.vetoed == 1


# ---------------------------------------------------------------------------
# 预测态：用协方差，不用那个冻住的 observed_at
# ---------------------------------------------------------------------------
def test_a_prediction_is_judged_by_its_covariance_not_by_its_frozen_timestamp():
    """盲区预测的 ``observed_at`` **刻意**冻在最后一次真观测上，可能好几秒前。

    拿它算年龄会把**每一条**盲区预测都否掉——而盲区预测存在的理由正是
    在这段时间里继续报数。正确的量是滤波器外推出来的协方差：预测步已经把
    时间折进 ``P`` 里了，所以 ``position_std_cm`` 同样是"对**当前**位置的估计"。
    """
    now = 10.0
    held = vision(state=RobotState.BLIND_PREDICTION, observed_at=0.0, std=20.0)

    # 按时间算是 5000cm（早就超界），按协方差算是 60cm
    assert position_error_bound_cm(held, now=now, cfg=CFG) == pytest.approx(
        PREDICTION_SIGMA * 20.0
    )

    arb = Arbiter(CFG)
    d = decide_one(arb, [stamped(held, received_at=now)], now=now)
    assert d.tier == "vision_blind"
    assert d.error_bound_cm == pytest.approx(60.0)
    assert arb.vetoed == 0


def test_a_prediction_without_a_covariance_falls_back_to_time():
    """给不出不确定度的预测，我们**没有任何依据**说它准——退回时间上界。

    这是诚实，不是保守：槽位转为 MISSING、``vetoed`` 涨一格、在诊断里看得见，
    而不是悄悄报一个谁也评估不了的坐标。
    """
    arb = Arbiter(CFG)
    now = 10.0
    # 刚转进盲区预测（observed_at 才 0.5s 前）：时间上界 250cm，还算数
    fresh_enough = vision(
        state=RobotState.BLIND_PREDICTION, observed_at=now - 0.5, std=None
    )
    d = decide_one(arb, [stamped(fresh_enough, received_at=now)], now=now)
    assert d.tier == "vision_blind"
    assert d.error_bound_cm == pytest.approx(250.0)
    assert arb.vetoed == 0

    # 冻了很久的那个：上界 5000cm，否掉
    arb2 = Arbiter(CFG)
    long_frozen = vision(state=RobotState.BLIND_PREDICTION, observed_at=0.0, std=None)
    assert decide_one(arb2, [stamped(long_frozen, received_at=now)], now=now).empty
    assert arb2.vetoed == 1


def test_occlusion_hold_also_falls_back_to_time_when_it_has_no_covariance():
    """``OCCLUSION_HOLD`` 和 ``BLIND_PREDICTION`` 一样是预测态。

    真实流水线里两者都带着世界滤波器的协方差（见 ``track/types.py``），
    所以协方差那一支才是常态；这里钉的是"没有时的退路",免得将来有人把
    ``is_predicted`` 的分支写成只认 ``BLIND_PREDICTION``。
    """
    now = 10.0
    s = vision(state=RobotState.OCCLUSION_HOLD, observed_at=now - 0.1, std=None)
    assert position_error_bound_cm(s, now=now, cfg=CFG) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# 上界落到决策里
# ---------------------------------------------------------------------------
def test_the_decision_carries_the_bound_on_every_path():
    """四种决策路径都要带上"这条坐标准不准"，缺一种就等于那一种在日志里
    和别的情形长得一模一样（``R1 vision_occlusion_hold`` 后面跟 ±78cm
    和跟 ±1200cm 说的是完全不同的事，而档位名是一样的）。
    """
    arb = Arbiter(CFG)
    t = 10.0

    # 1) 首次采纳
    d = decide_one(arb, [stamped(vision(observed_at=t, std=2.0), t)], now=t)
    assert d.tier == "vision_measured" and d.error_bound_cm == 0.0

    # 2) 同档位刷新（档位没变、源没变，走的是另一条 return）
    d = decide_one(arb, [stamped(vision(observed_at=t - 0.2, std=2.0), t)], now=t)
    assert d.tier == "vision_measured" and d.error_bound_cm == pytest.approx(100.0)

    # 3) 挑战者在攒驻留、现任被继续持有：报的是**现任**的上界。
    #    拿挑战者的上界填在这里，会让"驻留期间"这一段看起来比实际准——
    #    而这段正是最该被看见的。
    d = decide_one(arb, 
        [stamped(vision(observed_at=t - 0.2), t),      # 现任 VISION，上界 100cm
         stamped(radio(observed_at=t, x=1010), t)],    # 挑战者 RADIO，上界 0cm
        now=t,
    )
    assert d.sample.source is SourceKind.VISION, "dwell 还没走完，不该换源"
    assert d.error_bound_cm == pytest.approx(100.0), "报的是现任的上界，不是挑战者的"

    # 4) 没有任何候选
    d = decide_one(arb, [], now=t + DT)
    assert d.empty and d.error_bound_cm is None


# ---------------------------------------------------------------------------
# 刻意不做：精度轴不许推翻阶梯
# ---------------------------------------------------------------------------
def test_a_hard_bound_never_loses_to_a_predictions_three_sigma():
    """一条 0.9 秒前的无线电（上界 **450cm**）对一条盲区预测（3σ = **30cm**）。

    数字说盲区"更准"，但那是拿一个**硬上界**去比一个**猜测的 3σ**：前者是
    "最多偏这么多"，后者是滤波器对自己外推的自信。用后者赢前者，等于让猜测
    盖过测量。所以阶梯照旧——``radio_stale``(rank 2) 赢 ``vision_blind``(rank 4)。
    """
    now = 10.0
    stale = radio(observed_at=now - 0.9)                              # 上界 450cm
    blind = vision(state=RobotState.BLIND_PREDICTION,
                   observed_at=now - 5.0, std=10.0)                   # 上界 30cm

    assert position_error_bound_cm(blind, now=now, cfg=CFG) == pytest.approx(30.0)
    assert position_error_bound_cm(stale, now=now, cfg=CFG) == pytest.approx(450.0)

    arb = Arbiter(CFG)
    d = decide_one(arb, 
        [stamped(blind, received_at=now), stamped(stale, received_at=now)], now=now
    )
    assert d.tier == "radio_stale", "阶梯赢了——130% 的精度差不足以推翻证据的种类"
    assert d.error_bound_cm == pytest.approx(450.0)


# ---------------------------------------------------------------------------
# 同档位内的次序
# ---------------------------------------------------------------------------
def test_the_tighter_bound_wins_inside_one_tier():
    """同档位取上界更小的那个——**与它在序列里的位置无关**。

    今天两个源不共档位（视觉和无线电的档位集不相交），所以这一条还轮不到。
    它测的是**规则本身**：没有它，同档位的候选只能靠"谁在序列里靠前"取胜，
    而那是没人设计过的规则。加第三个源的那天，这条就是它的次序定义。

    这里用同一路无线电的两条报文来构造同档位——真实场景是链路重传/多路解码，
    也正是 ``Arbiter.decide`` 收**序列**而不是字典的理由。
    """
    now = 10.0
    for order in ((0.3, 0.9), (0.9, 0.3)):
        arb = Arbiter(CFG)
        samples = [stamped(radio(observed_at=now - age), now) for age in order]

        d = decide_one(arb, samples, now=now)
        assert d.tier == "radio_stale"
        assert d.error_bound_cm == pytest.approx(150.0), (
            f"序列顺序 {order} 下选错了——次序规则不该依赖谁写在前面"
        )
        assert d.sample.observed_at == pytest.approx(now - 0.3)


# ---------------------------------------------------------------------------
# 整条回放：健康数据一帧都不该被否
# ---------------------------------------------------------------------------
def test_replay_keeps_healthy_measurements_and_rejects_far_blind_waypoints():
    """实测保持可用；航点误差须包含输出点相对滤波中心的偏移。"""
    frames, loop = _run()
    arbiter = loop.stage_b.arbiter

    assert arbiter.vetoed > 0, "回放中的远航点应被误差门限拒绝"
    for frame in frames:
        if frame.frame_seq >= 1:
            assert frame.robots["B3"].state is RobotState.MEASURED
        if frame.frame_seq >= 31:
            assert frame.robots["R2"].state is RobotState.MEASURED
        if 1 <= frame.frame_seq <= 24:
            assert frame.robots["B1"].state is RobotState.MEASURED

    bounds = []
    for frame in frames:
        for robot in frame.robots.values():
            if robot.position_std_cm is None:
                continue
            weight = PREDICTION_SIGMA if robot.is_predicted else 1.0
            bounds.append(weight * robot.position_std_cm + robot.prediction_offset_cm)

    assert bounds, "回放里一条带协方差的样本都没有，这条测试等于没跑"
    assert max(bounds) <= CFG.max_position_error_cm, (
        f"最大上界 {max(bounds):.1f}cm，与阈值 {CFG.max_position_error_cm}cm 的"
        "余量已不足四倍"
    )


def test_reset_clears_the_veto_counter():
    """``reset`` 给标定变更 / 回放换段用。计数器不清的话，第二段一开头就会
    带着上一段的否决数，而那个数解释的是另一批数据。"""
    arb = Arbiter(CFG)
    decide_one(arb, [stamped(radio(observed_at=1.0), received_at=10.0)], now=10.0)
    assert arb.vetoed == 1

    arb.reset()
    assert arb.vetoed == 0
