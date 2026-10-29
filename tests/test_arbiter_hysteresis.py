"""融合仲裁 —— 方案 §九 测试 6 与 M5 的验收。

M5 的验收原话是：**"插入无线电桩后零改动赢得仲裁"**。所以 ``test_radio_stub_wins_arbitration_with_zero_vision_changes``
那条测试**只使用公开契约**（``SourceSample`` + ``LatestValueBuffer``）来演示接入，
不碰视觉侧任何模块——它要是需要改视觉代码才能通过，验收就是没达成。

其余测试盯的是"丝滑"的另一半：**不切换**。一个每秒跳两次的坐标凑不满
"连续 3 秒"，所以"该切的时候切"只对一半，"不该切的时候坚决不切"才是重点。
"""

from __future__ import annotations

import pytest

from jg_radar.config import FuseConfig
from jg_radar.fuse import (
    Arbiter,
    LatestValueBuffer,
    PublishLayer,
    SlotDecision,
    StampedSample,
    classify,
    radio_tier,
    vision_tier,
)
from jg_radar.types import RobotState, SourceKind, SourceSample

CFG = FuseConfig()  # 默认值即方案 §七 的阶梯与阈值
DT = 0.05  # 20fps


def sample(
    name: str = "R1",
    x: float = 1000.0,
    y: float = 700.0,
    *,
    source: SourceKind = SourceKind.VISION,
    state: RobotState = RobotState.MEASURED,
    observed_at: float = 0.0,
    confidence: float = 0.9,
) -> SourceSample:
    return SourceSample(
        source=source,
        name=name,
        position_cm=(int(x), int(y)),
        state=state,
        confidence=confidence,
        observed_at=observed_at,
        revision=1,
        is_predicted=state.is_prediction,
    )


def stamped(s: SourceSample, received_at: float) -> StampedSample:
    return StampedSample(sample=s, received_at=received_at)


def radio(name: str = "R1", x: float = 1000.0, y: float = 700.0, *, observed_at: float):
    return sample(
        name, x, y, source=SourceKind.RADIO, state=RobotState.MEASURED,
        observed_at=observed_at,
    )


# ---------------------------------------------------------------------------
# 档位判定：新鲜度与看门狗
# ---------------------------------------------------------------------------
def test_vision_watchdog_measures_liveness_not_information_age():
    """看门狗量的是"源还活着吗"，不是"这条信息多旧"。

    这两者在预测态上是**故意**分开的：一个 BLIND_PREDICTION 样本的
    ``observed_at`` 停滞在最后一次真观测（可能好几秒前），但源每帧都在写。
    拿 observed_at 去看门狗，所有预测态都会被判成"视觉死了"，
    盲区预测就永远发不出去。
    """
    s = sample(state=RobotState.BLIND_PREDICTION, observed_at=0.0)

    # 信息很旧（3 秒），但源刚写过 —— 必须是合法候选
    live = stamped(s, received_at=3.0)
    assert live.information_age(3.0) == pytest.approx(3.0)
    assert live.liveness_age(3.0) == pytest.approx(0.0)
    assert vision_tier(live, now=3.0, watchdog_s=0.5) == "vision_blind"

    # 信息很新，但源 1 秒没动静了 —— 必须被看门狗拿掉
    dead = stamped(sample(state=RobotState.MEASURED, observed_at=2.9), received_at=2.0)
    assert vision_tier(dead, now=3.0, watchdog_s=0.5) is None


def test_radio_tier_uses_information_age():
    """无线电的陈旧看的是空口接收时刻——缓冲里躺着一条 5 秒前的报文，
    它依然是陈旧的，不会因为我"刚读到"就变新鲜。"""
    fresh = stamped(radio(observed_at=9.9), received_at=10.0)
    assert radio_tier(fresh, now=10.0, fresh_s=0.2, stale_s=10.0) == "radio_fresh"

    stale = stamped(radio(observed_at=5.0), received_at=10.0)  # 刚读到，但报文很旧
    assert radio_tier(stale, now=10.0, fresh_s=0.2, stale_s=10.0) == "radio_stale"

    # 注意 stale_s 是闭区间：年龄正好 10.0s 仍算"陈旧但可用"，要超出去才算没了
    assert radio_tier(stamped(radio(observed_at=0.0), received_at=10.0),
                      now=10.0, fresh_s=0.2, stale_s=10.0) == "radio_stale"
    gone = stamped(radio(observed_at=-1.0), received_at=10.0)
    assert radio_tier(gone, now=10.0, fresh_s=0.2, stale_s=10.0) is None


def test_missing_is_never_a_candidate():
    """MISSING 没有坐标，任何档位都不该采纳它。"""
    s = sample(state=RobotState.MISSING, x=0, y=0)
    s = SourceSample(**{**s.__dict__, "position_cm": None})
    assert classify(stamped(s, 0.0), now=0.0, cfg=CFG) is None


def test_unknown_tier_name_fails_loudly():
    """阶梯里写错一个档位名，会静默地让那一档永远轮不到——必须启动即失败。"""
    with pytest.raises(ValueError, match="未知档位"):
        Arbiter(FuseConfig(priority=("radio_fresh", "visoin_measured")))


# ---------------------------------------------------------------------------
# 现任保持
# ---------------------------------------------------------------------------
def test_incumbent_holds_against_a_lower_priority_source():
    """radio_stale 在阶梯上低于 vision_measured，永远抢不走。

    报文的年龄特意取 0.5s 而不是"随便一个陈旧值"：9 秒前的那种会被精度轴
    在准入那一步直接否掉，于是这条测试会**因为错误的原因**通过——它想验的是
    阶梯，却让否决替它把答案给了。0.5s 的报文上界是 250cm，稳稳在阈值内，
    于是"抢不走"这件事只可能归因于阶梯。
    """
    arb = Arbiter(CFG)
    for i in range(20):
        t = i * DT
        d = arb.decide(
            [stamped(sample("R1", 1000, 700, observed_at=t), t),
             # 0.5 秒前的报文 ⇒ radio_stale（rank 2），低于 vision_measured（rank 1）
             stamped(radio("R1", 1000, 700, observed_at=t - 0.5), t)],
            now=t,
        )["R1"]
        assert d.tier == "vision_measured"
        assert d.sample.source is SourceKind.VISION
    assert arb.switch_count == 0
    assert arb.vetoed == 0, "这条测试里不该有任何候选被精度轴否掉——否则它验的就不是阶梯"


def test_a_single_frame_priority_flip_does_not_switch():
    """挑战者只闪了一帧就被 dwell 挡掉——单帧翻转不构成换源理由。"""
    arb = Arbiter(CFG)
    t = 0.0
    arb.decide([stamped(sample("R1", 1000, 700, observed_at=t), t)], now=t)

    t += DT
    d = arb.decide(
        [stamped(sample("R1", 1000, 700, observed_at=t), t),
         stamped(radio("R1", 1010, 700, observed_at=t), t)],
        now=t,
    )["R1"]
    assert d.tier == "vision_measured", "第一帧就该被 dwell 挡住"
    assert d.sample is not None, "被挡住时仍然要把现任发出去，不能凭空制造一次降级"

    t += DT
    d = arb.decide([stamped(sample("R1", 1000, 700, observed_at=t), t)], now=t)["R1"]
    assert d.tier == "vision_measured"
    assert arb.switch_count == 0


def test_radio_fresh_takes_over_after_the_dwell():
    """优先级更高的源连续满足 dwell_s 之后接管——而且**在那一帧之前不接管**。"""
    arb = Arbiter(CFG)
    t = 0.0
    arb.decide([stamped(sample("R1", 1000, 700, observed_at=t), t)], now=t)

    seen = []
    for _ in range(12):
        t += DT
        d = arb.decide(
            [stamped(sample("R1", 1000, 700, observed_at=t), t),
             stamped(radio("R1", 1030, 700, observed_at=t), t)],
            now=t,
        )["R1"]
        seen.append(d.tier)

    first_radio = seen.index("radio_fresh") * DT + DT
    assert 0.25 <= first_radio <= 0.40, f"接管时刻 {first_radio}s 超出 dwell_s=0.3 的预期"
    assert seen[-1] == "radio_fresh"
    assert arb.switch_count == 1


# ---------------------------------------------------------------------------
# 跳变抑制
# ---------------------------------------------------------------------------
def test_a_demoted_challenger_must_restart_its_dwell():
    """挑战者掉出"满足条件"的状态后，连续性就断了，回来时驻留要重攒。

    否则一个闪烁的源（fresh 一下、stale 一下）可以拿一段早已中断的时间
    冒充"连续满足 dwell_s"，一帧就接管——驻留这道门等于不存在。

    那"掉出去"的一帧要掉到 ``radio_stale``，**不能掉到被否掉**：被否掉是
    从候选集合里消失，而这里要验的正是"它还在候选里、只是降了档"。
    """
    arb = Arbiter(CFG)
    t = 0.0
    arb.decide([stamped(sample("R1", 1000, 700, observed_at=t), t)], now=t)

    # radio_fresh 攒 0.25s —— 离 dwell_s=0.3 还差一点
    for _ in range(5):
        t += DT
        d = arb.decide([stamped(sample("R1", 1000, 700, observed_at=t), t),
                        stamped(radio("R1", 1010, 700, observed_at=t), t)], now=t)["R1"]
        assert d.tier == "vision_measured"

    # 一帧掉到 radio_stale：它此刻**不满足** fresh 条件，连续性断了
    t += DT
    arb.decide([stamped(sample("R1", 1000, 700, observed_at=t), t),
                stamped(radio("R1", 1010, 700, observed_at=t - 0.5), t)], now=t)
    assert arb.vetoed == 0

    # 立刻恢复 fresh。旧计时若被留下，此刻已经 0.30s ≥ dwell_s，会当场接管。
    t += DT
    d = arb.decide([stamped(sample("R1", 1000, 700, observed_at=t), t),
                    stamped(radio("R1", 1010, 700, observed_at=t), t)], now=t)["R1"]
    assert d.tier == "vision_measured", "掉出候选后驻留必须重攒，不能接着用旧计时"


def test_a_large_jump_needs_the_longer_reacquire():
    """跳变超过 switch_jump_cm，门槛从 300ms 抬到 800ms。"""
    arb = Arbiter(CFG)
    t = 0.0
    arb.decide([stamped(sample("R1", 200, 700, observed_at=t), t)], now=t)

    took_over_at = None
    for i in range(40):
        t += DT
        d = arb.decide(
            [stamped(sample("R1", 200, 700, observed_at=t), t),
             stamped(radio("R1", 200 + 400, 700, observed_at=t), t)],  # 400cm ≫ 150
            now=t,
        )["R1"]
        if d.tier == "radio_fresh" and took_over_at is None:
            took_over_at = (i + 1) * DT
    assert took_over_at is not None
    assert 0.75 <= took_over_at <= 0.95, f"接管时刻 {took_over_at}s，应为 reacquire_s=0.8"


def test_a_jumpy_challenger_never_completes_its_dwell():
    """挑战者自己乱跳时，"一致样本"不成立，计时反复清零，永远接管不了。

    这一条防的是：只看"它优先级高、它存在了 300ms"，于是一个正在乱跳的源
    也能接管，接手之后继续乱跳。
    """
    arb = Arbiter(CFG)
    t = 0.0
    arb.decide([stamped(sample("R1", 1000, 700, observed_at=t), t)], now=t)

    for i in range(60):
        t += DT
        # 每一帧都在 ±600cm 之间来回蹦
        jitter = 600 if i % 2 else -600
        d = arb.decide(
            [stamped(sample("R1", 1000, 700, observed_at=t), t),
             stamped(radio("R1", 1000 + jitter, 700, observed_at=t), t)],
            now=t,
        )["R1"]
        assert d.tier == "vision_measured", f"第 {i} 帧被乱跳的挑战者接管了"
    assert arb.switch_count == 0


def test_a_degraded_incumbent_never_hands_the_slot_to_an_out_of_bounds_stranger():
    """视觉从 MEASURED 掉到 HOLD，而缓冲里躺着一条 9 秒前的 radio_stale。

    这是最容易漏的一条路径：``radio_stale`` 在阶梯上排在 ``vision_occlusion_hold``
    之上，于是视觉一降级，它就"啪"地成了最优。它此前优先级更低，从没攒过驻留、
    从没经过跳变检查，而 9 秒足够机器人跑遍全场。

    注意现任**源**并没有消失（视觉还在产出 HOLD），所以走的是
    ``incumbent is not None`` 那条分支——等待期间还能继续报视觉的坐标。

    跳变门能兜住它，但那是**绕远路**：9 秒的报文最多可能偏 9 × 500 = 4500cm，
    它本身早就一文不值了，等 800ms 再采纳只是把一次瞬移推迟 800ms。精度轴
    在准入那一步就否掉它，于是这里能断言一个**更强**的性质——不管等多久，
    它都不会被采纳，一帧都不会。
    """
    arb = Arbiter(CFG)
    t = 0.0
    arb.decide([stamped(sample("R1", 200, 700, observed_at=t), t)], now=t)

    # 无线电在很远的地方"一直报着"（9 秒前收到的报文），视觉一路 MEASURED
    for _ in range(10):
        t += DT
        arb.decide(
            [stamped(sample("R1", 200, 700, observed_at=t), t),
             stamped(radio("R1", 2500, 700, observed_at=t - 9.0), t)],
            now=t,
        )

    # 视觉掉到 HOLD：MEASURED 不再是候选，radio_stale 立刻成了最优
    for i in range(60):  # 3 秒 —— 远超 reacquire_s=0.8
        t += DT
        d = arb.decide(
            [stamped(sample("R1", 200, 700, state=RobotState.OCCLUSION_HOLD,
                            observed_at=t - 0.1), t),
             stamped(radio("R1", 2500, 700, observed_at=t - 9.0), t)],
            now=t,
        )["R1"]
        assert d.sample is None or d.sample.source is not SourceKind.RADIO, (
            f"第 {i} 帧采纳了一条 9 秒前的报文——它最多可能偏 4500cm"
        )
        assert d.tier in (None, "vision_occlusion_hold")

    # 每一帧都被否 —— 这正是"有一路在报，但我们一条都没发出去"的那个征兆
    assert arb.vetoed >= 70


def test_a_vanished_incumbent_reports_nothing_while_a_challenger_waits():
    """现任**源**整个消失、挑战者在半个场地外 —— 等待期间**诚实地说"不知道"**。

    这条走的是 ``incumbent is None`` 那条分支，跟上一条完全不同：那里现任源还在
    产出（哪怕只是 HOLD），等待期间还能继续报它的坐标；这里没有东西可报。
    剩下两个选项都不能选——报挑战者等于绕过自己刚设下的跳变门（2300cm 的
    瞬移），报那个已经消失的现任等于假装还有数据。

    反过来也要确认它**没有**变成永久沉默：攒够 ``reacquire_s`` 之后必须接管。
    """
    arb = Arbiter(CFG)
    t = 0.0
    arb.decide([stamped(sample("R1", 200, 700, observed_at=t), t)], now=t)

    # 视觉先正常跑一段，把"最后报出的位置"钉在 (200,700)——等待期间就是拿它
    # 当跳变基准的（``state.position`` 特意不随现任消失而清空）。
    for _ in range(10):
        t += DT
        arb.decide([stamped(sample("R1", 200, 700, observed_at=t), t)], now=t)

    took_over = None
    for i in range(40):
        t += DT
        # 视觉彻底不见了：这一个槽位的候选里只剩无线电
        d = arb.decide(
            [stamped(radio("R1", 2500, 700, observed_at=t), t)],  # 新鲜：上界 0
            now=t,
        )["R1"]
        if d.sample is not None:
            assert d.sample.source is SourceKind.RADIO, "视觉已经不产出了，报出来的只可能是无线电"
            took_over = (i + 1) * DT
            break
        assert d.tier is None

    assert took_over is not None, "新鲜报文攒够 reacquire_s 之后应该接管"
    assert took_over >= 0.75, f"接管时刻 {took_over}s，应为 reacquire_s=0.8"
    assert arb.vetoed == 0, "新鲜报文的上界是 0cm，不该被精度轴碰到"


# ---------------------------------------------------------------------------
# M5 验收：无线电桩零改动赢得仲裁
# ---------------------------------------------------------------------------
def test_radio_stub_wins_arbitration_with_zero_vision_changes():
    """**M5 验收。**

    整条接入只用到公开契约：构造 ``SourceSample(source=RADIO)`` → 写进
    ``LatestValueBuffer`` → 仲裁自然认它。视觉侧一行没动，配置一行没改
    （``radio_fresh`` 本来就在阶梯里）。
    """
    buf = LatestValueBuffer()
    arb = Arbiter(CFG)
    pub = PublishLayer(CFG, slot_names=["R1", "B1"])

    t = 0.0
    # 视觉先把 R1 立起来
    buf.put(sample("R1", 900, 700, observed_at=t), now=t)
    d = arb.decide(buf.snapshot(), now=t)
    assert d["R1"].tier == "vision_measured"

    # 无线电上线。**只有下面这三行**是"接入"。
    buf.put(radio("R1", 905, 705, observed_at=t), now=t)

    # 走完 dwell_s
    for _ in range(20):
        t += DT
        buf.put(sample("R1", 900, 700, observed_at=t), now=t)
        buf.put(radio("R1", 905, 705, observed_at=t), now=t)
        d = arb.decide(buf.snapshot(), now=t)

    assert d["R1"].tier == "radio_fresh"
    frame = pub.build(d, frame_seq=1, captured_at=t, published_at=t)
    assert frame.robots["R1"].source is SourceKind.RADIO
    assert frame.robots["R1"].position_cm == (905, 705)


# ---------------------------------------------------------------------------
# 发布层
# ---------------------------------------------------------------------------
def test_publish_fills_every_slot_with_missing():
    """跟踪器只产出"有轨迹且身份已定"的槽位，输出字典是稀疏的；
    对外协议是固定字段的，少一个槽位不是"没说"而是"格式错"。"""
    pub = PublishLayer(CFG, slot_names=["R1", "B1", "B7"])
    arb = Arbiter(CFG)
    t = 0.0
    arb.decide([stamped(sample("R1", 1000, 700, observed_at=t), t)], now=t)

    frame = pub.build(arb.decide([stamped(sample("R1", 1000, 700, observed_at=t), t)], now=t),
                      frame_seq=7, captured_at=t, published_at=t)
    assert set(frame.robots) == {"R1", "B1", "B7"}
    assert frame.robots["R1"].state is RobotState.MEASURED
    for name in ("B1", "B7"):
        assert frame.robots[name].state is RobotState.MISSING
        assert frame.robots[name].position_cm is None
    frame.validate()


def test_publish_keeps_measurement_time_and_orders_publications_across_source_switch():
    """保留旧源真实时间，发布顺序使用独立序号。"""
    pub = PublishLayer(CFG, slot_names=["R1"])
    seen = []

    # 视觉：observed_at 一路领先
    t = 10.05
    d = {"R1": SlotDecision("R1", sample("R1", 900, 700, observed_at=t),
                            "vision_measured", False)}
    seen.append(pub.build(d, frame_seq=1, captured_at=t, published_at=t).robots["R1"])

    # 无线电接手，但它的时间戳比视觉上一条**更早**（解码延迟）
    t = 10.1
    d = {"R1": SlotDecision("R1", radio("R1", 900, 700, observed_at=10.02),
                            "radio_fresh", True)}
    seen.append(pub.build(d, frame_seq=2, captured_at=t, published_at=t).robots["R1"])

    assert seen[0].observed_at == 10.05
    assert seen[1].observed_at == 10.02
    assert seen[1].source is SourceKind.RADIO, "换源本身仍然要如实记录"


def test_a_tracker_missing_retracts_the_stale_measured_sample():
    """视觉把一条轨迹判成 MISSING 之后，缓冲里的旧 MEASURED **必须被覆盖掉**。

    最新值缓冲只留最新一条，所以写入自带撤回效果。要是这层"优化"掉没有坐标的
    MISSING 样本，仲裁层就会继续拿 0.1 秒前那条 MEASURED 当实时测量报出去
    —— 陈旧数据伪装成新鲜测量，这是契约里最不能容忍的失效模式。
    """
    from jg_radar.fuse import VisionSource
    from jg_radar.types import RobotSlotOutput

    src = VisionSource()
    buf = LatestValueBuffer()

    buf.put_many(src.submit({
        "R1": RobotSlotOutput(
            name="R1", position_cm=(1000, 700), state=RobotState.MEASURED,
            confidence=0.9, observed_at=0.0, source=SourceKind.VISION,
            is_predicted=False,
        )
    }, now=0.0), now=0.0)

    t = 0.1  # 远小于 500ms 看门狗
    buf.put_many(src.submit({
        "R1": RobotSlotOutput(
            name="R1", position_cm=None, state=RobotState.MISSING,
            confidence=0.0, observed_at=t, source=SourceKind.VISION,
            is_predicted=False,
        )
    }, now=t), now=t)

    arb = Arbiter(CFG)
    d = arb.decide(buf.snapshot(), now=t)
    # MISSING 不是任何档位的候选，所以它压根不会出现在决策里——发布层会把
    # 这个空槽位填成 MISSING。关键是**绝不能**再冒出一条 MEASURED。
    decided = d.get("R1")
    assert decided is None or decided.sample is None or \
        decided.sample.state is not RobotState.MEASURED, (
        "MISSING 没能撤回上一条 MEASURED —— 仲裁层还在把一个已判死的坐标当测量"
    )


def test_end_to_end_a_radio_source_joins_a_running_vision_pipeline():
    """M5 验收的加强版：**真的跑一条视觉流水线**，中途把无线电插进去。

    上面那条用手搓样本演示契约；这条把 ``MultiTracker`` 真跑起来，视觉侧
    的每一行代码都是原来那些，接入动作只有"多注册一个源"。
    """
    from jg_radar.config import TrackConfig
    from jg_radar.fuse import RadioReading, RadioSource, VisionSource
    from jg_radar.track import MultiTracker, TrackObservation

    tracker = MultiTracker(TrackConfig(), slot_names=["R1", "B1"], waypoints=[])
    vision = VisionSource()
    radio_src = RadioSource()
    buf = LatestValueBuffer()
    arb = Arbiter(CFG)
    pub = PublishLayer(CFG, slot_names=["R1", "B1"])

    def drive(t: float, *, cls: str = "B1", digit: str = "1"):
        o = TrackObservation(
            bbox=(860.0, 680.0, 940.0, 720.0),
            world_cm=(900.0, 700.0),
            confidence=0.9,
            car_class=cls,
            digit=digit,
        )
        buf.put_many(vision.submit(tracker.step([o], now=t), now=t), now=t)

    # 视觉先跑起来：攒够命中，身份由整车类别 B1 建立
    t = 0.0
    for _ in range(4):
        drive(t)
        t += DT
    frame = pub.build(arb.decide(buf.snapshot(), now=t), frame_seq=1,
                      captured_at=t, published_at=t)
    assert frame.robots["B1"].state is RobotState.MEASURED
    assert frame.robots["B1"].source is SourceKind.VISION

    # ---- 接入无线电：以下是全部改动 ----
    radio_src.feed([RadioReading("B1", (905, 705), observed_at=t)], now=t)
    buf.put_many(radio_src.poll(t), now=t)

    # 两个源同时供数，走完 dwell_s。
    # 仲裁**必须每帧都跑**：滞回是有状态的，驻留计时靠 ``decide`` 推进。
    # 攒到最后一次性调用，计时器只会从那一刻开始算，永远等不到接管。
    for _ in range(10):
        drive(t)
        radio_src.feed([RadioReading("B1", (905, 705), observed_at=t)], now=t)
        buf.put_many(radio_src.poll(t), now=t)
        t += DT
        decisions = arb.decide(buf.snapshot(), now=t)

    frame = pub.build(decisions, frame_seq=2, captured_at=t, published_at=t)
    assert frame.robots["B1"].source is SourceKind.RADIO
    assert frame.robots["B1"].position_cm == (905, 705)
    assert frame.robots["R1"].state is RobotState.MISSING, "没被供数的槽位照旧 MISSING"
    frame.validate()


# ---------------------------------------------------------------------------
# 槽位彻底消失
# ---------------------------------------------------------------------------
def test_a_slot_with_no_candidates_at_all_reports_empty():
    """所有源都不报这个槽位了。决策必须是空的一条——发布层据此填 MISSING。

    这条路径和"降级到 vision_blind"不是一回事：那时还有候选、还有坐标；
    这里是真的一条候选都没有。
    """
    arb = Arbiter(CFG)
    t = 0.0
    d = arb.decide([stamped(sample("R1", 1000, 700, observed_at=t), t)], now=t)["R1"]
    assert d.tier == "vision_measured"

    # 视觉源整个消失（进程没了、相机拔了）
    t += DT
    d = arb.decide([], now=t)["R1"]
    assert d.sample is None and d.tier is None
    assert d.empty, "没有候选时 SlotDecision.empty 必须为真"
    assert d.switched, "从有源到没源是一次档位变化，要能被诊断看见"

    # 再消失一帧：历史已经清干净，这个槽位连条目都不再产生
    # （发布层照样会填 MISSING——"没有决策"和"决策为空"对外是同一件事）
    t += DT
    assert "R1" not in arb.decide([], now=t)
    assert arb.switch_count == 1, "切换只在真正发生的那一帧计一次"


def test_a_vision_sample_older_than_the_watchdog_is_dropped():
    """源进程还在、但不再写缓冲（卡死）——看门狗过期后同样不能当候选。"""
    arb = Arbiter(CFG)
    buf = LatestValueBuffer()
    t = 0.0
    buf.put(sample("R1", 1000, 700, observed_at=t), now=t)
    assert arb.decide(buf.snapshot(), now=t)["R1"].tier == "vision_measured"

    # 缓冲没清，但源已经 1 秒没写了 > vision_watchdog_s=0.5
    t = 1.0
    d = arb.decide(buf.snapshot(), now=t)["R1"]
    assert d.sample is None, "卡死的源不能继续供数"


def test_fused_samples_are_not_candidates():
    """``SourceKind.FUSED`` 是给输出用的来源标签，不是一个输入源。
    未知来源一律不做候选——**不猜**。"""
    arb = Arbiter(CFG)
    s = sample("R1", 1000, 700, source=SourceKind.FUSED, observed_at=0.0)
    assert classify(stamped(s, 0.0), now=0.0, cfg=CFG) is None
    assert "R1" not in arb.decide([stamped(s, 0.0)], now=0.0)


def test_reset_clears_the_hysteresis_state():
    """``reset`` 给标定变更 / 回放换段用：不清干净的话，上一段序列的"现任"
    会带到下一段，第二段开头就会凭空多出一次切换。"""
    arb = Arbiter(CFG)
    t = 0.0
    arb.decide([stamped(sample("R1", 1000, 700, observed_at=t), t)], now=t)
    assert arb.switch_count == 0

    arb.reset()
    assert arb.switch_count == 0
    # reset 之后同一个槽位是"第一次见到"，不产生切换
    d = arb.decide([stamped(sample("R1", 1000, 700, observed_at=t), t)], now=t)["R1"]
    assert d.tier == "vision_measured" and not d.switched
    assert arb.switch_count == 0


def test_both_sources_satisfy_the_same_protocol():
    """视觉与无线电形状相同——这正是"将来插进来能零改动"的前提。
    形状一旦分叉，"零改动"就成了一句空话。"""
    from jg_radar.fuse import RadioSource, Source, VisionSource

    assert isinstance(VisionSource(), Source)
    assert isinstance(RadioSource(), Source)
    assert VisionSource().kind is SourceKind.VISION
    assert RadioSource().kind is SourceKind.RADIO


# ---------------------------------------------------------------------------
# 无线电桩自身的约束
# ---------------------------------------------------------------------------
def test_radio_source_rejects_timestamps_from_the_future():
    """未来时间戳会被 max(0, ·) 钳成 0 年龄，于是那条样本**永远新鲜**——
    断流永远不会被发现。这种错必须在源头炸掉。"""
    from jg_radar.fuse import RadioReading, RadioSource

    src = RadioSource()
    with pytest.raises(ValueError, match="在未来"):
        src.feed([RadioReading("R1", (100, 200), observed_at=99.0)], now=10.0)


def test_radio_source_is_silent_until_someone_feeds_it():
    """桩不主动产出任何东西——没接解码器时它必须完全安静。"""
    from jg_radar.fuse import RadioReading, RadioSource

    src = RadioSource()
    assert list(src.poll(now=0.0)) == []
    src.feed([RadioReading("R1", (100, 200), observed_at=0.0)], now=0.0)
    out = list(src.poll(now=0.0))
    assert len(out) == 1 and out[0].source is SourceKind.RADIO
    assert list(src.poll(now=0.0)) == [], "取过一次就该清空，不能重复上报"
