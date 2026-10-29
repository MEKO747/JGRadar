"""四态状态机 —— 方案 §九 测试 8 的那条降级链。

这条链是雷达站最容易悄悄出错的地方：状态错了不会报错、不会崩，只会让裁判
系统拿到一个"看起来正常"的坐标。所以这里既测纯函数（每条分支单独测），
也测整条链路（真的走一遍 MEASURED→HOLD→BLIND→MISSING）。
"""

from __future__ import annotations

import pytest

from jg_radar.config import load_config
from jg_radar.track import MultiTracker, StateThresholds, TrackObservation, resolve_state
from jg_radar.types import RobotState

TH = StateThresholds(lost_predict_s=0.8, ttl_s=3.0, reacquire_hits=2)

# 20fps，与方案 §十一 的 50ms 预算一致
DT = 0.05


def obs(x: float, y: float, world=None, *, cls: str = "B1", digit: str = "1",
        w: float = 80.0, h: float = 40.0, conf: float = 0.9) -> TrackObservation:
    return TrackObservation(
        bbox=(x - w / 2, y - h / 2, x + w / 2, y + h / 2),
        world_cm=world,
        confidence=conf,
        car_class=cls,
        digit=digit,
    )


# ======================================================================
# 纯函数：每条分支单独锁死
# ======================================================================
def test_measured_requires_a_match():
    st = resolve_state(
        matched=True, age_s=0.0, previous=None, consecutive_hits=1,
        has_filter=True, has_waypoint=True, thresholds=TH,
    )
    assert st is RobotState.MEASURED


def test_matched_without_a_previous_blind_state_promotes_immediately():
    """新轨迹第一帧就该是 MEASURED。

    重捕确认只防"从盲区被单帧假检测拽回来"，不该拖慢一条正常轨迹的起步——
    否则每条新轨迹都要先当两帧预测，白白丢掉 100ms。
    """
    for prev in (None, RobotState.MEASURED, RobotState.OCCLUSION_HOLD):
        st = resolve_state(
            matched=True, age_s=0.0, previous=prev, consecutive_hits=1,
            has_filter=True, has_waypoint=True, thresholds=TH,
        )
        assert st is RobotState.MEASURED, prev


def test_short_loss_holds():
    st = resolve_state(
        matched=False, age_s=0.5, previous=RobotState.MEASURED, consecutive_hits=0,
        has_filter=True, has_waypoint=True, thresholds=TH,
    )
    assert st is RobotState.OCCLUSION_HOLD


def test_long_loss_goes_blind_only_where_there_is_a_waypoint():
    common = dict(
        matched=False, age_s=1.5, previous=RobotState.OCCLUSION_HOLD,
        consecutive_hits=0, has_filter=True, thresholds=TH,
    )
    assert resolve_state(has_waypoint=True, **common) is RobotState.BLIND_PREDICTION
    assert resolve_state(has_waypoint=False, **common) is RobotState.MISSING


def test_past_ttl_is_missing_even_with_waypoints():
    st = resolve_state(
        matched=False, age_s=3.5, previous=RobotState.BLIND_PREDICTION, consecutive_hits=0,
        has_filter=True, has_waypoint=True, thresholds=TH,
    )
    assert st is RobotState.MISSING


def test_track_without_a_world_filter_is_missing_not_hold():
    """从来没有过场地坐标的轨迹没有任何可外推的东西，不能报 OCCLUSION_HOLD。"""
    st = resolve_state(
        matched=False, age_s=0.1, previous=RobotState.MEASURED, consecutive_hits=0,
        has_filter=False, has_waypoint=True, thresholds=TH,
    )
    assert st is RobotState.MISSING


def test_reacquire_from_blind_needs_consecutive_hits():
    """盲区里一帧假检测不该把状态翻回 MEASURED。"""
    st = resolve_state(
        matched=True, age_s=2.0, previous=RobotState.BLIND_PREDICTION,
        consecutive_hits=1, has_filter=True, has_waypoint=True, thresholds=TH,
    )
    assert st is RobotState.BLIND_PREDICTION

    st = resolve_state(
        matched=True, age_s=2.0, previous=RobotState.BLIND_PREDICTION,
        consecutive_hits=TH.reacquire_hits, has_filter=True, has_waypoint=True,
        thresholds=TH,
    )
    assert st is RobotState.MEASURED


def test_thresholds_reject_a_ttl_that_would_skip_blind():
    with pytest.raises(ValueError, match="BLIND_PREDICTION"):
        StateThresholds(lost_predict_s=1.0, ttl_s=0.5)


def test_reacquire_hits_must_be_at_least_one():
    with pytest.raises(ValueError, match="reacquire_hits"):
        StateThresholds(reacquire_hits=0)


# ======================================================================
# 整条链路
# ======================================================================
def _tracker(cfg=None) -> MultiTracker:
    return MultiTracker((cfg or load_config("config/default.yaml")).track)


def _run(tracker, frames, start=0.0):
    """跑一串观测，返回 ``[(时刻, {槽位: 状态})]``。"""
    out = []
    now = start
    for f in frames:
        res = tracker.step(f, now=now)
        out.append((now, {k: v.state for k, v in res.items()}))
        now += DT
    return out


def test_the_degradation_ladder_walks_in_order():
    """空检测 → MEASURED → OCCLUSION_HOLD → BLIND_PREDICTION → MISSING → 消失。

    "消失"是设计的一部分：轨迹越过 ttl 后被回收，此后由发布层把该槽位填成
    MISSING（见 ``MultiTracker._reap``）。这里断言它**先报过一次 MISSING**
    再消失，而不是无声无息地没了。
    """
    tracker = _tracker()
    for i in range(10):
        tracker.step([obs(500.0 + i * 3, 600.0, (500.0 + i * 3, 750.0))], now=i * DT)

    states = []
    now = 10 * DT
    for _ in range(80):
        res = tracker.step([], now=now)
        states.append(res["B1"].state if "B1" in res else None)
        now += DT

    seen: list[RobotState | None] = []
    for s in states:
        if not seen or seen[-1] != s:
            seen.append(s)

    assert seen == [
        RobotState.OCCLUSION_HOLD,
        RobotState.BLIND_PREDICTION,
        RobotState.MISSING,
        None,
    ], f"降级链不是单调的：{seen}"


def test_blind_prediction_reports_the_waypoint_not_a_straight_line():
    """盲区坐标必须落在某个航点上，不能是恒速外推的结果。

    恒速外推 3 秒 = 编造（见 ``blind.py``）。这里直接检查坐标就是配置里
    某个航点的坐标。
    """
    tracker = _tracker()
    for i in range(10):
        tracker.step([obs(500.0 + i * 3, 600.0, (500.0 + i * 3, 750.0))], now=i * DT)

    now = 10 * DT
    blind = None
    for _ in range(60):
        res = tracker.step([], now=now)
        now += DT
        if res.get("B1") and res["B1"].state is RobotState.BLIND_PREDICTION:
            blind = res["B1"]
            break

    assert blind is not None, "根本没走到盲区预测"
    waypoints = {tuple(int(c) for c in w.position_cm) for w in tracker.waypoints}
    assert blind.position_cm in waypoints, (
        f"盲区坐标 {blind.position_cm} 不在航点集 {waypoints} 里——"
        "说明用上了滤波器外推"
    )
    assert blind.is_predicted


def test_state_never_oscillates_during_a_clean_loss():
    """一次丢失期间状态只能单调降级，不许来回跳。

    每帧重算状态是必然的（时间在走），所以抖动只可能来自判据本身不单调。
    这条测试把"状态序列里同一个状态出现两次"当作失败。
    """
    tracker = _tracker()
    for i in range(10):
        tracker.step([obs(500.0, 600.0, (500.0, 750.0))], now=i * DT)

    now = 10 * DT
    seq = []
    for _ in range(80):
        res = tracker.step([], now=now)
        now += DT
        seq.append(res["B1"].state.value if "B1" in res else "absent")

    collapsed = [s for i, s in enumerate(seq) if i == 0 or seq[i - 1] != s]
    assert len(collapsed) == len(set(collapsed)), f"状态抖动：{collapsed}"


def test_a_single_spurious_detection_does_not_pull_a_blind_track_back():
    """盲区里出现一帧孤立检测：状态仍是 BLIND_PREDICTION，且坐标仍是航点。

    这是"预测被当成测量"的镜像失效——把一次不可信的检测当成"重新看到了"。
    确认计数没攒够之前，对外一律还是预测态。
    """
    tracker = _tracker()
    for i in range(10):
        tracker.step([obs(500.0, 600.0, (500.0, 750.0))], now=i * DT)

    now = 10 * DT
    for _ in range(30):  # 1.5s，进入盲区
        res = tracker.step([], now=now)
        now += DT
    assert res["B1"].state is RobotState.BLIND_PREDICTION

    # 在预测框附近放一帧检测：能动到状态机的只有它
    res = tracker.step([obs(500.0, 600.0, (500.0, 750.0))], now=now)
    assert "B1" in res, "轨迹被整体丢了，这条测试就没意义了"

    # 先确认这一帧检测**真的被配上了**。否则状态停在 BLIND_PREDICTION 只是
    # 因为门限把它拒了，测试就成了空转——测不到重捕确认这条规则。
    assert tracker.last_diagnostics["matched"] == 1, "检测被门限拒了，测的不是重捕规则"
    assert tracker.tracks[0].consecutive_hits == 1

    assert res["B1"].state is RobotState.BLIND_PREDICTION, (
        "单帧检测把盲区轨迹拽回了 MEASURED"
    )
    assert res["B1"].is_predicted


def test_consecutive_hits_do_reacquire():
    """连上够多次就该回到 MEASURED——确认门不能变成永久锁死。"""
    tracker = _tracker()
    for i in range(10):
        tracker.step([obs(500.0, 600.0, (500.0, 750.0))], now=i * DT)

    now = 10 * DT
    for _ in range(30):
        tracker.step([], now=now)
        now += DT

    for i in range(TH.reacquire_hits):
        res = tracker.step([obs(500.0, 600.0, (500.0, 750.0))], now=now)
        now += DT
    assert res["B1"].state is RobotState.MEASURED


def test_without_waypoints_a_long_loss_is_missing_not_blind():
    """没配航点时不许报 BLIND_PREDICTION —— 没有候选就先说不知道。"""
    cfg = load_config("config/default.yaml")
    import dataclasses

    track_cfg = dataclasses.replace(cfg.track, blind_waypoints=())
    tracker = MultiTracker(track_cfg)

    for i in range(10):
        tracker.step([obs(500.0, 600.0, (500.0, 750.0))], now=i * DT)

    now = 10 * DT
    states = set()
    for _ in range(40):  # 走到 0.8s 之后
        res = tracker.step([], now=now)
        now += DT
        if "B1" in res:
            states.add(res["B1"].state)
    assert RobotState.BLIND_PREDICTION not in states
    assert RobotState.MISSING in states


# ======================================================================
# 契约
# ======================================================================
def test_every_published_slot_satisfies_the_contract():
    """整条链路的输出都要过 ``RobotSlotOutput.validate()``。

    流水线里已经调了一次；这里再走一遍，是为了让契约违反在**测试**里炸，
    而不是在赛场上炸。
    """
    tracker = _tracker()
    now = 0.0
    for i in range(40):
        obs_list = (
            [obs(500.0 + i * 2, 600.0, (500.0 + i * 2, 750.0))]
            if i % 7 != 5 else []
        )
        for slot in tracker.step(obs_list, now=now).values():
            slot.validate()
        now += DT


def test_observed_at_stops_advancing_while_predicting():
    """预测态的 ``observed_at`` 必须停住。

    下游就是靠 ``published_at - observed_at`` 判断信息新鲜度的。如果预测时
    它跟着 now 走，一条已经丢了 2 秒的坐标看起来会"刚刚更新过"。
    """
    tracker = _tracker()
    for i in range(10):
        tracker.step([obs(500.0, 600.0, (500.0, 750.0))], now=i * DT)

    now = 10 * DT
    stamped = []
    for _ in range(30):
        res = tracker.step([], now=now)
        if "B1" in res:
            stamped.append((now, res["B1"].observed_at, res["B1"].state))
        now += DT

    assert stamped
    for now_v, observed, state in stamped:
        assert state.is_prediction
        assert observed < now_v
    # 而且它一直停在最后一次真观测的时刻上
    assert len({o for _, o, _ in stamped}) == 1


def test_determinism_across_identical_runs():
    """同样的输入序列跑两遍必须逐字段相同——回放可复现的前提。"""
    def run():
        tracker = _tracker()
        now = 0.0
        frames = []
        for i in range(60):
            dets = []
            if i % 11 != 3:
                dets.append(obs(400.0 + i * 1.5, 500.0 + i * 0.5,
                                (400.0 + i * 1.5, 700.0 + i * 0.5)))
            if i % 5 == 0:
                dets.append(obs(900.0, 300.0, (1500.0, 900.0), cls="R3", digit="3"))
            res = tracker.step(dets, now=now)
            frames.append({k: v.to_dict() for k, v in res.items()})
            now += DT
        return frames

    assert run() == run()
