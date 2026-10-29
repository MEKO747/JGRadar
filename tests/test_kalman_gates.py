"""滤波器与双闸门 —— 方案 §九 测试 5。

核心命题只有一句：**坏检测污染不了轨迹**。这里把它拆成三层来验：

1. 滤波器自身数值正确（中心/尺寸的参数化、Joseph 形式、协方差单调发散）
2. 闸门判据正确（马氏距离、原始跳变，以及两者**互补**）
3. 闸门在链路上真的拦住了东西（而且拦下之后轨迹继续活着，不是一起完蛋）
"""

from __future__ import annotations

import numpy as np
import pytest

from jg_radar.config import load_config
from jg_radar.track import (
    BoxKalman,
    MultiTracker,
    TrackObservation,
    WorldKalman,
    clamp_dt,
    gate_mask,
    raw_jump_matrix,
)

DT = 0.05


def obs(x, y, world=None, *, cls="B1", w=80.0, h=40.0):
    return TrackObservation(
        bbox=(x - w / 2, y - h / 2, x + w / 2, y + h / 2),
        world_cm=world, confidence=0.9, car_class=cls,
    )


# ======================================================================
# dt 钳制
# ======================================================================
def test_clamp_dt_keeps_the_interval_usable():
    assert clamp_dt(-1.0) == pytest.approx(1.0 / 120.0)
    assert clamp_dt(0.0) == pytest.approx(1.0 / 120.0)
    assert clamp_dt(float("nan")) == pytest.approx(1.0 / 120.0)
    assert clamp_dt(10.0) == pytest.approx(0.5)
    assert clamp_dt(0.05) == pytest.approx(0.05)


def test_zero_dt_does_not_invent_elapsed_time():
    """没有经过物理时间时不引入位移或过程噪声。重复观测由调用方拒绝。"""
    kf = BoxKalman((0.0, 0.0, 10.0, 10.0))
    before = float(np.trace(kf.covariance))
    kf.predict(0.0)
    assert float(np.trace(kf.covariance)) == before


# ======================================================================
# 滤波器
# ======================================================================
def test_box_kalman_round_trips_the_bbox_convention():
    """状态是 ``(cx,cy,w,h)``，对外是 ``(x1,y1,x2,y2)``。转换错一次就会
    让所有关联距离变成垃圾，而且不会报错。"""
    bbox = (100.0, 200.0, 180.0, 240.0)
    kf = BoxKalman(bbox)
    got = kf.bbox
    assert got == pytest.approx(bbox)
    assert kf.center == pytest.approx((140.0, 220.0))
    assert kf.size == pytest.approx((80.0, 40.0))


def test_box_kalman_converges_to_a_static_target():
    bbox = (100.0, 200.0, 180.0, 240.0)
    kf = BoxKalman(bbox)
    for _ in range(30):
        kf.predict(DT)
        kf.update_bbox(bbox)
    assert kf.bbox == pytest.approx(bbox, abs=0.5)
    assert kf.velocity == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1.0)


def test_box_kalman_learns_velocity_and_predicts_ahead():
    """恒速目标：外推之后框应该**往前走**，而不是原地不动。

    原地不动的框和真实目标的 IoU 会随遮挡时间迅速掉到 0.5 以下，一恢复检测
    就可能配错——这正是要速度状态的原因。
    """
    kf = BoxKalman((100.0, 200.0, 180.0, 240.0))
    for i in range(0, 20):
        x = 100.0 + i * 6.0
        kf.predict(DT)
        kf.update_bbox((x, 200.0, x + 80.0, 240.0))

    x_before = kf.bbox[0]
    kf.predict(0.5)
    x_after = kf.bbox[0]
    assert x_after > x_before + 40.0, "外推没有跟着速度走"


def test_joseph_form_keeps_the_covariance_positive_definite():
    """``P = (I−KH)P`` 那套在增益接近 1 时会丢掉正定性。

    这里做 500 次「预测 + 更新」，再用同一个观测量做 500 次「预测 + 极低噪声
    更新」，逼增益接近 1。协方差一旦出现负特征值，后面的马氏距离就是负数，
    门限全乱——而且不会抛异常，只会静默地把坏检测放进来。
    """
    kf = BoxKalman((100.0, 200.0, 180.0, 240.0), measurement_noise=1e-3)
    rng = np.random.default_rng(0)
    for _ in range(500):
        kf.predict(DT)
        b = (100.0 + rng.normal(0, 0.5), 200.0, 180.0, 240.0)
        kf.update_bbox(b)
        P = kf.covariance
        assert np.allclose(P, P.T, atol=1e-9), "协方差不对称"
        assert np.linalg.eigvalsh(P).min() > 0, "协方差失去正定性"


def test_mahalanobis_grows_as_the_prediction_ages():
    """丢了越久，同一个检测的马氏距离就越大——闸门才会"越丢越谨慎"。"""
    kf = BoxKalman((100.0, 200.0, 180.0, 240.0))
    for _ in range(10):
        kf.predict(DT)
        kf.update_bbox((100.0, 200.0, 180.0, 240.0))

    far = (140.0, 200.0, 220.0, 240.0)  # 偏了 40px
    d_short = kf.mahalanobis_bbox(far)
    kf.predict(1.0)
    d_long = kf.mahalanobis_bbox(far)
    assert d_long < d_short, (
        "外推之后协方差应该变大、同一个残差的马氏距离应该变小；"
        "结果是变大，说明预测没有把不确定性算进去"
    )


def test_world_kalman_position_std_grows_while_coasting():
    """``position_std_cm`` 是下游判断"这坐标还能不能信"的唯一依据。

    它必须随遮挡时长增长。一个恒定的标准差等于没有信息，下游会一直以为
    这条坐标和刚测到的一样准。
    """
    kf = WorldKalman((500.0, 750.0))
    for _ in range(10):
        kf.predict(DT)
        kf.update((500.0, 750.0))
    std_fresh = kf.position_std_cm

    for _ in range(40):  # 2 秒不更新
        kf.predict(DT)
    std_stale = kf.position_std_cm

    assert std_stale > std_fresh * 3.0, (
        f"2 秒没观测，标准差只从 {std_fresh:.2f} 涨到 {std_stale:.2f}"
    )


def test_world_kalman_velocity_comes_out_in_cm_per_second():
    kf = WorldKalman((500.0, 750.0))
    x = 500.0
    for _ in range(40):
        kf.predict(DT)
        x += 300.0 * DT  # 300 cm/s
        kf.update((x, 750.0))
    vx, vy = kf.velocity_cm_s
    assert vx == pytest.approx(300.0, rel=0.15)
    assert vy == pytest.approx(0.0, abs=20.0)


def test_measurement_noise_must_be_positive():
    with pytest.raises(ValueError, match="measurement_noise"):
        BoxKalman((0.0, 0.0, 10.0, 10.0), measurement_noise=0.0)


# ======================================================================
# 闸门判据
# ======================================================================
def test_raw_jump_is_the_distance_between_observations():
    last = np.array([[100.0, 100.0]])
    det = np.array([[100.0, 100.0], [103.0, 104.0], [200.0, 100.0]])
    d = raw_jump_matrix(last, det)
    assert d[0, 0] == pytest.approx(0.0)
    assert d[0, 1] == pytest.approx(5.0)
    assert d[0, 2] == pytest.approx(100.0)


def test_raw_jump_treats_a_never_observed_track_as_no_jump():
    """新轨迹没有"上一次观测"，跳变门对它无从谈起。

    这里给 0 而不是 ``inf``：``inf`` 会把新轨迹的**所有**配对都拒掉，
    新轨迹就永远配不上任何东西，只能靠新建——同一辆车会被反复建成新轨迹。
    """
    last = np.array([[np.nan, np.nan]])
    det = np.array([[999.0, 999.0]])
    assert raw_jump_matrix(last, det)[0, 0] == pytest.approx(0.0)


def test_gates_are_conjunctive():
    raw = np.array([[10.0, 500.0]])
    chi2 = np.array([[1.0, 50.0]])
    same = np.array([[True, False]])
    ok = gate_mask(raw_jump=raw, raw_jump_max=120.0,
                   chi2=chi2, chi2_max=9.49, same_label=same)
    assert ok.tolist() == [[True, False]]
    # 第二个配对同时踩了三个门


def test_a_nan_chi2_is_rejected_not_treated_as_pass():
    """算不出马氏距离（协方差非正定）时必须拒。当成"合格"是最坏的选择。"""
    chi2 = np.array([[np.nan, 1.0]])
    ok = gate_mask(chi2=chi2, chi2_max=9.49)
    assert ok.tolist() == [[False, True]]


def test_gate_mask_needs_at_least_one_matrix():
    with pytest.raises(ValueError, match="至少"):
        gate_mask()


# ======================================================================
# 链路上真的拦得住
# ======================================================================
def _tracker():
    return MultiTracker(load_config("config/default.yaml").track)


def _establish(tracker, *, x=500.0, y=600.0, n=12, world=(500.0, 750.0), start=0.0):
    now = start
    for _ in range(n):
        tracker.step([obs(x, y, world)], now=now)
        now += DT
    return now


def test_a_teleporting_detection_is_rejected():
    tracker = _tracker()
    now = _establish(tracker)

    far = obs(900.0, 600.0, (1800.0, 750.0))  # 跳了 400px，远超 raw_jump_px=120
    res = tracker.step([far], now=now)

    assert tracker.last_diagnostics["matched"] == 0, "跳变检测被配上了"
    assert tracker.tracks[0].misses == 1
    assert res["B1"].state.value == "occlusion_hold", (
        "拒了检测之后轨迹应该走遮挡保持，而不是继续报 MEASURED"
    )


def test_the_track_survives_a_rejected_detection():
    """拦掉坏检测之后轨迹必须还活着——否则"拦"和"丢"就没有区别了。

    而且被拒的检测会开出一条**候选**轨迹（未确认），它绝不能把原来那条
    挤下槽位：字典按槽位名索引，谁后写谁赢。
    """
    tracker = _tracker()
    now = _establish(tracker)
    track_id = tracker.tracks[0].track_id

    out = tracker.step([obs(900.0, 600.0, (1800.0, 750.0))], now=now)
    now += DT

    assert out["B1"].track_id == track_id, "槽位被新冒出来的候选轨迹抢走了"
    assert tracker.tracks[0].hits == 12  # 坏检测没有计入命中

    # 候选轨迹下一帧没再被看到，就该消失——一帧假检测只活一帧
    tracker.step([], now=now)
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].track_id == track_id, "轨迹被重建了，等于丢了再新建"


def test_a_one_frame_false_positive_is_never_published():
    """孤立一帧的假检测连一次都不该对外报出去。"""
    tracker = _tracker()
    now = 0.0
    tracker.step([obs(900.0, 600.0, (1800.0, 750.0), cls="R3")], now=now)
    assert "R3" not in tracker.step([], now=now + DT), (
        "单帧假检测形成了对外输出"
    )


def test_a_persistent_detection_is_confirmed_and_then_published():
    """确认门不能把真的也挡在外面：连续出现两帧就该上报。"""
    tracker = _tracker()
    now = 0.0
    first = tracker.step([obs(900.0, 600.0, (1800.0, 750.0), cls="R3")], now=now)
    assert "R3" not in first, "第一帧就上报了，确认门没生效"

    now += DT
    second = tracker.step([obs(900.0, 600.0, (1800.0, 750.0), cls="R3")], now=now)
    assert "R3" in second
    assert second["R3"].state.value == "measured"


def test_raw_jump_gate_uses_the_last_observation_not_the_prediction():
    """这是双闸门里最容易被写错的一处。

    如果跳变门拿**预测框**去比，滤波器被缓慢带偏时预测框会跟着偏，跳变看起来
    就"不大"，门形同虚设。用**上一次真实观测**去比才拦得住。

    构造：目标一路向右跑，丢失 1 秒让预测冲出去，然后在一个"预测位置附近、
    但离上次观测很远"的地方放一个检测。残差门满意，跳变门不满意。
    """
    tracker = _tracker()
    now = 0.0
    x = 500.0
    speed_px_s = 200.0
    for _ in range(15):
        tracker.step([obs(x, 600.0, (500.0, 750.0))], now=now)
        x += speed_px_s * DT
        now += DT

    last_center = tracker.tracks[0].last_center
    for _ in range(20):  # 1 秒不更新，预测跑到前面去了
        tracker.step([], now=now)
        now += DT

    predicted = tracker.tracks[0].predicted_bbox
    pred_cx = (predicted[0] + predicted[2]) * 0.5
    jump_from_pred = abs(pred_cx - last_center[0])
    assert jump_from_pred > 120.0, (
        f"构造失败：预测只离开了 {jump_from_pred:.0f}px，还不到跳变门限"
    )

    # 把检测放在预测位置上：残差门会满意，只有跳变门能拦
    res = tracker.step([obs(pred_cx, 600.0, (700.0, 750.0))], now=now)
    assert tracker.last_diagnostics["matched"] == 0, (
        "检测落在预测位置就被放行了——跳变门用的是预测框，不是上次观测"
    )
    assert "B1" in res


def test_a_detection_without_a_projection_never_enters_tracking():
    """投不到场地上的检测不进关联——它会抢走真实轨迹的配对。"""
    tracker = _tracker()
    now = _establish(tracker)
    res = tracker.step([obs(505.0, 600.0, None)], now=now)
    assert tracker.last_diagnostics["dropped_no_world"] == 1
    assert tracker.last_diagnostics["matched"] == 0
    assert tracker.tracks[0].misses == 1
    assert res["B1"].state.value == "occlusion_hold"


def test_two_targets_do_not_swap_identities():
    """两个目标交叉而过时不许换身份——换身份 = 报错人，比漏报更糟。"""
    tracker = _tracker()
    now = 0.0
    for i in range(20):
        a = 400.0 + i * 12.0
        b = 900.0 - i * 12.0
        tracker.step(
            [obs(a, 600.0, (400.0 + i * 12.0, 500.0)),
             obs(b, 600.0, (900.0 - i * 12.0, 1000.0), cls="R3")],
            now=now,
        )
        now += DT

    # 交叉之后各自继续
    for i in range(20):
        a = 640.0 + i * 12.0
        b = 660.0 - i * 12.0
        tracker.step(
            [obs(a, 600.0, (640.0 + i * 12.0, 500.0)),
             obs(b, 600.0, (660.0 - i * 12.0, 1000.0), cls="R3")],
            now=now,
        )
        now += DT

    by_label = {t.label: t for t in tracker.tracks}
    assert set(by_label) == {"B1", "R3"}
    # B1 属于场地 y=500 那条轨迹，R3 属于 y=1000 那条
    assert by_label["B1"].world.position_cm[1] == pytest.approx(500.0, abs=60.0)
    assert by_label["R3"].world.position_cm[1] == pytest.approx(1000.0, abs=60.0)


# ======================================================================
# 过程噪声：滤波器不许把自己那条有效观测拒掉
# ======================================================================
def _worst_chi2_under_constant_acceleration(q: float, accel_px_s2: float,
                                            *, seconds: float = 3.0) -> float:
    """无噪声、恒加速目标的 χ² 最大值。

    没有噪声 ⇒ 残差**全部**来自系统性滞后，不是随机涨落。所以这个最大值
    就是"滤波器会不会把自己的有效观测判成离群点"。带噪声的版本测不了这个：
    χ² 门本来就是 95% 分位，几千个样本里总有几个合格观测天然超过它，
    最大值必然超标，那是门在正常工作，不是失效。
    """
    kf = BoxKalman((100.0, 100.0, 160.0, 140.0), process_noise=q,
                   measurement_noise=9.0)
    worst = 0.0
    for k in range(1, int(seconds / DT) + 1):
        t = k * DT
        cx = 130.0 + 0.5 * accel_px_s2 * t * t
        box = (cx - 30.0, 100.0, cx + 30.0, 140.0)
        kf.predict(DT)
        worst = max(worst, kf.mahalanobis_bbox(box))
        kf.update_bbox(box)
    return worst


def test_the_box_filter_never_rejects_its_own_valid_measurements():
    """**核心契约。** 一个在机动的目标，它的**有效**观测必须留在门内。

    过程噪声配小了，滤波器会过度相信自己的速度：状态卡在离真值一段距离的
    地方（而观测每帧都在喂它）、协方差同时越收越紧，于是 χ² 门把**自己的
    有效观测**拒掉。拒掉的后果不是"少了一帧"，是**永久分叉**——预测只会
    越飘越远，这条轨迹再也配不上任何观测，同时另起一条重复轨迹。对外表现
    是目标每一帧都看得见、雷达却一路报 HOLD/BLIND。

    端到端回放里完整复现过：B3 连续可见，第 48 帧断轨。

    ``accel_px_s2`` 取 395 ≈ 5 m/s²（机器人横向机动的量级，按 12MP+12mm 在
    11m 处的 1px≈1.27cm 折算）。**这个门槛是按真实机动定的，不是按 fixture
    定的**——fixture 的正弦机动只有约 0.3 m/s²，按它调出来的过程噪声在实车
    上照样会锁死（q=64 时下面这条就是 χ²=15，一样拒）。
    """
    gate = float(load_config("config/default.yaml").track.gate_chi2)
    q = float(load_config("config/default.yaml").track.box_process_noise_px2)

    worst = _worst_chi2_under_constant_acceleration(q, 395.0)
    assert worst < gate, (
        f"过程噪声 q={q} 跟不上 5 m/s² 的机动：χ² 峰值 {worst:.1f} > 门限 {gate}，"
        "滤波器会把自己的有效观测拒掉，轨迹随后永久分叉"
    )


def test_a_gentle_maneuver_is_handled_with_a_wide_margin():
    """轻机动（0.4 m/s² 量级）下必须留出很大的余量，而不是刚好卡在门限上。"""
    cfg = load_config("config/default.yaml").track
    worst = _worst_chi2_under_constant_acceleration(
        float(cfg.box_process_noise_px2), 30.0
    )
    assert worst < float(cfg.gate_chi2) / 3.0, f"余量不足：χ²={worst:.2f}"


def test_the_process_noise_is_not_so_small_that_filtering_makes_it_worse():
    """过程噪声小到一定程度，滤波**反而有害**：输出比原始测量还抖。

    这不是"效果差一点"，是滤波器已经发散了——它在用自己那个错误的速度
    外推，每帧只修正一点点。
    """
    rng = np.random.default_rng(0)
    q = float(load_config("config/default.yaml").track.box_process_noise_px2)
    sigma = 3.0
    filt_err, raw_err = [], []
    n = int(6.0 / DT)
    for _ in range(20):
        kf = BoxKalman((100.0, 100.0, 160.0, 140.0), process_noise=q,
                       measurement_noise=sigma**2)
        for k in range(1, n + 1):
            t = k * DT
            cx = 130.0 + 0.5 * 30.0 * t * t
            meas = cx + rng.normal(0.0, sigma)
            kf.predict(DT)
            kf.update_bbox((meas - 30.0, 100.0, meas + 30.0, 140.0))
            if k > n // 2:
                raw_err.append(meas - cx)
                filt_err.append((kf.bbox[0] + kf.bbox[2]) / 2.0 - cx)

    gain = float(np.std(raw_err)) / float(np.std(filt_err))
    assert gain > 1.2, (
        f"滤波后的抖动是原始测量的 {1/gain:.2f} 倍——过程噪声太小，滤波器已发散"
    )
