"""drop / send_stale 的对比：同一份观测，两条策略只差断流那几帧怎么填。

这里的断言全部是**方向性**的——不断言具体秒数，因为秒数取决于场景假设
（延迟、噪声、速度）。具体的数字由 `tools/marking_sim.py` 跑出来进报告，
测试只负责把「什么时候该谁赢」这条物理关系钉住。
"""

from __future__ import annotations

import pytest

from jg_radar.eval.marking_scenario import (
    Scenario,
    StalePolicy,
    compare,
    distance_m,
    run,
    run_ideal,
)


def test_measurements_are_shared_by_both_policies():
    """同一段输入：两个策略必须吃同一串观测，否则对比没有意义。"""
    scenario = Scenario(outage_s=1.0)
    first = scenario.measurements()
    second = scenario.measurements()  # 固定种子 → 完全可复算
    assert first == second
    outage = [item for item in first if item[1] is None]
    assert outage, "断流窗口里应该没有新数据"


def test_stationary_target_send_stale_keeps_climbing():
    """目标没动时，旧坐标仍然准确 → 照发旧值让 x 继续累积。"""
    scenario = Scenario(speed_mps=0.0, outage_s=2.0, outage_start_s=3.0)
    results = compare(scenario)
    stale = results[StalePolicy.SEND_STALE]
    drop = results[StalePolicy.DROP]
    outage_end = scenario.outage_start_s + scenario.outage_s
    assert stale.p_at(outage_end) > drop.p_at(outage_end)
    # 站着不动的目标，断流期间照发旧值不应出现任何「错误」判定
    assert stale.counts()["error"] == 0
    assert drop.counts()["interrupt"] >= 4  # 2 秒断流 = 至少 4 次中断判定


def test_fast_target_drop_wins():
    """目标 3 m/s、断流 2 秒 → 漂移 6 米，旧坐标早就判错了。"""
    scenario = Scenario(speed_mps=3.0, outage_s=2.0, outage_start_s=3.0)
    results = compare(scenario)
    outage_end = scenario.outage_start_s + scenario.outage_s
    assert results[StalePolicy.DROP].p_at(outage_end) > results[StalePolicy.SEND_STALE].p_at(outage_end)
    assert results[StalePolicy.SEND_STALE].counts()["error"] > 0


def test_error_rate_is_higher_for_send_stale_than_for_drop():
    """判错的代价：照发旧值是每个发送周期判一次，drop 是每 0.5 秒判一次。"""
    scenario = Scenario(speed_mps=3.0, outage_s=2.0, outage_start_s=3.0)
    results = compare(scenario)
    stale = results[StalePolicy.SEND_STALE]
    drop = results[StalePolicy.DROP]
    # 2 秒断流：drop 补 4 次中断；send_stale 按 4.7Hz 判 9 次左右
    assert stale.counts()["error"] >= 2 * drop.counts()["interrupt"]


def test_slow_drift_still_favours_send_stale():
    """漂移在 1.6 米以内时，旧坐标至少是半准确 → 仍然比不发划算。"""
    scenario = Scenario(speed_mps=1.0, outage_s=0.4, outage_start_s=3.0)
    results = compare(scenario)
    outage_end = scenario.outage_start_s + scenario.outage_s
    assert results[StalePolicy.SEND_STALE].p_at(outage_end) >= results[StalePolicy.DROP].p_at(outage_end)


def test_ideal_has_no_errors_and_reaches_its_thresholds_earliest():
    scenario = Scenario(speed_mps=2.0, outage_s=1.0, outage_start_s=3.0)
    ideal = run_ideal(scenario)
    assert ideal.counts()["error"] == 0
    assert ideal.counts()["interrupt"] == 0
    t100 = ideal.first_time_at(100.0)
    assert t100 is not None
    # 4.7Hz 冷启动：第 0 秒就有第一拍，P=n(n+1)/2 ≥ 100 需要 14 拍 → 13 个周期
    assert t100 == pytest.approx(13 / scenario.send_hz, abs=1e-6)


def test_drop_has_exactly_one_interrupt_per_half_second():
    scenario = Scenario(send_hz=4.7, outage_s=1.0, outage_start_s=3.0, speed_mps=0.0)
    result = run(scenario, StalePolicy.DROP)
    assert result.counts()["interrupt"] == 2


def test_distance_helper_is_metres():
    assert distance_m((0.0, 0.0), (0.3, 0.4)) == pytest.approx(0.5)
