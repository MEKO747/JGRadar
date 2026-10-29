"""信箱与计时埋点的契约。

信箱的语义是**反直觉**的：它丢弃数据是设计意图，不是故障。所以这里测的
重点不是"东西还在不在"，而是"**留下来的必须是最新的那一个**"——
丢错一个（留下旧的、丢掉新的）在运行时不报错、不崩溃，只会让上报的坐标
永远慢半拍，而那看起来跟"目标跑得快"一模一样。
"""

from __future__ import annotations

import threading
import time

import pytest

from jg_radar.pipeline.mailbox import Mailbox, StageAPacket
from jg_radar.pipeline.timings import Timings
from jg_radar.track.types import TrackObservation


def packet(seq: int, *, captured_at: float | None = None, n_obs: int = 0) -> StageAPacket:
    obs = tuple(
        TrackObservation(bbox=(10.0 * i, 0.0, 10.0 * i + 5, 20.0))
        for i in range(n_obs)
    )
    return StageAPacket(
        frame_seq=seq,
        captured_at=float(seq) if captured_at is None else captured_at,
        observations=obs,
    )


# ----------------------------------------------------------------------
# 信箱
# ----------------------------------------------------------------------
def test_take_returns_the_newest_and_never_the_oldest():
    """**核心契约。** 满了要挤掉**最旧的**，不是拒收新的。

    反过来做（拒收新的、留着旧的）的话，信箱里会永远卡着一个谁也没见过的
    老包，而且因为容量是满的，**再也进不来任何新东西**——整条流水线
    从此定格在最开始那几帧上，对外表现是"坐标不动了"。
    """
    box = Mailbox(capacity=2)
    for i in range(5):
        box.put(packet(i))

    assert len(box) == 2
    assert box.latest().frame_seq == 4
    taken = box.take()
    assert taken is not None and taken.frame_seq == 4
    assert len(box) == 0
    assert box.dropped == 4  # 溢出挤掉 3 个 + take 丢掉 1 个


def test_take_discards_the_backlog_instead_of_working_through_it():
    """Stage B 卡了一下之后，**不该**去补旧账。

    这是整个信箱存在的原因。若改成逐个取走，Stage B 恢复后会连着报出
    好几帧历史坐标，每一帧都又落后一点——下游看到的是一段"慢半拍的回放"，
    而且完全没有任何错误信号。
    """
    box = Mailbox(capacity=8)
    for i in range(6):
        box.put(packet(i))

    taken = box.take()
    assert taken.frame_seq == 5, "取走的必须是最新的，不是队头"
    assert len(box) == 0, "剩下的历史帧必须被丢掉，而不是排队等下一次处理"
    assert box.dropped == 5


def test_a_packet_carries_one_timestamp_for_everything_inside():
    """同刻冻结：包上只有一个时刻，包内所有观测都属于它。

    每个观测各带各的时刻的话，下游迟早会拿"最新那个观测的时间"代表整包，
    于是包的年龄被**低估**，超龄判断失效——一个早就该丢的包被发了出去。
    """
    p = packet(3, captured_at=100.0, n_obs=4)
    assert p.captured_at == 100.0
    assert len(p) == 4
    assert p.age(100.25) == pytest.approx(0.25)
    # 年龄完全由外部 now 决定，包自己不偷偷记一个"创建时间"
    assert p.age(99.0) == pytest.approx(-1.0)


def test_an_empty_mailbox_says_none_rather_than_raising():
    """没有新帧是**正常运行状态**，不是错误。"""
    box = Mailbox()
    assert box.take() is None
    assert box.latest() is None
    assert len(box) == 0


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        Mailbox(capacity=0)


def test_wait_latest_returns_none_on_timeout_instead_of_hanging():
    box = Mailbox()
    start = time.perf_counter()
    assert box.wait_latest(timeout=0.05) is None
    assert time.perf_counter() - start >= 0.04


def test_wait_latest_wakes_up_on_put():
    """线程化路径靠这个阻塞等待，不该忙等。"""
    box = Mailbox()
    got: list[StageAPacket | None] = []

    def consumer() -> None:
        got.append(box.wait_latest(timeout=2.0))

    th = threading.Thread(target=consumer)
    th.start()
    time.sleep(0.05)
    box.put(packet(7))
    th.join(timeout=2.0)

    assert not th.is_alive()
    assert got and got[0] is not None and got[0].frame_seq == 7


def test_clear_releases_waiters():
    box = Mailbox()
    th = threading.Thread(target=lambda: box.wait_latest(timeout=2.0))
    th.start()
    time.sleep(0.05)
    box.clear()
    th.join(timeout=2.0)
    assert not th.is_alive()


def test_concurrent_producers_and_a_consumer_never_corrupt_state():
    box = Mailbox(capacity=4)
    stop = threading.Event()
    seen: list[int] = []
    lock = threading.Lock()

    def producer(base: int) -> None:
        for i in range(400):
            box.put(packet(base * 1000 + i))

    def consumer() -> None:
        while not stop.is_set():
            p = box.take()
            if p is not None:
                with lock:
                    seen.append(p.frame_seq)

    threads = [threading.Thread(target=producer, args=(k,)) for k in range(3)]
    threads.append(threading.Thread(target=consumer))
    for th in threads:
        th.start()
    for th in threads[:3]:
        th.join()
    stop.set()
    threads[3].join(timeout=2.0)

    assert len(box) <= box.capacity
    # 取走的每一帧都必须**新于**它之前取走的那一帧？不——多个生产者交错时
    # 序号本就会乱。真正要保证的是：拿到的都是真实存在过的包。
    assert all(0 <= s < 3 * 1000 + 400 for s in seen)


def test_extend_counts_every_eviction():
    box = Mailbox(capacity=2)
    evicted = box.extend(packet(i) for i in range(6))
    assert evicted == 4
    assert box.latest().frame_seq == 5


def test_reset_counters_leaves_the_packets_alone():
    box = Mailbox(capacity=1)
    for i in range(3):
        box.put(packet(i))
    assert box.dropped == 2
    box.reset_counters()
    assert box.dropped == 0
    assert box.latest().frame_seq == 2


# ----------------------------------------------------------------------
# 计时
# ----------------------------------------------------------------------
def test_timings_seed_the_first_sample_instead_of_ramping_from_zero():
    """第一帧必须**就是**观测值。

    从 0 开始爬的话，启动后头十几帧的均值会被严重低估，而那正是最需要
    看真实数字的时候——你会以为预算很宽裕，直到 EMA 慢慢爬上来才发现不对。
    """
    t = Timings(alpha=0.1)
    assert t.observe("stage_a", 0.032) == pytest.approx(0.032)
    assert t.ema("stage_a") == pytest.approx(0.032)


def test_timings_ema_follows_the_configured_alpha():
    t = Timings(alpha=0.5)
    t.observe("x", 1.0)
    t.observe("x", 2.0)
    assert t.ema("x") == pytest.approx(1.5)
    t.observe("x", 2.0)
    assert t.ema("x") == pytest.approx(1.75)


def test_alpha_of_one_means_no_averaging():
    t = Timings(alpha=1.0)
    t.observe("x", 5.0)
    t.observe("x", 0.25)
    assert t.ema("x") == pytest.approx(0.25)


def test_timings_rejects_a_nonsense_alpha():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            Timings(alpha=bad)


def test_measure_records_even_when_the_block_raises():
    """卡住或抛异常的那一段**最**需要被看见。"""
    t = Timings()
    with pytest.raises(RuntimeError):
        with t.measure("boom"):
            raise RuntimeError("x")
    assert t.count("boom") == 1


def test_timings_snapshot_keys_carry_their_unit():
    """``diagnostics`` 是一个扁平数字字典——单位必须写在键名里，否则分不清。"""
    t = Timings()
    t.observe("stage_a", 0.02)
    snap = t.snapshot()
    assert set(snap) == {"t_stage_a_ms"}
    assert snap["t_stage_a_ms"] == pytest.approx(20.0)
    assert t.snapshot(unit="s")["t_stage_a_s"] == pytest.approx(0.02)


def test_timings_rejects_an_unknown_unit():
    with pytest.raises(ValueError):
        Timings().snapshot(unit="furlongs")


def test_timings_never_reports_a_negative_duration():
    """时钟回拨（或注入的假时钟乱来）不该污染 EMA。"""
    t = Timings()
    assert t.observe("x", -1.0) == 0.0


def test_last_and_count_track_what_actually_happened():
    t = Timings()
    t.observe("x", 1.0)
    t.observe("x", 3.0)
    assert t.count("x") == 2
    assert t.last("x") == pytest.approx(3.0)
    assert t.ema("x") == pytest.approx(1.0 * 0.9 + 3.0 * 0.1)


def test_reset_clears_everything():
    t = Timings()
    t.observe("x", 1.0)
    t.reset()
    assert t.names() == () and t.ema("x") == 0.0 and t.count("x") == 0


def test_the_clock_is_injectable_so_tests_do_not_sleep():
    ticks = iter([10.0, 10.25])
    t = Timings(clock=lambda: next(ticks))
    with t.measure("stage_a"):
        pass
    assert t.ema("stage_a") == pytest.approx(0.25)
