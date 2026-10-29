"""最新值缓冲的契约。

这个模块小到看起来不值得单独测，但它藏着一个**静默失效**级别的坑：
键必须是 ``(来源, 机器人名)``，不能只用机器人名。

只用名字的话，视觉和无线电同时报 ``B1`` 时后写的会把先写的覆盖掉，缓冲里
永远只剩一个源的数据。不会报错、不会崩——仲裁层只是再也看不见第二个候选，
于是"优先级"和"滞回"全部退化成"谁写得靠后就听谁的"。

这个 bug 是靠 M5 那条端到端测试（真跑跟踪器 + 中途插入无线电）抓出来的，
纯粹读代码或单测单个类都发现不了。所以在这里把它钉死。
"""

from __future__ import annotations

import threading

from jg_radar.fuse import LatestValueBuffer
from jg_radar.types import RobotState, SourceKind, SourceSample


def sample(
    name: str = "B1",
    x: int = 1000,
    y: int = 700,
    *,
    source: SourceKind = SourceKind.VISION,
) -> SourceSample:
    return SourceSample(
        source=source,
        name=name,
        position_cm=(x, y),
        state=RobotState.MEASURED,
        confidence=0.9,
        observed_at=0.0,
        revision=1,
        is_predicted=False,
    )


def radio(name: str = "B1", x: int = 1200, y: int = 800) -> SourceSample:
    return sample(name, x, y, source=SourceKind.RADIO)


def test_two_sources_reporting_the_same_robot_both_survive():
    """**回归测试。** 视觉和无线电同报 B1，两个候选都必须还在。

    这条要是挂了，仲裁层就只剩一个候选可选——优先级阶梯形同虚设，
    而整个系统不会有任何报错。
    """
    buf = LatestValueBuffer()
    buf.put(sample("B1", 1000, 700), now=0.0)
    buf.put(radio("B1", 1200, 800), now=0.0)

    assert len(buf) == 2, "同名的两个源被后者覆盖了——仲裁层将看不见两个候选"
    got = {(s.sample.source, s.sample.position_cm) for s in buf.snapshot()}
    assert got == {
        (SourceKind.VISION, (1000, 700)),
        (SourceKind.RADIO, (1200, 800)),
    }


def test_a_source_only_overwrites_its_own_slot():
    """同一个源重复写是刷新；不该动别的源的格子。"""
    buf = LatestValueBuffer()
    buf.put(sample("B1", 1000, 700), now=0.0)
    buf.put(radio("B1", 1200, 800), now=0.0)
    buf.put(sample("B1", 1010, 710), now=0.1)  # 视觉刷新

    assert len(buf) == 2
    v = buf.get(SourceKind.VISION, "B1")
    r = buf.get(SourceKind.RADIO, "B1")
    assert v is not None and v.sample.position_cm == (1010, 710)
    assert r is not None and r.sample.position_cm == (1200, 800)


def test_only_the_latest_value_is_kept():
    """不做队列。堆积的历史样本没有用处，反而会引出"追着补发旧数据"
    这种最危险的失效模式：旧坐标看起来像新坐标。"""
    buf = LatestValueBuffer()
    for i in range(10):
        buf.put(sample("B1", 1000 + i, 700), now=float(i))
    assert len(buf) == 1
    assert buf.snapshot()[0].sample.position_cm == (1009, 700)


def test_snapshot_is_a_copy():
    """返回副本：生产者不该能改掉调用方正在遍历的那个视图。"""
    buf = LatestValueBuffer()
    buf.put(sample("B1"), now=0.0)
    snap = buf.snapshot()
    buf.put(sample("R1"), now=0.0)
    assert len(snap) == 1 and len(buf) == 2


def test_clear_empties_the_buffer():
    buf = LatestValueBuffer()
    buf.put(sample("B1"), now=0.0)
    buf.put(radio("B1"), now=0.0)
    buf.clear()
    assert len(buf) == 0 and buf.snapshot() == ()
    assert buf.get(SourceKind.VISION, "B1") is None


def test_concurrent_producers_do_not_corrupt_the_buffer():
    """视觉线程和无线电线程真的会同时写。锁不是为了好看。"""
    buf = LatestValueBuffer()
    names = [f"B{i}" for i in range(1, 7)]

    def worker(source: SourceKind) -> None:
        for _ in range(200):
            for n in names:
                buf.put(sample(n, source=source), now=0.0)

    threads = [
        threading.Thread(target=worker, args=(k,))
        for k in (SourceKind.VISION, SourceKind.RADIO)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert len(buf) == 2 * len(names)
    for n in names:
        assert buf.get(SourceKind.VISION, n) is not None
        assert buf.get(SourceKind.RADIO, n) is not None
