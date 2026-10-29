"""事件循环：把帧源、Stage A、Stage B 串起来跑。

两种跑法，**默认是单线程的那一种**：

``EventLoop``
    一帧接一帧，全部在调用线程里。回放与测试走这条。给定同一份输入，
    输出逐字节可复现——因为循环里没有任何并发，也没有读墙上时钟。

``ThreadedEventLoop``
    实时部署走这条：Stage A 在后台线程里跑检测，Stage B 在主线程里
    按节奏发布，中间隔一个信箱。Stage A 偶尔超时不会让上报卡住——
    信箱会把积压的旧帧丢掉，Stage B 永远只发最新的那一帧。

## 为什么默认是单线程

因为**确定性比吞吐重要**，至少在这个阶段是这样。并发会引入"同一份输入
两次跑出不同结果"，于是任何金标准比对都失去意义，任何"这一帧为什么
变成了 HOLD"的排查都变成猜。先把单线程跑通、把结果钉成金标准，
再上线程——那时候线程引入的差异会立刻暴露出来，而不是混在一堆噪声里。

## 时钟归谁管

循环是唯一决定 ``now`` 从哪来的地方：

- ``frame_clock``（默认）：``now = 包的采集时刻``。回放用。帧龄恒为 0，
  超龄判断永不触发——这是**对的**，因为回放里的"慢"不是真的慢，
  是 CPU 快慢，拿它去判超龄只会让快机器和慢机器跑出不同结果。
- ``wall_clock``：``now = time.monotonic()``。实时用。帧龄就是真实的
  排队延迟，超龄判断在这里才有意义。

**别在回放里用 wall_clock**，也别在实时里用 frame_clock——后者会让
``max_commit_age_s`` 永远不触发，这个保护就等于不存在。
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Iterator

from jg_radar.io.frame_source import FrameSource, iterate
from jg_radar.pipeline.mailbox import Mailbox, StageAPacket
from jg_radar.pipeline.stage_a import StageA
from jg_radar.pipeline.stage_b import StageB
from jg_radar.types import FrameOutput

__all__ = [
    "EventLoop",
    "ThreadedEventLoop",
    "default_clock",
    "frame_clock",
    "wall_clock",
]

Clock = Callable[[StageAPacket], float]


def default_clock(source) -> Clock:
    """按**帧源的时间线**选时钟。**别按线程模式选。**

    帧源没声明 ``is_live`` 时**报错，不替你猜**。猜错有两个方向，其中一个是
    要命的那种：

    - 把回放源当实时源 → 帧龄是几十万秒 → ``StageB`` 当场炸。吵，但安全。
    - 把实时源当回放源 → 帧龄恒为 0 → ``max_commit_age_s`` **永远不触发**，
      超龄数据被照常上报。全程没有任何异常，症状只是"位置偶尔旧一点"。

    一个忘了写 ``is_live`` 的相机驱动会落进第二种，而那正好是整条 EGA 式流水线
    存在的理由。所以这里宁可让它在启动那一刻就停下来。

    这条规则原来写反过：``ThreadedEventLoop`` 的默认值是 ``wall_clock``，
    那对实时是对的、对回放是错的。拿它跑回放目录，帧龄是"帧时间线（第 0 帧
    在 100 秒）对开机至今"的差——几十万秒——于是每一帧都超出
    ``MAX_PLAUSIBLE_AGE_S``，``StageB`` 直接炸。炸得**对**（那确实是两个时钟
    混在一起），但不该发生：``--threaded`` 跑回放是个正当需求，用来对比两种
    跑法在同一份输入上给不给同样的结果。

    决定用哪个 ``now`` 的是**采集端拿什么打的戳**，跟"跑在几个线程里"毫无
    关系。所以规则挂在帧源的 ``is_live`` 上，两种循环类都不再有各自的默认值
    ——那种"两个默认值不一样"的写法本身就是个陷阱。
    """
    if not hasattr(source, "is_live"):
        raise TypeError(
            f"{type(source).__name__} 没有声明 is_live，事件循环不知道该用哪个时钟。"
            "这个值决定 now 取采集时刻还是墙上时钟，而**猜错的方向是会静默失效的"
            "那一个**：把实时源当成回放源，帧龄就恒为 0，max_commit_age_s 永远不"
            "触发，超龄数据照常上报——没有异常、没有告警，只是位置偶尔旧一点。"
            "所以这里报错，不替你选。回放/合成/内存源写 is_live = False，"
            "相机驱动写 is_live = True 并用 time.monotonic() 打 captured_at。"
        )
    return wall_clock if source.is_live else frame_clock


def frame_clock(packet: StageAPacket) -> float:
    """``now`` 取自帧时间线。**回放的默认值。**

    帧龄于是恒为 0：回放里没有"排队延迟"这回事，两帧之间隔的那点时间
    完全取决于 CPU 有多快。拿它判超龄会让快机器保留、慢机器丢弃，
    同一份输入跑出两份输出——金标准就没了。
    """
    return float(packet.captured_at)


def wall_clock(packet: StageAPacket) -> float:
    """``now`` 取自单调墙钟。**实时部署用。**

    要求帧源也用 ``time.monotonic()`` 打 ``captured_at``，否则帧龄会是
    荒唐的值——``StageB`` 对此有体检，见 ``ClockMismatch``。
    """
    del packet
    return time.monotonic()


class _LoopBase:
    """两种跑法共用的记账与收尾。"""

    def __init__(self, *, source: FrameSource, stage_a: StageA, stage_b: StageB,
                 sink=None, clock: Clock | None = None) -> None:
        self.source = source
        self.stage_a = stage_a
        self.stage_b = stage_b
        self.sink = sink
        # ``None`` = 跟着帧源的时间线走。两个子类**都不给**各自的默认值：
        # 一个默认 frame、一个默认 wall 的写法本身就是个陷阱（见 default_clock）。
        self.clock = clock or default_clock(source)
        self.frames = 0
        self.dropped_stale = 0

    def _emit(self, out: FrameOutput | None) -> FrameOutput | None:
        if out is None:
            self.dropped_stale += 1
            return None
        self.frames += 1
        if self.sink is not None:
            self.sink.write(out)
        return out

    def diagnostics(self) -> dict[str, float]:
        return {
            "loop_frames": float(self.frames),
            "loop_dropped_stale": float(self.dropped_stale),
            **self.stage_a.diagnostics(),
            **self.stage_b.diagnostics(),
        }

    def _finish(self) -> None:
        try:
            self.source.close()
        finally:
            if self.sink is not None:
                self.sink.close()


class EventLoop(_LoopBase):
    """单线程、确定性。回放与测试的默认跑法。"""

    def run(self) -> Iterator[FrameOutput]:
        """逐帧产出。**只产出真正发出去的帧**——超龄的整帧不出现在这里。

        超龄的帧不是"输出了一帧空的"，而是**什么都没发**。两者在下游
        看起来完全不同：前者是一个说"我不知道"的槽位，后者是时间轴上
        直接缺了一块。本项目的输出契约里没有"我没发"这个状态，
        所以这里把它表达成"不产生帧"，并在 ``dropped_stale`` 里计数。
        """
        try:
            for frame in iterate(self.source):
                packet = self.stage_a.run(frame)
                out = self.stage_b.process(packet, now=self.clock(packet))
                emitted = self._emit(out)
                if emitted is not None:
                    yield emitted
        finally:
            # 调用方可能中途 break 掉这个生成器。生成器被抛弃时不是"跑完了"，
            # 但源和 sink 仍然必须关——否则文件句柄会留到解释器退出。
            self._finish()

    def run_to_completion(self) -> list[FrameOutput]:
        """跑完，把输出收成一个列表。测试与金标准比对用这个。"""
        return list(self.run())


class ThreadedEventLoop(_LoopBase):
    """Stage A 在后台线程，Stage B 在调用线程。**实时部署走这条。**

    ``run()`` 返回的是 Stage B 产出的帧，顺序与单线程版一致——只不过
    Stage A 跑在前面，中间积压的包会被信箱丢掉（``Mailbox.dropped``）。
    丢掉的那些帧同样不会出现在输出里。

    生产者线程里的异常会被**捕获并重新抛出**，不会静默吞掉。一个悄悄死掉的
    Stage A 会让流水线一路输出全 MISSING，而那看起来跟"这场确实没敌人"
    一模一样——这是本项目反复要挡的那一类失效。

    ## 生产者要按帧时间线节流

    实时部署里**帧自己会来**：相机每 50ms 给一帧，生产者再快也快不过它。
    回放里没有这个天然节拍——``read()`` 立刻返回下一帧，于是生产者一路狂奔，
    而信箱按设计**只留最新的**（见 ``mailbox`` 模块），结果是消费线程还在
    启动，96 帧就已经全被挤掉了，只收到 3 帧。那不是"丢几帧"，是整段回放
    变成噪声，而 ``--threaded`` 存在的唯一理由（对比两种跑法）就此落空。

    所以非实时源默认按**录制速度**喂帧（``pace = 1.0``）。这不是为了好看：
    节流之后信箱不再积压，线程版与单线程版的输出就可以逐帧对比，而**那正是
    用来发现"并发悄悄改坏了共享状态"的手段**。实时源不节流——帧本来就来
    多快是多快，再压一道等于自己给自己加延迟。
    """

    def __init__(
        self,
        *,
        source: FrameSource,
        stage_a: StageA,
        stage_b: StageB,
        sink=None,
        clock: Clock | None = None,
        mailbox: Mailbox | None = None,
        poll_timeout: float = 0.05,
        pace: float | None = None,
    ) -> None:
        super().__init__(source=source, stage_a=stage_a, stage_b=stage_b, sink=sink,
                         clock=clock)
        self.mailbox = mailbox if mailbox is not None else Mailbox(capacity=2)
        self.poll_timeout = float(poll_timeout)
        if self.poll_timeout <= 0.0:
            raise ValueError("poll_timeout 必须为正")
        if pace is None:
            pace = 0.0 if getattr(source, "is_live", False) else 1.0
        self.pace = float(pace)
        """生产者喂帧的速度倍率。``1.0`` = 录制速度，``2.0`` = 两倍速，
        ``<= 0`` = 不节流（跑多快算多快）。"""
        self._pace_origin: tuple[float, float] | None = None
        self._thread: threading.Thread | None = None
        self._producer_done = threading.Event()
        self._producer_error: BaseException | None = None
        self._stop = threading.Event()

    # -- 生产者 -----------------------------------------------------------
    def _throttle(self, captured_at: float) -> None:
        """把生产者压到帧时间线上：睡到"这一帧该到的墙钟时刻"。

        拿**帧自己的** ``captured_at`` 跟墙钟起点对齐，而不是每帧睡一个
        固定 ``dt``——后者会把每帧的处理耗时累加进误差里，跑几百帧之后
        节拍就漂了。这里每一帧都重新对表，处理慢了下一帧就少睡一点。

        睡眠切成 ``poll_timeout`` 的小段并盯着 ``_stop``：否则一个长回放
        收尾时要等最后一帧睡完才停得下来，``run()`` 的 ``join`` 会白等。
        """
        if self.pace <= 0.0:
            return
        now = time.monotonic()
        if self._pace_origin is None:
            self._pace_origin = (float(captured_at), now)
            return
        t0, w0 = self._pace_origin
        target = w0 + (float(captured_at) - t0) / self.pace
        while True:
            remaining = target - time.monotonic()
            if remaining <= 0.0:
                return
            if self._stop.wait(min(remaining, self.poll_timeout)):
                return

    def _produce(self) -> None:
        try:
            for frame in iterate(self.source):
                if self._stop.is_set():
                    break
                self._throttle(frame.captured_at)
                self.mailbox.put(self.stage_a.run(frame))
        except BaseException as exc:  # noqa: BLE001 - 必须原样带给消费者
            self._producer_error = exc
        finally:
            self._producer_done.set()

    @property
    def producer_error(self) -> BaseException | None:
        return self._producer_error

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._produce, name="stage-a", daemon=True)
        self._thread.start()

    # -- 消费者 -----------------------------------------------------------
    def consume(self) -> Iterator[FrameOutput]:
        """一直取到"生产者结束**且**信箱空了"。"""
        while True:
            if self._producer_error is not None:
                # 先抛错再谈别的：生产者挂了，后面"什么也没有"是假象。
                raise self._producer_error
            packet = self.mailbox.wait_latest(timeout=self.poll_timeout)
            if packet is None:
                if self._producer_done.is_set() and len(self.mailbox) == 0:
                    break
                if getattr(self.source, "is_live", False):
                    emitted = self._emit(self.stage_b.tick(now=time.monotonic()))
                    if emitted is not None:
                        yield emitted
                continue
            out = self.stage_b.process(packet, now=self.clock(packet))
            emitted = self._emit(out)
            if emitted is None and getattr(self.source, "is_live", False):
                emitted = self._emit(self.stage_b.tick(now=time.monotonic()))
            if emitted is not None:
                yield emitted

    def run(self) -> Iterator[FrameOutput]:
        self.start()
        try:
            yield from self.consume()
        finally:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=5.0)
            self._finish()

    def run_to_completion(self) -> list[FrameOutput]:
        return list(self.run())

    # -- 诊断 -------------------------------------------------------------
    def diagnostics(self) -> dict[str, float]:
        return {
            **super().diagnostics(),
            "mailbox_len": float(len(self.mailbox)),
            "mailbox_dropped": float(self.mailbox.dropped),
        }
