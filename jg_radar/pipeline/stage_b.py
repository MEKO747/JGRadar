"""Stage B：有序接纳视觉观测，独立仲裁与发布。

process(packet, now) 在任何状态更新前检查帧龄/顺序，按 captured_at 推进跟踪。
有效包生成视觉最新值，退休身份显式撤回；原始观测时间始终保留。
tick(now) 在视觉没有新包时继续处理已有来源并发布，绝不刷新视觉看门狗。
实时线程在空闲或视觉包超龄时调用 tick；确定性回放只按帧时间线推进。
"""

from __future__ import annotations
import math

from jg_radar.fuse.arbiter import Arbiter
from jg_radar.fuse.contract import LatestValueBuffer
from jg_radar.fuse.publish import PublishLayer
from jg_radar.fuse.sources.vision_source import VisionSource
from jg_radar.pipeline.mailbox import StageAPacket
from jg_radar.pipeline.timings import Timings
from jg_radar.types import FrameOutput

__all__ = ["MAX_PLAUSIBLE_AGE_S", "ClockMismatch", "StageB"]

MAX_PLAUSIBLE_AGE_S = 60.0
"""帧龄超过这个值就认为**不是"很旧"，而是两个时钟搞混了**。

两个时钟混用会让帧龄错到几十年，**两个方向都可能**，``_check_age`` 两个
分支各管一头：

``captured_at`` 用 epoch（``time.time()``，约 1.7e9）、``now`` 用
``time.monotonic()``（开机至今，几千秒）
    帧龄是**极大的负数**，走"时间倒流"那一支。
``captured_at`` 用 monotonic、``now`` 用 epoch
    帧龄是**极大的正数**（约 1.7e9 秒），也就是本常量这一支。

方向搞反了不影响拦截效果（两边都炸），但会影响日志里指错方向——排查的人
照着"检查采集端"去查，而问题在判断端。所以两支分开写、分开测。

不拦的后果是**每一帧都被判超龄丢弃**，对外一路输出全 MISSING——看起来
跟"这一场确实没敌人"一模一样，而且没有任何报错。宁可启动就炸。
"""


class ClockMismatch(ValueError):
    """帧龄在物理上不可能。几乎总是两个不同的时钟被混在了一起。"""


class StageB:
    """观测包 → 对外帧。**有状态**：跟踪器、缓冲、仲裁器、发布层都活在这里。"""

    def __init__(
        self,
        *,
        tracker,
        pipeline,
        fuse,
        buffer: LatestValueBuffer | None = None,
        arbiter: Arbiter | None = None,
        publisher: PublishLayer | None = None,
        vision: VisionSource | None = None,
        timings: Timings | None = None,
    ) -> None:
        # 两份配置分开收，**不合并**：``pipeline`` 管的是"这一帧还值不值得发"，
        # ``fuse`` 管的是"这一帧该听谁的"。合成一个对象的话，早晚会有人把
        # ``max_commit_age_s`` 写进 fuse 段，而它在那里不会有任何效果——
        # 不报错，只是那个保护静默地不存在了。
        self.cfg = pipeline
        self.fuse_cfg = fuse
        self.tracker = tracker
        self.buffer = buffer if buffer is not None else LatestValueBuffer()
        self.arbiter = arbiter or Arbiter(fuse)
        self.publisher = publisher or PublishLayer(fuse)
        self.vision = vision or VisionSource()
        self.timings = timings or Timings()

        self.dropped_stale = 0
        """因超龄被整个丢掉的帧数。**这是"无超龄提交"这个验收指标的分子。**"""
        self.processed = 0
        self.max_observed_age = 0.0
        self.last_age = 0.0
        self._last_packet: StageAPacket | None = None

    # -- 主入口 -----------------------------------------------------------
    def process(self, packet: StageAPacket, *, now: float) -> FrameOutput | None:
        """处理一个包。超龄时返回 ``None``（**什么都没做**）。"""
        age = self._check_age(packet, now)
        self.last_age = age
        self.max_observed_age = max(self.max_observed_age, age)

        if age > float(self.cfg.max_commit_age_s):
            # 整个丢掉，就地返回。**不推进跟踪器、不写缓冲**——见模块文档。
            self.dropped_stale += 1
            return None

        captured_at = float(packet.captured_at)
        if self._last_packet is not None and (
            captured_at <= self._last_packet.captured_at
            or packet.frame_seq <= self._last_packet.frame_seq
        ):
            self.dropped_stale += 1
            return None

        with self.timings.measure("track"):
            # 采集时刻，不是 now：滤波器的 dt 必须等于物理上过去的时间。
            slots = self.tracker.step(packet.observations, now=captured_at)

        with self.timings.measure("fuse"):
            # 样本自带 captured_at 作为 observed_at（尤其是预测态，它就该
            # 停在最后一次真观测上，那是设计不是遗漏）；写进缓冲的时刻
            # 才是 now —— 看门狗问的是"这个源此刻还活着吗"。
            self.vision.submit(slots, now=captured_at)
            samples = self.vision.poll(now)
            self.buffer.put_many(samples, now=now)
            decisions = self.arbiter.decide(self.buffer.snapshot(), now=now)

        with self.timings.measure("publish"):
            # 这一刻 "publish" 这一段还没跑完，所以诊断里的 t_publish_ms 是
            # **上一帧**的值。对 EMA 来说无所谓，写下来是为了别人别去对它。
            frame = self.publisher.build(
                decisions,
                frame_seq=packet.frame_seq,
                captured_at=captured_at,
                published_at=now,
                diagnostics=self.diagnostics(),
            )

        self.processed += 1
        self._last_packet = packet
        return frame

    def tick(self, *, now: float) -> FrameOutput:
        """没有新视觉包时仍仲裁/发布；不推进视觉观测、不刷新视觉看门狗。"""
        packet = self._last_packet
        decisions = self.arbiter.decide(self.buffer.snapshot(), now=now)
        return self.publisher.build(
            decisions,
            frame_seq=packet.frame_seq if packet is not None else -1,
            captured_at=packet.captured_at if packet is not None else now,
            published_at=now,
            diagnostics=self.diagnostics(),
        )

    # -- 时钟体检 ---------------------------------------------------------
    def _check_age(self, packet: StageAPacket, now: float) -> float:
        """算出帧龄，并挡掉"两个时钟搞混了"这种物理上不可能的值。

        略微为负是允许的（浮点、以及"采集时刻比判断时刻晚一个毫秒"这种
        纯粹的顺序巧合），钳到 0。**大幅为负**则是时间倒流，一定是时钟问题。

        两个分支分别对应两个时钟混用的**两个方向**，报错文案指的方向必须是
        对的那个——指错了，排查的人就会去翻不该翻的那一端。见 ``MAX_PLAUSIBLE_AGE_S``。
        """
        age = float(now) - float(packet.captured_at)
        if not math.isfinite(age):
            raise ClockMismatch("采集和判断时刻必须是有限的单调时钟值")
        if age < -0.5:
            raise ClockMismatch(
                f"第 {packet.frame_seq} 帧的采集时刻 {packet.captured_at:.6f} "
                f"晚于判断时刻 {now:.6f}（{age:.3f}s）。采集与判断必须走同一个时钟。"
                "这个方向（帧来自未来）通常是**采集端用了 time.time()（epoch，"
                "约 1.7e9 秒）而判断端用 time.monotonic()**：这样算出来的帧龄是"
                "极大的负数。采集端请改用 time.monotonic() 打时间戳。"
            )
        if age > MAX_PLAUSIBLE_AGE_S:
            raise ClockMismatch(
                f"第 {packet.frame_seq} 帧的帧龄是 {age:.1f}s，超出物理可能。"
                "采集与判断必须走同一个时钟。这个方向（帧来自很久以前）通常是"
                "**采集端用了 time.monotonic() 而判断端用了 time.time()**："
                "这样帧龄会是约 1.7e9 秒。注意它的症状是**每帧都被丢弃、对外"
                "一路全 MISSING**，看上去和'这场确实没敌人'一模一样——所以这里"
                "直接炸，不静默。"
            )
        return max(0.0, age)

    # -- 诊断 -------------------------------------------------------------
    def diagnostics(self, *, extra: dict[str, float] | None = None) -> dict[str, float]:
        diag: dict[str, float] = {
            "commit_age_ms": self.last_age * 1e3,
            "max_commit_age_ms": self.max_observed_age * 1e3,
            "dropped_stale": float(self.dropped_stale),
            "arbiter_switches": float(self.arbiter.switch_count),
            # "有一路数据在源源不断地报着，但我们一条都没发出去"的唯一外部征兆。
            # 这个数一直涨说明某个源喂进来的坐标已经旧到不值得报了（去看它的
            # 信息年龄），或者某个预测态压根没带 position_std_cm（走的是
            # position_error_bound_cm 的第三条分支）。
            "arbiter_vetoed": float(self.arbiter.vetoed),
            "buffer_slots": float(len(self.buffer)),
            **self.tracker.last_diagnostics,
            **self.timings.snapshot(),
        }
        if extra:
            diag.update({k: float(v) for k, v in extra.items()})
        return diag

    def reset(self) -> None:
        self.buffer.clear()
        self.arbiter.reset()
        self.publisher.reset()
        self.vision.reset()
        self.tracker.reset()
        self._last_packet = None
        self.dropped_stale = 0
        self.processed = 0
        self.max_observed_age = 0.0
        self.last_age = 0.0
