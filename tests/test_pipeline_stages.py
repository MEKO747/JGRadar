"""Stage A / Stage B / 事件循环的契约。

这三个模块是**唯一**知道"两个时钟"存在的地方，所以这里的重点全在时钟上：

- ``captured_at``（采集时刻）推进滤波器，``now``（判断时刻）管新鲜度。
  混用它们不报错、不崩溃，只会让回放和实时的行为不一致——而在回放里
  一切正常、到实车上才开始丢帧，是最难查的一类问题。
- 两个时钟混用有**两个方向**，帧龄一个是极大的负数、一个是极大的正数，
  两条分支各管一头，报错文案指的方向必须是对的（见 ``MAX_PLAUSIBLE_AGE_S``）。
- ``max_commit_age_s`` 这道保护**在诚实回放里永远不会触发**：回放的
  ``now`` 就取自帧时间线，帧龄恒为 0。所以"它没触发"不能证明它有用，
  要证明它有用必须**注入一个偏斜的时钟**。下面那几条就是这么做的。

Stage A 的测试盯的是另一件事：**投影不出来 ≠ 看不见**。投不到场地上的
观测必须照样传下去，因为它在图像域仍然有效。丢掉它等于把一条本来能维持
的轨迹拆散，而重建期间身份要重新投票——身份是本项目最慢的变量。

## 框的坐标是从查找表里现取的

不硬编码像素。能不能投到地面取决于相机内外参，改了 ``config/camera.yaml``
之后硬编码的框会**静默地**变成"投影失败"，于是这些测试开始测别的东西、
还全绿。下面 ``box_that_projects`` 从查找表的有效掩码里取点，改标定也跟得上。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from jg_radar.config import FuseConfig, PipelineConfig
from jg_radar.detect.armor_detector import ArmorHit
from jg_radar.detect.digit_classifier import UNKNOWN_CLASS, DigitVote
from jg_radar.detect.factory import CascadeResult
from jg_radar.detect.protocol import DetectionRecord
from jg_radar.io.frame_source import Frame, ListFrameSource, ReplayFrameSource
from jg_radar.pipeline import (
    MAX_PLAUSIBLE_AGE_S,
    ClockMismatch,
    EventLoop,
    StageA,
    StageB,
    ThreadedEventLoop,
    default_clock,
)
from jg_radar.pipeline.build import default_replay_dir
from jg_radar.pipeline.event_loop import frame_clock, wall_clock
from jg_radar.pipeline.mailbox import StageAPacket
from jg_radar.track import MultiTracker
from jg_radar.track.types import TrackObservation


# ----------------------------------------------------------------------
# 构造件
# ----------------------------------------------------------------------
def _config():
    from jg_radar.config.loader import load_config

    root = Path(__file__).resolve().parents[1]
    return load_config(
        root / "config" / "default.yaml",
        root / "config" / "camera.yaml",
        root / "config" / "field.yaml",
    )


def _build_lookup():
    """用回放 fixture **自带的那套标定**建表。

    ``config/camera.yaml`` 里的外参目前还是占位值（相机还没选型定标），拿它
    建出来的表一个有效像素都没有，于是"投影失败"会变成这些测试的常态——它们
    就全都在测别的、还全绿。fixture 的标定是真能用的，端到端回放走的也是它。
    """
    lk = ReplayFrameSource(default_replay_dir()).build_lookup()
    assert lk.valid.any(), "回放 fixture 的标定建不出有效查找表"
    return lk


LOOKUP = _build_lookup()


def car(label: str, bbox, *, conf: float = 0.9) -> DetectionRecord:
    return DetectionRecord(class_id=0, confidence=conf, bbox=tuple(bbox), label=label)


def cascade(cars=(), armors=(), digits=()) -> CascadeResult:
    return CascadeResult(
        cars=list(cars),
        rois=np.zeros((0, 4), dtype=np.float64),
        armors=list(armors),
        digits=list(digits),
    )


def armor(car_index: int, bbox, *, conf: float = 0.9) -> ArmorHit:
    return ArmorHit(
        car_index=car_index,
        record=DetectionRecord(class_id=0, confidence=conf, bbox=tuple(bbox)),
    )


def vote(label: str, *, conf: float = 0.9) -> DigitVote:
    return DigitVote(class_id=0, confidence=conf, label=label)


def unknown(*, conf: float = 0.9) -> DigitVote:
    return DigitVote(class_id=UNKNOWN_CLASS, confidence=conf, label="")


CAR_W_PX = 60.0
CAR_H_PX = 40.0


def box_that_projects(LOOKUP, *, where: float = 0.5) -> tuple[float, float, float, float]:
    """造一个**确实能投到地面上**的框。

    ``where`` 在有效区域的横向分布里选位置（0=最左，1=最右）。世界坐标取自
    底边中点，所以这里把底边中点放在那个有效像素上。
    """
    ys, xs = np.nonzero(LOOKUP.valid)
    assert len(xs) > 0, "查找表没有任何有效像素——标定或相机配置坏了"
    left, right = np.percentile(xs, 10), np.percentile(xs, 90)
    target_x = left + (right - left) * float(where)
    band = xs[np.abs(xs - target_x) <= max(1.0, (right - left) * 0.05)]
    band_y = ys[np.abs(xs - target_x) <= max(1.0, (right - left) * 0.05)]
    u = float(np.median(band))
    v = float(np.median(band_y))

    x1, y2 = u - CAR_W_PX / 2.0, v
    box = (x1, y2 - CAR_H_PX, x1 + CAR_W_PX, y2)
    world, ok = LOOKUP.sample_bbox_ground(np.array([box], dtype=np.float64))
    assert ok[0], "选出来的框投不到地面上——查找表有效掩码与取点逻辑不一致"
    assert np.isfinite(world[0]).all()
    return box


BOXES = (
    box_that_projects(LOOKUP, where=0.0),
    box_that_projects(LOOKUP, where=1.0),
)
"""一左一右两个都能投影的框，用来测两车场景。"""


def stage_b(**kw) -> StageB:
    """一个装配好的 Stage B。跟踪/融合配置默认取仓库里那份。"""
    cfg = kw.pop("config", None) or _config()
    return StageB(
        tracker=MultiTracker(cfg.track),
        pipeline=kw.pop("pipeline", PipelineConfig()),
        fuse=kw.pop("fuse", FuseConfig()),
        **kw,
    )


def packet_at(t: float, seq: int = 1) -> StageAPacket:
    return StageAPacket(frame_seq=seq, captured_at=t, observations=())


def obs_packet(box, t: float, seq: int = 1) -> StageAPacket:
    """带一条真观测的包，用来验证"状态确实被动了"。"""
    return StageAPacket(
        frame_seq=seq,
        captured_at=t,
        observations=(
            TrackObservation(bbox=box, world_cm=(1400.0, 700.0), confidence=0.9,
                             car_class="B1"),
        ),
    )


# ======================================================================
# Stage A —— 一帧 → 一组观测
# ======================================================================
def test_a_frame_without_a_cascade_is_refused():
    """帧既没带检测结果、Stage A 又没有级联 —— **炸，不许静默返回空**。

    静默返回空会伪装成一次正常的空检测，于是"模型加载失败"和"这场确实
    没敌人"在对外输出上完全一样。这是全项目最不能容忍的一种混淆。
    """
    with pytest.raises(ValueError, match="级联"):
        StageA(LOOKUP).run(Frame(seq=1, captured_at=0.0, cascade=None))


def test_a_frame_with_an_explicitly_empty_cascade_is_a_real_observation():
    """带了一张空检测结果 = **真的看过了，什么也没有**。不该炸。"""
    packet = StageA(LOOKUP).run(Frame.with_cascade(1, 0.0, cascade()))
    assert packet.observations == ()
    assert packet.frame_seq == 1


def test_stage_a_reads_the_anchor_from_the_bottom_of_the_box():
    """世界坐标取自**框底边中点**，不是框中心。

    俯视相机下穿过框中心的射线更平，打到地面时要多走一段，坐标被投到机器人
    **身后**。这个偏差随距离增长（实测 87–182cm），而 ``switch_jump_cm`` 才
    150cm——锚点选错，远处目标每次换源都会被判成跳变。
    """
    box = BOXES[0]
    ground, ok = LOOKUP.sample_bbox_ground(np.array([box], dtype=np.float64))
    center, ok2 = LOOKUP.sample_bbox_center(np.array([box], dtype=np.float64))
    assert ok[0] and ok2[0]

    packet = StageA(LOOKUP).run(Frame.with_cascade(1, 0.0, cascade([car("B1", box)])))
    got = packet.observations[0].world_cm
    assert got == pytest.approx((float(ground[0, 0]), float(ground[0, 1])), abs=1e-6)
    # 而且它确实和框中心投出来的不是同一个点
    assert abs(got[0] - float(center[0, 0])) + abs(got[1] - float(center[0, 1])) > 1.0


def test_an_unprojectable_box_still_produces_an_observation():
    """投影不出来（框在画面外/查表无效）**不能把观测丢掉**，只是没有坐标。

    这条观测在图像域仍然有效：还能参与关联、还能攒身份票。丢掉的代价是一
    条本来可以维持的轨迹被拆散重建，而重建期间身份要重新投票。
    """
    off_screen = (-900.0, -900.0, -800.0, -800.0)
    packet = StageA(LOOKUP).run(
        Frame.with_cascade(1, 0.0, cascade([car("B1", off_screen)]))
    )
    assert len(packet.observations) == 1
    assert packet.observations[0].world_cm is None
    assert packet.observations[0].car_class == "B1"


def test_stage_a_does_not_judge_only_converts():
    """Stage A 不做"这个检测靠不靠谱"的判断——那是跟踪层双闸门的活。"""
    degenerate = car("B1", (10.0, 10.0, 10.0, 10.0), conf=0.0)  # 零面积框
    packet = StageA(LOOKUP).run(Frame.with_cascade(1, 0.0, cascade([degenerate])))
    assert len(packet.observations) == 1


def test_digit_is_taken_by_index_parallelism_not_by_car():
    """``digits[j]`` 与 ``armors[j]`` **按下标一一对应**（``Cascade.run`` 的约定）。

    按 car_index 过滤出属于这台车的装甲板，再回头取**同下标**的数字。
    """
    box = BOXES[0]
    x1, y1, x2, y2 = box
    mid = (x1 + x2) / 2.0
    c = cascade(
        [car("B1", box)],
        [
            armor(0, (x1 + 4, y1 + 4, mid - 2, y1 + 20)),
            armor(0, (mid + 2, y1 + 4, x2 - 4, y1 + 20)),
        ],
        [vote("1", conf=0.4), vote("2", conf=0.8)],
    )
    packet = StageA(LOOKUP).run(Frame.with_cascade(1, 0.0, c))
    assert packet.observations[0].digit == "2"  # 置信度最高的那个


def test_a_digit_belonging_to_another_car_is_never_borrowed():
    """另一台车的数字不许被借过来。下标对不上就跳过，不许"就近取一个"。"""
    a, b = BOXES
    c = cascade(
        [car("B1", a), car("R2", b)],
        [armor(0, a), armor(1, b)],
        [unknown(conf=0.1), vote("2", conf=0.99)],
    )
    packet = StageA(LOOKUP).run(Frame.with_cascade(1, 0.0, c))
    assert packet.observations[0].digit == ""   # B1：自己那个没读出来
    assert packet.observations[1].digit == "2"  # R2


def test_a_short_digit_list_never_indexes_out_of_range():
    """数字批比装甲板批短（后端丢了一块）时要跳过，不是 IndexError。"""
    box = BOXES[0]
    c = cascade([car("B1", box)], [armor(0, box), armor(0, box)], [vote("1", conf=0.5)])
    packet = StageA(LOOKUP).run(Frame.with_cascade(1, 0.0, c))
    assert packet.observations[0].digit == "1"


def test_unlabeled_cars_are_counted_because_they_mean_a_broken_class_map():
    """类别表配错会让**每一条**观测都拿不到类别，于是身份永远定不下来、
    雷达站一路上报"没有敌人"。这个计数涨起来就是在说这件事。"""
    stage = StageA(LOOKUP)
    stage.run(
        Frame.with_cascade(1, 0.0, cascade([car("", BOXES[0]), car("B1", BOXES[1])]))
    )
    assert stage.unlabeled == 1
    assert stage.diagnostics()["stage_a_unlabeled"] == 1.0


def test_the_packet_freezes_one_timestamp_for_every_observation():
    """包内所有观测共享**同一个** ``captured_at``。

    EGA 的设计：同刻冻结。每帧内各观测各带自己时间戳的话，下游要处理
    "同一帧里两个时刻"，而这两个时刻都没有物理意义。
    """
    c = cascade([car("B1", BOXES[0]), car("R2", BOXES[1])])
    packet = StageA(LOOKUP).run(Frame.with_cascade(7, 123.5, c))
    assert packet.captured_at == 123.5
    assert len(packet.observations) == 2


# ======================================================================
# Stage B —— 时钟体检
# ======================================================================
def test_an_epoch_stamped_source_against_a_monotonic_loop_is_refused():
    """采集端 ``time.time()``、判断端 ``time.monotonic()``：帧龄是极大的**负数**。

    走"时间倒流"那一支。文案必须指出问题在采集端——指错方向的话，排查的人
    会去翻判断端，那里其实是对的。
    """
    with pytest.raises(ClockMismatch, match="采集端用了 time.time"):
        stage_b().process(packet_at(1.7e9), now=1234.5)


def test_a_monotonic_stamped_source_against_an_epoch_loop_is_refused():
    """反方向：采集端 monotonic、判断端 ``time.time()``，帧龄约 1.7e9 秒。

    走 ``MAX_PLAUSIBLE_AGE_S`` 那一支。**不拦的后果是每一帧都被判超龄丢弃**，
    对外一路全 MISSING——看起来和"这一场确实没敌人"一模一样，且没有报错。
    """
    with pytest.raises(ClockMismatch, match="判断端用了 time.time"):
        stage_b().process(packet_at(1234.5), now=1.7e9)


def test_the_age_ceiling_sits_where_it_says_it_does():
    assert MAX_PLAUSIBLE_AGE_S == 60.0


@pytest.mark.parametrize("ahead", [0.0, 0.001, 0.4])
def test_a_slightly_early_timestamp_is_tolerated_and_clamped(ahead):
    """略微为负是允许的（浮点、以及"采集比判断早一个毫秒"这种顺序巧合）。

    把它也判成时钟问题的话，正常运行时会随机炸。
    """
    b = stage_b()
    assert b.process(packet_at(100.0 + ahead), now=100.0) is not None
    assert b.last_age == 0.0


# ======================================================================
# Stage B —— 超龄丢弃
# ======================================================================
def test_a_fresh_packet_is_committed_and_an_over_age_one_is_dropped():
    """**超龄保护的唯一有效证明方式是注入偏斜时钟。**

    回放的 ``now`` 取自帧时间线，帧龄恒为 0，这道判断在诚实回放里永远不触发
    ——"它没触发"因此什么也证明不了。这里手动把 ``now`` 往后拨。
    """
    b = stage_b(pipeline=PipelineConfig(max_commit_age_s=0.15))

    assert b.process(packet_at(100.0), now=100.10) is not None  # 100ms < 150ms
    assert b.dropped_stale == 0

    assert b.process(packet_at(100.0, seq=2), now=100.20) is None  # 200ms > 150ms
    assert b.dropped_stale == 1
    assert b.processed == 1


def test_the_drop_threshold_is_strictly_greater_than():
    """刚好等于门限不算超龄。差一帧就丢是没道理的。"""
    b = stage_b(pipeline=PipelineConfig(max_commit_age_s=0.15))
    assert b.process(packet_at(100.0), now=100.145) is not None
    assert b.dropped_stale == 0


def test_an_over_age_packet_must_not_touch_the_tracker_at_all():
    """**丢弃要丢得干净。** 超龄包若先推进滤波器、再写缓冲、最后才在发布那一刻
    被拦下，它已经污染了状态：滤波器的 dt 被拉长、缓冲里被塞进一条陈旧样本。
    """
    b = stage_b(pipeline=PipelineConfig(max_commit_age_s=0.15))
    steps: list[float] = []
    original = b.tracker.step

    def spy(observations, *, now):
        steps.append(now)
        return original(observations, now=now)

    b.tracker.step = spy

    # 两帧才够 min_hits 确认轨迹、进而产出样本——只喂一帧的时候槽位还是 TENTATIVE
    assert b.process(obs_packet(BOXES[0], 100.00, seq=1), now=100.00) is not None
    assert b.process(obs_packet(BOXES[0], 100.05, seq=2), now=100.05) is not None
    assert len(steps) == 2
    assert len(b.buffer) == 1, "前置条件：新鲜包确实写进了缓冲"

    assert b.process(obs_packet(BOXES[0], 100.10, seq=3), now=100.50) is None
    assert len(steps) == 2, "超龄包推进了跟踪器——丢掉的是半个包，不是整个包"
    assert len(b.buffer) == 1, "超龄包写进了缓冲"


def test_a_dropped_packet_does_not_advance_the_frame_counter():
    """超龄的帧不是"输出了一帧空的"，而是**什么都没发**。

    两者在下游看起来完全不同：前者是一个说"我不知道"的槽位，后者是时间轴上
    直接缺了一块。本项目没有"我没发"这个状态，所以表达成"不产生帧"。
    """
    b = stage_b(pipeline=PipelineConfig(max_commit_age_s=0.15))
    b.process(packet_at(100.0), now=100.30)
    assert b.processed == 0
    assert b.diagnostics()["dropped_stale"] == 1.0


def test_max_commit_age_is_read_from_the_pipeline_section_not_the_fuse_one():
    """``max_commit_age_s`` 归 ``pipeline`` 段管。

    两份配置分开收是有意的：合成一个对象的话，早晚有人把它写进 ``fuse`` 段，
    而它在那里不会有任何效果——不报错，只是那个保护**静默地不存在了**。
    """
    b = stage_b(pipeline=PipelineConfig(max_commit_age_s=0.01))
    assert b.process(packet_at(100.0), now=100.5) is None
    assert b.dropped_stale == 1


# ======================================================================
# Stage B —— 提交
# ======================================================================
def test_a_committed_frame_carries_both_clocks_and_the_right_sequence():
    out = stage_b().process(packet_at(100.0, seq=42), now=100.02)
    assert out is not None
    assert out.frame_seq == 42
    assert out.captured_at == 100.0
    assert out.published_at == 100.02


def test_every_slot_is_present_and_the_invariants_hold_on_an_empty_scene():
    """空场景也要产出**全部 12 个槽位**，且每条都满足输出契约的不变式。

    "没看到敌人"必须表达成 ``MISSING``，不能表达成"这一帧没发"或"少几个槽位"
    ——裁判侧看到的是槽位，缺一个和报一个 MISSING 是两回事。
    """
    out = stage_b().process(packet_at(100.0), now=100.0)
    assert out is not None
    assert len(out.robots) == 12
    for name, slot in out.robots.items():
        slot.validate()  # 坏数据不出进程
        assert slot.state.value == "missing"
        assert slot.position_cm is None
        assert slot.is_predicted is False, name


def test_the_tracker_is_advanced_on_the_capture_clock_not_the_judgement_clock():
    """滤波器用 ``captured_at`` 推进：dt 必须等于**物理上过去的时间**。

    拿 ``now`` 去推的话，一次 1 秒的卡顿会让状态一步跳 1 秒，预测直接冲出去。
    这里两帧的 ``captured_at`` 只差 50ms，而 ``now`` 差了 1 秒——跟踪器的
    记账时刻必须跟着前者走。
    """
    b = stage_b(pipeline=PipelineConfig(max_commit_age_s=5.0))
    b.process(obs_packet(BOXES[0], 100.00, seq=1), now=100.00)
    b.process(obs_packet(BOXES[0], 100.05, seq=2), now=101.00)

    assert len(b.tracker._tracks) == 1
    track = b.tracker._tracks[0]
    assert track.filtered_at == pytest.approx(100.05), (
        "跟踪器被 now 推进了——卡顿时预测会冲出去"
    )
    assert track.last_observed_at == pytest.approx(100.05)


def test_the_source_watchdog_uses_the_judgement_clock_not_the_capture_clock():
    """样本写进缓冲用的是 ``now``（"这个源此刻还活着吗"），不是 ``captured_at``。

    看门狗量的是 ``received_at``。预测态的 ``observed_at`` 本来就该很旧
    ——那是**设计**，不是故障；拿它去看门狗会把所有预测态都判成死源。
    """
    # 把 now 拨到采集时刻之后 1 秒，两个时刻就**分得开**了——否则它们相等，
    # 这条测试对"用错了哪一个"完全没有分辨力。
    b = stage_b(pipeline=PipelineConfig(max_commit_age_s=5.0))
    b.process(obs_packet(BOXES[0], 100.00, seq=1), now=101.00)
    b.process(obs_packet(BOXES[0], 100.05, seq=2), now=101.05)

    stamped = b.buffer.snapshot()
    assert len(stamped) > 0, "前置条件：观测确实变成了样本"
    for s in stamped:
        assert s.received_at == 101.05, "看门狗读的是判断时刻，不是采集时刻"
        assert s.sample.observed_at == 100.05, "对外报出的观测时刻必须是采集时刻"


# ======================================================================
# 事件循环
# ======================================================================
class ClosingSource:
    """包一层 ``ListFrameSource``，只为了看得见 ``close()`` 被调用了。

    ``ListFrameSource.close`` 是空操作，拿它测"关没关"等于没测。

    ``is_live`` 得自己声明：它是 Protocol 的一部分，包一层不会自动继承。
    """

    is_live = False

    def __init__(self, frames) -> None:
        self._inner = ListFrameSource(frames)
        self.closed = False

    def read(self):
        return self._inner.read()

    def close(self) -> None:
        self.closed = True


class RecordingSink:
    def __init__(self) -> None:
        self.frames: list = []
        self.closed = False

    def write(self, out) -> None:
        self.frames.append(out)

    def close(self) -> None:
        self.closed = True


def make_loop(box, *, n_frames: int, dt: float, clock=frame_clock, sink=None):
    cfg = _config()
    frames = [
        Frame.with_cascade(i, 100.0 + i * dt, cascade([car("B1", box)]))
        for i in range(n_frames)
    ]
    return EventLoop(
        source=ClosingSource(frames),
        stage_a=StageA(LOOKUP),
        stage_b=stage_b(config=cfg),
        sink=sink,
        clock=clock,
    )


def test_the_frame_clock_never_drops_anything_because_age_is_identically_zero():
    """回放的 ``now`` 取自帧时间线 ⇒ 帧龄恒为 0 ⇒ 超龄判断永不触发。

    这是**对的**：回放里的"慢"不是真的慢，是 CPU 快慢，拿它判超龄只会让
    快机器和慢机器跑出不同结果，金标准就没了。
    """
    loop = make_loop(BOXES[0], n_frames=12, dt=0.05, clock=frame_clock)
    outs = loop.run_to_completion()
    assert len(outs) == 12
    assert loop.dropped_stale == 0
    assert loop.diagnostics()["max_commit_age_ms"] == 0.0


def test_an_injected_skewed_clock_actually_drops_frames_end_to_end():
    """把 ``now`` 整体往后拨，超龄丢弃必须在**循环这一层**真的发生。

    没有这条测试，"无超龄提交"这个验收指标就是空转的：帧龄恒为 0 当然
    永远不超龄。这里拨快 1 秒、门限 150ms ⇒ 每一帧都该被丢掉。
    """
    loop = make_loop(
        BOXES[0], n_frames=8, dt=0.05,
        clock=lambda packet: packet.captured_at + 1.0,
    )
    outs = loop.run_to_completion()
    assert outs == []
    assert loop.dropped_stale == 8
    assert loop.frames == 0


def test_a_clock_that_is_skewed_backwards_raises_rather_than_dropping_silently():
    loop = make_loop(
        BOXES[0], n_frames=4, dt=0.05,
        clock=lambda packet: packet.captured_at - 5.0,
    )
    with pytest.raises(ClockMismatch):
        loop.run_to_completion()


def test_the_wall_clock_is_fine_as_long_as_the_source_is_stamped_the_same_way():
    """实时路径：帧源用 ``time.monotonic()`` 打戳，循环读同一个钟。

    这里用真实单调钟造帧，帧龄是真的排队延迟（很小），不该被丢掉。
    """
    import time

    cfg = _config()
    now = time.monotonic()
    frames = [
        Frame.with_cascade(i, now + i * 0.001, cascade([car("B1", BOXES[0])]))
        for i in range(4)
    ]
    loop = EventLoop(
        source=ClosingSource(frames),
        stage_a=StageA(LOOKUP),
        stage_b=stage_b(config=cfg),
        clock=wall_clock,
    )
    outs = loop.run_to_completion()
    assert len(outs) == 4
    assert loop.dropped_stale == 0


def test_abandoning_the_generator_still_closes_the_source_and_the_sink():
    """调用方中途 ``break`` 掉生成器。生成器被抛弃不是"跑完了"，
    但源和 sink 仍然必须关——否则文件句柄会留到解释器退出。"""
    import gc

    sink = RecordingSink()
    loop = make_loop(BOXES[0], n_frames=50, dt=0.05, sink=sink)
    for i, _ in enumerate(loop.run()):
        if i == 2:
            break
    gc.collect()
    assert loop.source.closed is True
    assert sink.closed is True


def test_the_sink_sees_exactly_the_frames_that_were_emitted():
    sink = RecordingSink()
    outs = make_loop(BOXES[0], n_frames=7, dt=0.05, sink=sink).run_to_completion()
    assert len(sink.frames) == len(outs) == 7
    assert [f.frame_seq for f in sink.frames] == list(range(7))


def test_the_sink_is_closed_when_the_loop_finishes_normally():
    sink = RecordingSink()
    make_loop(BOXES[0], n_frames=3, dt=0.05, sink=sink).run_to_completion()
    assert sink.closed is True


def test_the_replay_is_deterministic_frame_for_frame():
    """同一份输入跑两次必须**逐字段**相同。

    单线程是默认跑法就是为了这个：任何金标准比对都建立在"同样的输入给同样
    的输出"之上。
    """

    def run_once():
        loop = make_loop(BOXES[0], n_frames=20, dt=0.05)
        return [
            (
                o.frame_seq,
                {
                    n: (s.state.value, s.position_cm, s.source.value)
                    for n, s in o.robots.items()
                },
            )
            for o in loop.run_to_completion()
        ]

    assert run_once() == run_once()


def test_a_growing_track_is_reported_as_measured_not_predicted():
    """跟踪器真的通了：同一个框喂 20 帧，槽位必须从 MISSING 变成 MEASURED。"""
    outs = make_loop(BOXES[0], n_frames=20, dt=0.05).run_to_completion()
    last = outs[-1]
    assert last.robots["B1"].state.value == "measured"
    assert last.robots["B1"].position_cm is not None
    assert last.robots["B1"].is_predicted is False
    assert last.robots["R1"].state.value == "missing"


def test_a_track_that_stops_being_fed_walks_down_the_ladder():
    """停止喂观测后必须按 ``lost_predict_s`` / ``ttl_s`` 逐级降级，
    而不是**一帧掉到 MISSING**，也不是**永远停在 MEASURED**。

    状态机是"连续 3 秒"那条规则的全部实现：一次遮挡不该让裁判侧看到一个
    凭空消失的目标，而一条早就该判死的轨迹也不该一直占着槽位。
    """
    cfg = _config()
    box = BOXES[0]
    visible = [Frame.with_cascade(i, 100.0 + i * 0.05, cascade([car("B1", box)]))
               for i in range(20)]
    hidden = [Frame.with_cascade(20 + i, 101.0 + i * 0.05, cascade())
              for i in range(80)]
    loop = EventLoop(
        source=ClosingSource(visible + hidden),
        stage_a=StageA(LOOKUP),
        stage_b=stage_b(config=cfg),
    )
    seq = [o.robots["B1"].state.value for o in loop.run_to_completion()]
    # 压成"阶梯"：只留下状态发生变化的那些段，读起来就是 MEASURED → HOLD → BLIND → MISSING
    order = [s for i, s in enumerate(seq) if i == 0 or s != seq[i - 1]]

    assert order == ["missing", "measured", "occlusion_hold",
                     "blind_prediction", "missing"], order


# ======================================================================
# 时钟与节流都挂在帧源上，不挂在跑法上
# ======================================================================
class LiveSource:
    """假装是相机驱动：``is_live`` 为真，``captured_at`` 用墙上时钟打。

    本项目范围内没有相机驱动（方案 §范围），所以只能造一个——但这条规则恰恰
    是接相机时最容易踩的那一脚：谁把 ``is_live`` 忘了置 ``True``，整个实时
    链路就会用帧时间线去判断新鲜度。
    """

    is_live = True

    def __init__(self, count: int, box, *, dt: float = 0.05) -> None:
        self.count = count
        self.dt = dt
        self.box = box
        self._i = 0
        self.t0 = time.monotonic()

    def read(self):
        if self._i >= self.count:
            return None
        i = self._i
        self._i += 1
        return Frame.with_cascade(
            i, self.t0 + i * self.dt, cascade([car("B1", self.box)])
        )

    def close(self) -> None:
        pass


def _loop_pair(source):
    """同一个源分别用两种跑法包一遍。"""
    cfg = _config()
    kw = dict(source=source, stage_a=StageA(LOOKUP), stage_b=stage_b(config=cfg))
    return EventLoop(**kw), ThreadedEventLoop(**kw)


def test_the_loop_clock_follows_the_source_not_the_threading_mode():
    """决定用哪个 ``now`` 的是**采集端拿什么打的戳**，不是"跑在几个线程里"。

    这条规则原来写反过：``ThreadedEventLoop`` 的默认时钟写死成 ``wall_clock``。
    那对实时是对的，对回放是灾难——帧时间线的第 0 帧落在 100.0 秒，拿它去和
    "开机至今"相减得到几十万秒的帧龄，每一帧都超出 ``MAX_PLAUSIBLE_AGE_S``，
    整段回放直接炸。炸得没错（那确实是两个时钟混在一起），但 ``--threaded``
    跑回放是个正当需求。

    判据只看 ``is_live``，于是**同一个源在两种跑法下必须拿到同一个时钟**——
    真出现两个不同的默认值，就说明规则又被挂到跑法上去了。
    """
    replay = ClosingSource([])
    live = LiveSource(0, BOXES[0])

    assert default_clock(replay) is frame_clock
    assert default_clock(live) is wall_clock

    for source, expected in ((replay, frame_clock), (live, wall_clock)):
        single, threaded = _loop_pair(source)
        assert single.clock is expected
        assert threaded.clock is expected


def test_the_loop_clock_can_still_be_overridden():
    """显式给时钟时以给的为准——上面那条规则只是**默认值**。

    回放里就是想拿墙上时钟跑（比如量一遍真实排队延迟），或者测试要注入一个
    拧歪的时钟来逼出超龄分支，都得能覆盖掉。
    """
    single, _ = _loop_pair(ClosingSource([]))
    assert single.clock is frame_clock  # 没给就是默认

    forced = EventLoop(
        source=ClosingSource([]), stage_a=StageA(LOOKUP),
        stage_b=stage_b(config=_config()), clock=wall_clock,
    )
    assert forced.clock is wall_clock


def test_a_source_that_forgets_to_declare_is_live_is_refused():
    """忘了声明 ``is_live`` 要**报错**，不能被默认成一个方向。

    两个方向的后果差着量级，所以不能猜：

    - 回放被当成实时 → 帧龄几十万秒 → 当场炸。吵，但安全。
    - 实时被当成回放 → 帧龄恒为 0 → ``max_commit_age_s`` 永远不触发，
      超龄数据照常上报。**全程无异常**，只是位置偶尔旧一点。

    一个忘了写 ``is_live`` 的相机驱动会落进第二种——而静默失效正是本项目
    反复要挡的那一类。
    """

    class Forgot:
        def read(self):
            return None

        def close(self) -> None:
            pass

    with pytest.raises(TypeError, match="is_live"):
        default_clock(Forgot())


def test_a_replay_source_is_paced_and_a_live_source_is_not():
    """生产者只在**回放**里节流。

    回放的 ``read()`` 立刻返回下一帧，生产者会一路狂奔到结束；信箱按设计只留
    最新的，于是消费线程还没启动，整段回放就只剩最后几帧了（实测 96 帧里只剩
    3 帧）。节流之后两种跑法才能逐帧对比——那是查"并发改坏了共享状态"的唯一
    手段，不节流的话那条比对比的是噪声。

    实时不能节流：帧本来就来多快是多快，再压一道等于给自己加延迟，而雷达的
    全部价值就在时效性上。
    """
    _, threaded_replay = _loop_pair(ClosingSource([]))
    _, threaded_live = _loop_pair(LiveSource(0, BOXES[0]))

    assert threaded_replay.pace == 1.0, "回放应当按录制速度喂帧"
    assert threaded_live.pace == 0.0, "实时不该节流"


def test_pacing_can_be_switched_off_explicitly():
    """``pace=0`` 关掉节流；这正是 ``--no-pace`` 走的那条路。"""
    cfg = _config()
    loop = ThreadedEventLoop(
        source=ClosingSource([]), stage_a=StageA(LOOKUP),
        stage_b=stage_b(config=cfg), pace=0.0,
    )
    assert loop.pace == 0.0
    # 关掉之后 _throttle 必须立刻返回，不能还睡在那儿
    loop._throttle(1234.5)
    assert loop._pace_origin is None
