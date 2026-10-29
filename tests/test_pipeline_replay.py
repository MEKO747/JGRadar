"""回放集成测试：一份回放目录 → 端到端的对外输出。

这是 M6 的验收门。前面每个模块各自都有自己的测试，但它们都是**分开**测的：
变换测往返、跟踪测闸门、仲裁测滞回。分开测全绿、合起来报错，是这类系统最
常见的失败方式——四份开源方案里没有一份做过这件事（方案 §九 引用的共同空白）。

所以这里断言的是**整条链路跑出来的状态阶梯**，一个字都不改。金标准写死在
文件里而不是从代码里算出来：从被测代码导出期望值，等于没测。

## 为什么有的断言要注入时钟

流水线默认用 ``frame_clock``，``now`` 直接取自包的采集时刻，帧龄于是**恒等于 0**，
``max_commit_age_s`` 这条保护永远不会触发。在回放里这是对的（回放的"慢"是 CPU
慢，不是数据旧），但它意味着"没超龄"这件事在默认配置下**无法通过跑一遍来证明**
——它恒真，测试是空的。

要证明那扇门真的会关，就得**把时钟拧歪**：喂进去一个比采集时刻晚
``max_commit_age_s`` 以上的 ``now``，看整帧是不是真的被丢掉。见
``test_an_over_age_frame_is_dropped_before_it_reaches_the_tracker``。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from jg_radar.config.loader import load_config
from jg_radar.io.frame_source import ReplayFrameSource
from jg_radar.pipeline.build import build_loop, default_replay_dir
from jg_radar.types import FrameOutput, RobotSlotOutput, RobotState, SourceKind
from tools.replay_runner import check_contract, summarise

ROOT = Path(__file__).resolve().parents[1]
FRAMES = 96

#: 回放目录跑出来的状态阶梯，**金标准**。
#:
#: B1 是这里面信息量最大的一条：四种状态按 ``missing → measured →
#: occlusion_hold → blind_prediction → missing`` 走全了。它是"目标出场、
#: 被看见、被挡住、彻底消失"的完整生命周期，也正是四态契约存在的理由。
#: 少于三种状态的阶梯说明状态机在某一档卡住了；状态来回横跳（measured 和
#: hold 交替出现多次）说明闸门在抖，两者都会在这条断言上现形。
GOLDEN_LADDERS = {
    # 航点偏移计入误差后，超限的猜测不发布。两个短暂缺失段均是门限拒绝，
    # 不是新观测丢失；B3/R2 的健康实测阶梯保持不变。
    "B1": "missing×1 → measured×24 → occlusion_hold×16 → missing×4 → blind_prediction×10 → missing×41",
    "B3": "missing×1 → measured×95",
    "R2": "missing×31 → measured×65",
}

#: 除上面三条之外，其余槽位应当**全程** MISSING。压成一行常量是为了让
#: "哪个槽位多报了一次"一眼可见。
ALL_MISSING = "missing×96"


def _config():
    return load_config(
        str(ROOT / "config" / "default.yaml"),
        str(ROOT / "config" / "camera.yaml"),
        str(ROOT / "config" / "field.yaml"),
    )


def _run(*, threaded: bool = False, clock=None):
    """跑一遍整条流水线，返回 （发出的帧, 循环）。"""
    source = ReplayFrameSource(default_replay_dir())
    kwargs = {"threaded": threaded}
    if clock is not None:
        kwargs["clock"] = clock
    loop = build_loop(_config(), frame_source=source, **kwargs)
    return list(loop.run()), loop


def _contract_view(frames) -> list[tuple]:
    """只取**契约内容**：帧标识 + 各槽位输出。

    诊断里的 ``t_*_ms`` 是实测耗时，同一台机器连跑两次都不会相同。确定性
    断言必须把它排除掉，否则测的是"这台机器忙不忙"，不是"流水线确不确定"。
    """
    return [
        (f.frame_seq, f.captured_at, f.published_at, dict(f.robots)) for f in frames
    ]


def _slot(name, state, *, position=(100, 100), observed_at=1.0, is_predicted=None):
    """造一个槽位。``is_predicted`` 默认按状态推——要造违规样本时才显式给。"""
    return RobotSlotOutput(
        name=name,
        position_cm=position,
        state=state,
        confidence=0.9,
        observed_at=observed_at,
        source=SourceKind.VISION,
        is_predicted=state.is_prediction if is_predicted is None else is_predicted,
    )


def _frame(seq, slots, *, captured_at=None):
    return FrameOutput(
        frame_seq=seq,
        captured_at=seq * 0.05 if captured_at is None else captured_at,
        published_at=seq * 0.05 if captured_at is None else captured_at,
        robots={s.name: s for s in slots},
    )


# ---------------------------------------------------------------------------
# 金标准
# ---------------------------------------------------------------------------
def test_the_replay_reproduces_the_golden_state_ladders():
    """整条链路跑出来的状态阶梯必须与金标准逐字相同。"""
    frames, loop = _run()
    ladders = summarise(frames, loop.diagnostics())["state_ladders"]

    assert len(frames) == FRAMES
    # 先钉槽位数：某个槽位整个消失（发都不发）也是回归，但它不会体现在
    # 上面那张"非全 MISSING"的对照表里，会悄悄漏过去。
    assert len(ladders) == 12

    notable = {name: seq for name, seq in ladders.items() if seq != ALL_MISSING}
    assert notable == GOLDEN_LADDERS

    # 默认时钟下帧龄恒为 0，所以**不该**有任何一帧因为超龄被丢。
    # 这条和下面那个注入时钟的测试是一对：那个证明门会关，这个证明门
    # 不该无端关。
    assert loop.dropped_stale == 0


def test_every_emitted_frame_satisfies_the_output_contract():
    """跑出来的每一帧都要过契约。违规要指名道姓，不能只说"失败了"。"""
    frames, _ = _run()
    assert check_contract(frames) == []


def test_the_replay_is_deterministic_in_everything_but_timing():
    """同一份输入跑两次，上报内容必须逐字节相同。

    这是整个 M6 阶段把默认跑法定为单线程的理由（见 ``event_loop`` 模块文档）：
    只有结果可复现，金标准比对和"这一帧为什么变成 HOLD"的排查才有意义。
    """
    first, _ = _run()
    second, _ = _run()
    assert _contract_view(first) == _contract_view(second)


def test_the_threaded_loop_agrees_with_the_single_threaded_loop():
    """线程版与单线程版给出**同样的**输出。

    这条是实时部署前唯一能查到"并发悄悄改坏了共享状态"的手段——那种 bug
    在单线程里根本不出现，上了实车才以"偶尔报个错坐标"的形式冒出来。

    能这么比，靠的是线程版的生产者按录制速度节流（``ThreadedEventLoop.pace``）。
    不节流的话生产者会在消费线程启动前就把 96 帧全喂完，信箱按设计只留最新
    的一帧，整段回放只剩 3 帧——那时这条断言比的是噪声。
    """
    single, _ = _run(threaded=False)
    threaded, loop = _run(threaded=True)

    assert _contract_view(single) == _contract_view(threaded)
    # 输出一样还不够：如果线程版是靠"丢了一大半、剩下的碰巧一样"达成的，
    # 那它一样得毫无意义。信箱必须一个包都没丢。
    assert loop.diagnostics()["mailbox_dropped"] == 0


# ---------------------------------------------------------------------------
# 超龄提交
# ---------------------------------------------------------------------------
def test_an_over_age_frame_is_dropped_before_it_reaches_the_tracker():
    """帧龄越限时整帧丢弃，**并且跟踪器一点都不许动**。

    时钟是注入的：默认 ``frame_clock`` 下帧龄恒为 0，这条保护永远不触发，
    直接跑一遍只能证明"没炸"，证明不了门会关。

    "不许动"是这里真正的重点。丢掉旧帧只解决了一半问题——如果跟踪器已经
    拿它推进过状态，轨迹上就留下了那个已经被判定为"太旧、不该采信"的位置，
    而且再也擦不掉。所以 ``StageB.process`` 是在超龄分支里**就地返回**的。
    """
    limit = float(_config().pipeline.max_commit_age_s)

    # 比阈值晚一点点：这一帧已经旧到不该采信
    stale = lambda packet: packet.captured_at + limit + 0.05  # noqa: E731
    frames, loop = _run(clock=stale)

    assert frames == []
    assert loop.dropped_stale == FRAMES
    # 一帧都没提交 → 跟踪器一帧都没被喂 → 一条轨迹都不该存在
    assert loop.diagnostics().get("tracks", 0) == 0


def test_a_frame_just_under_the_age_limit_is_still_committed():
    """差一点没超龄的帧必须照常发出。门是 "严格大于"，不是 "接近"。"""
    limit = float(_config().pipeline.max_commit_age_s)

    fresh = lambda packet: packet.captured_at + limit - 0.05  # noqa: E731
    frames, loop = _run(clock=fresh)

    assert len(frames) == FRAMES
    assert loop.dropped_stale == 0


# ---------------------------------------------------------------------------
# 契约检查器本身
# ---------------------------------------------------------------------------
def test_the_contract_checker_catches_an_observed_at_regression():
    """``observed_at`` 回退必须被抓住。

    这是 ``check_contract`` 唯一**比** ``validate()`` 多做的那件事：它要跨帧
    比较，而单帧的 ``validate()`` 结构上看不到上一帧。I1/I2 都在 ``types.py``
    里，这里不重复实现——重复过的代价见 ``check_contract`` 的文档。

    回退的危险在于下游是按 ``published_at - observed_at`` 判断信息新鲜度的：
    时间倒流会让它算出负的年龄，于是最旧的数据被当成最新的采信。
    """
    fresh = [_frame(0, [_slot("B1", RobotState.MEASURED, observed_at=5.0)])]
    rewound = [_frame(1, [_slot("B1", RobotState.MEASURED, observed_at=4.0)])]
    assert check_contract(fresh + rewound) != []

    # 对照：不回退就不该报
    normal = [_frame(1, [_slot("B1", RobotState.MEASURED, observed_at=5.1)])]
    assert check_contract(fresh + normal) == []

    # MISSING 槽位不该被误判。曾经这里对着每一个 MISSING 都报"I1 不自洽"
    # （检查器抄的是方案原文那句 `is_predicted ⇔ state != MEASURED`），
    # 于是健康的回放跑出退出码 2——一个常态误报的 CI 门等于没有门。
    missing = [_frame(0, [_slot("R1", RobotState.MISSING, position=None)])]
    assert check_contract(missing) == []


def test_the_runner_exits_zero_on_a_healthy_replay():
    """命令行工具的退出码就是 CI 里那道门，它自己也得被测。

    退出码 2 表示跑了但违反契约。如果它在健康输入上就是 2，那么真正的契约
    违规混进来时没人分得出来——门照样是坏的。
    """
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "replay_runner.py"),
         "--frames", str(default_replay_dir()), "--quiet"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
