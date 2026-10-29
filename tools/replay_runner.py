#!/usr/bin/env python3
"""离线回放：一份回放目录 → 一帧帧的对外输出。

    python tools/replay_runner.py --frames tests/fixtures/replay_seq_001 \
        --config config/default.yaml --out out.jsonl

这是**不需要相机、不需要权重、不需要 GPU** 就能跑完整条流水线的那条路。
存在的理由是方案 §九 的第 7 条：四家开源方案的共同空白是测试，而"改了一行
变换代码，坐标到底变了多少"这种事，只有端到端跑一遍才知道。

## 它同时是排查工具

``--threaded`` 可以切到实时的双线程跑法。两种跑法在同一份输入上给出**同样的
输出**——逐帧同样的状态阶梯。对不上就说明时钟或者状态被共享了，而那是实时
部署里最难查的一类问题。

这条比对能成立，是因为线程版的生产者被**按录制速度节流**了（见
``ThreadedEventLoop.pace``）。不节流的话生产者会在消费线程启动之前就把 96 帧
全喂完，信箱按设计只留最新的一帧，于是回放只剩下 3 帧——对比就成了噪声。
``--no-pace`` 关掉节流（跑得快得多，但会大量丢帧），只在想看信箱丢帧行为时用。

退出码：0 正常；1 输入有问题（目录缺文件、配置读不了）；2 跑出来的结果
**违反输出契约**（不变式被破）——那种情况必须让 CI 红掉，不能只是打印一行。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jg_radar.config.loader import load_config  # noqa: E402
from jg_radar.io.frame_source import ReplayFrameSource  # noqa: E402
from jg_radar.io.sink import JsonlSink  # noqa: E402
from jg_radar.pipeline.build import build_loop  # noqa: E402
from jg_radar.types import FrameOutput  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="雷达视觉流水线的离线回放")
    p.add_argument("--frames", default="",
                   help="回放目录（含 manifest.json）。不给自己从配置的 io 段取帧源"
                        "（例如 io.frame_source=video 跑真实录像）")
    p.add_argument(
        "--config", default=str(ROOT / "config" / "default.yaml"), help="主配置"
    )
    p.add_argument(
        "--camera", default=str(ROOT / "config" / "camera.yaml"),
        help="相机配置；回放目录自带标定时会被忽略",
    )
    p.add_argument(
        "--field", default=str(ROOT / "config" / "field.yaml"), help="场地配置"
    )
    p.add_argument("--out", default="", help="输出 JSONL 路径（不给就只打印摘要）")
    p.add_argument(
        "--threaded", action="store_true",
        help="用实时那套双线程跑法（默认单线程、确定性）",
    )
    p.add_argument(
        "--no-pace", action="store_true",
        help="线程版的生产者不按录制速度节流（快，但信箱会丢大量帧。"
             "默认节流，好让两种跑法逐帧可比）",
    )
    p.add_argument("--quiet", action="store_true", help="只打印最后的摘要")
    return p.parse_args(argv)


def check_contract(frames: list[FrameOutput]) -> list[str]:
    """验输出契约。**返回的是违规描述，不是布尔**。

    只说"失败了"没法排查。要把"哪一帧、哪个槽位、违反了什么"讲清楚。

    分两段，各管各的：

    - **帧内**：交给 ``RobotSlotOutput.validate()``。I1（``is_predicted`` 与
      ``state`` 自洽）、I2（坐标不越界）都在那里，而且**只在那一处**。
    - **跨帧**：发布时刻与发布序号单调；同来源有效观测时间不倒退。
      换来源允许采用较旧样本，真实年龄不得由发布层修改。

    这里**刻意不重复实现 I1**。写过一版，用的是方案原文那句
    ``is_predicted ⇔ state != MEASURED``，而 ``types.py`` 后来把它收紧成了
    ``state.is_prediction``（``MISSING`` 没有坐标，不是预测）。两份定义一分叉，
    这段就对着每一个 MISSING 槽位报"不自洽"——**健康的回放跑出退出码 2**。
    一个常态误报的 CI 门，唯一的下场是所有人学会忽略它。一条规则一个家。
    """
    problems: list[str] = []
    last_published = float("-inf")
    last_publication_seq = -1
    last_observed: dict[tuple, float] = {}
    for frame in frames:
        if frame.published_at < last_published:
            problems.append(f"发布时刻回退：{frame.published_at} < {last_published}")
        last_published = frame.published_at
        if frame.publication_seq is not None:
            if frame.publication_seq <= last_publication_seq:
                problems.append(f"发布序号没有递增：{frame.publication_seq}")
            last_publication_seq = frame.publication_seq
        for name, slot in frame.robots.items():
            try:
                slot.validate()
            except Exception as exc:  # noqa: BLE001 - 契约违规一律收集，不中断
                problems.append(f"第 {frame.frame_seq} 帧 {name}: {exc}")
                continue

            # 换源允许采用较旧样本，但同一来源的有效观测不得倒退。
            if slot.state.value != "missing":
                key = (name, slot.source)
                previous = last_observed.get(key, float("-inf"))
                if slot.observed_at < previous - 1e-9:
                    problems.append(f"第 {frame.frame_seq} 帧 {name}: 同源观测时间回退")
                last_observed[key] = slot.observed_at

    return problems


def summarise(frames: list[FrameOutput], diagnostics: dict[str, float]) -> dict:
    """把状态序列压成"阶梯"，直接看出每个槽位走了哪条路。

    压成一行的字符串（``missing×1 → measured×24 → …``）而不是嵌套数组：
    这份报告是给人看的。数值那几项才是给机器看的。
    """
    ladders: dict[str, list[list]] = {}
    for frame in frames:
        for name, slot in frame.robots.items():
            seq = ladders.setdefault(name, [])
            if not seq or seq[-1][0] != slot.state.value:
                seq.append([slot.state.value, 1])
            else:
                seq[-1][1] += 1
    return {
        "frames": len(frames),
        "slots_reported": sorted({n for f in frames for n in f.robots}),
        "state_ladders": {
            n: " → ".join(f"{state}×{n_}" for state, n_ in seq)
            for n, seq in sorted(ladders.items())
        },
        "switches": int(diagnostics.get("arbiter_switches", 0)),
        "dropped_stale": int(diagnostics.get("dropped_stale", 0)),
        "max_commit_age_ms": round(float(diagnostics.get("max_commit_age_ms", 0.0)), 4),
        "tracks": int(diagnostics.get("tracks", 0)),
    }


def main(argv=None) -> int:
    args = parse_args(argv)

    # --frames 给的是**回放目录**（检测结果已经算好）。不给就是让配置自己
    # 决定帧源——例如 config/ega_replay.yaml 里的 io.frame_source=video，
    # 那种情况下每帧的检测是**真跑**的，可能需要几十秒到几分钟。
    source = None
    if args.frames:
        frames_dir = Path(args.frames)
        if not (frames_dir / "manifest.json").is_file():
            print(f"回放目录里没有 manifest.json：{frames_dir}", file=sys.stderr)
            return 1
        source = ReplayFrameSource(frames_dir)

    paths = [args.config]
    # 回放目录/视频自带标定的话，相机配置会被忽略（build_loop 从帧源取标定）。
    # 这里仍然把 camera.yaml 传进去，是为了让 track/fuse/pipeline 那些段
    # 在只有一份 camera.yaml 的裁剪环境下也能跑起来。
    # 用 --camera "" --field "" 可以显式跳过（空串不是文件）。
    for extra in (args.camera, args.field):
        if extra and Path(extra).is_file() and extra not in paths:
            paths.append(extra)

    try:
        cfg = load_config(*paths)
    except Exception as exc:  # noqa: BLE001
        print(f"配置读不了：{exc}", file=sys.stderr)
        return 1

    sink = JsonlSink(args.out) if args.out else None
    try:
        loop = build_loop(cfg, frame_source=source, sink=sink, threaded=args.threaded)
    except Exception as exc:  # noqa: BLE001
        # 装配失败（模型路径不对、视频打不开、mesh 缺失……）是**输入问题**，
        # 退出码 1；契约违规才是 2。分开是为了让 CI 能区分"没跑起来"和"跑错了"。
        print(f"装配失败：{exc}", file=sys.stderr)
        return 1
    if args.no_pace and hasattr(loop, "pace"):
        loop.pace = 0.0

    emitted: list[FrameOutput] = []
    for frame in loop.run():
        emitted.append(frame)
        if not args.quiet and not args.out:
            states = " ".join(
                f"{n}={s.state.value[:4]}" for n, s in sorted(frame.robots.items())
                if s.state.value != "missing"
            )
            print(f"帧 {frame.frame_seq:4d}  {states or '（全 MISSING）'}")

    report = summarise(emitted, loop.diagnostics())
    print(json.dumps(report, ensure_ascii=False, indent=2))

    problems = check_contract(emitted)
    if problems:
        # 契约违规必须让 CI 红掉。打印出来只是让人看得见，退出码才是拦住它的东西。
        for line in problems[:20]:
            print(f"契约违规：{line}", file=sys.stderr)
        if len(problems) > 20:
            print(f"...另有 {len(problems) - 20} 条", file=sys.stderr)
        return 2

    if args.out:
        counts = Counter(
            s.state.value for f in emitted for s in f.robots.values()
        )
        print(f"已写出 {args.out}；状态计数 = {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
