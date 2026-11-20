#!/usr/bin/env python3
"""用 2027 规则的 P 递推模型，比较「断流发 (0,0)」与「断流发旧值」。

**这不是实测**：输入是一段可复算的场景（匀速目标 + 固定断流窗口 + 固定噪声），
P 的计算逻辑逐字来自规则手册表 5-29。结论只能表述为"在这组场景假设下"。
真实链路的帧日志可以用 ``--mode log`` 直接喂进来，接真机时不用改模型。

用法::

    # 场景网格 + 曲线 + 报告
    .venv/bin/python tools/marking_sim.py --mode grid --out out/marking_sim/20261121

    # 把一次实跑的 0x0305 落盘日志算成 P(t)（需要给一条假设的匀速真值轨迹）
    .venv/bin/python tools/marking_sim.py --mode log --referee-log out/referee_0305.jsonl \\
        --slot B1 --speed-mps 2.0 --out out/marking_sim/log_20261121
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jg_radar.eval.marking_progress import Judgement, MarkingTracker  # noqa: E402
from jg_radar.eval.marking_scenario import (  # noqa: E402
    Scenario,
    StalePolicy,
    compare,
    compare_time,
    distance_m,
    run,
    run_ideal,
)
from tools.radio_sweep import svg_line_chart  # noqa: E402

DEFAULT_SPEEDS = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)
DEFAULT_OUTAGES = (0.3, 0.5, 1.0, 2.0)
DEFAULT_OUT = ROOT / "out" / "marking_sim" / datetime.now().strftime("%Y%m%d_%H%M")


def _rows_for(scenario: Scenario) -> list[dict[str, object]]:
    """同一场景下三条线：drop、send_stale、无断流基准。

    每行都带 ``p_at_compare`` 和「相对无断流基准少掉的 P」——把断流的代价
    和"速度导致的延迟误差"分开，否则 4m/s 那条线看起来像断流伤得特别重，
    其实是 0.2s 延迟本身就跨到了半准确。
    """
    ideal = run_ideal(scenario)
    # 基准必须和策略取**同一个时刻**：无断流场景的 compare_time 会早 0.5 秒
    baseline = ideal.p_at(compare_time(scenario))
    rows = []
    for policy in StalePolicy:
        result = run(scenario, policy)
        summary = result.summary()
        summary["ideal_p_at_compare"] = round(baseline, 3)
        summary["cost_vs_ideal"] = round(baseline - result.p_at_compare(), 3)
        rows.append(summary)
    summary = ideal.summary()
    summary["policy"] = "ideal"
    summary["ideal_p_at_compare"] = round(baseline, 3)
    summary["cost_vs_ideal"] = 0.0
    rows.append(summary)
    return rows


def run_grid(speeds, outages, *, send_hz: float) -> tuple[list[dict], list[Scenario]]:
    rows: list[dict] = []
    scenarios: list[Scenario] = []
    for outage in outages:
        for speed in speeds:
            scenario = Scenario(
                send_hz=send_hz,
                horizon_s=12.0,
                outage_s=outage,
                speed_mps=speed,
            )
            scenarios.append(scenario)
            rows.extend(_rows_for(scenario))
    return rows, scenarios


def crossover_drift(*, outage_s: float, send_hz: float, limit_mps: float = 6.0,
                    step_mps: float = 0.1) -> float | None:
    """``outage_s`` 断流下，``send_stale`` 从哪一段漂移开始不如 ``drop``。

    返回最后一个"send_stale 不劣于 drop"的漂移距离（米）。没找到就返回 None。
    """
    best = None
    speed = 0.0
    while speed <= limit_mps + 1e-9:
        scenario = Scenario(send_hz=send_hz, outage_s=outage_s, speed_mps=speed)
        outage_end = scenario.outage_start_s + scenario.outage_s
        results = compare(scenario)
        stale = results[StalePolicy.SEND_STALE].p_at(results[StalePolicy.SEND_STALE].compare_time)
        drop = results[StalePolicy.DROP].p_at(results[StalePolicy.DROP].compare_time)
        if stale >= drop:
            best = scenario.drift_m
        else:
            break  # 第一次翻转就停：再往后查出来的"又成立"是数值抖动，不是结论
        speed += step_mps
    return best


def render_report(rows, scenarios, *, send_hz: float, crossovers: dict[float, float | None]) -> str:
    lines = [
        "# drop vs send_stale：断流时该发 (0,0) 还是发旧值",
        "",
        "> **数据性质：合成场景**。目标坐标由匀速直线运动 + 固定种子噪声生成，断流窗口是",
        "> 人为设定的；P 的计算逻辑逐字来自《RoboMaster 2027 规则手册 V1.0.0》表 5-29。",
        "> 因此本文所有秒数都是「在这组场景假设下」的结果，不是赛场实测。",
        "",
        "## 模型与假设",
        "",
        f"- 发送/判定频率 **{send_hz} Hz**（等于我们的默认发送频率，规则原文：判定频率由雷达发送频率决定）",
        "- 准确 `d<0.8m`：上一次准确/半准确 → `x+=1.0`，否则 `x=1.0`；`P+=x`",
        "- 半准确 `0.8≤d<1.6m`：上一次准确/半准确 → `x+=0.5`，否则 `x=0.5`；`P+=x`",
        "- 错误 `d≥1.6m`：上一次错误/中断 → `x-=0.8`，否则 `x=-0.8`；`P+=x`",
        "- 数据中断：每累计 0.5 秒没收到该车数据，视为一次「错误」；`P` 限幅 `[0,150]`",
        "- 场景假设：端到端延迟 0.2s、位置噪声 σ=0.1m、断流从第 3.0 秒开始、跑 12 秒",
        "",
        "## 结论",
        "",
        "**结论是条件句，不是一句「drop 更好」**：",
        "",
        "- `x` 是**动量不是计分器**：连续准确时 `x` 每次 +1.0 线性增长，`P` 二次增长；"
        "半准确继续 +0.5，**P 还在涨**。所以「发了旧值但只偏了 1 米」不但不掉，反而可能"
        "让 P 涨得更多（`x` 越大，一次半准确加得越多）。",
        "- 一旦某帧 `d ≥ 1.6m`，`x` 立刻被**打回 −0.8**（不是减掉 0.8）——前面整段累积归零，"
        "这才是断流真正的代价。之后连续错误让 `|x|` 每次再 +0.8，`P` 越掉越快。",
        "- `drop` 是每 0.5 秒补一次中断（2Hz 判定），`send_stale` 是每个发送周期判一次"
        f"（{send_hz:g}Hz）。同样是判错，`send_stale` 的衰减速度是 `drop` 的 "
        f"{send_hz:g}/2 ≈ {send_hz / 2:.1f} 倍——**所以「一旦开始判错」它是崩得更快的那个**。",
        "- 反过来，断流短于 0.5 秒时 `drop` 连一次中断都不会触发（几乎零代价）；"
        "这时只有旧坐标还落在 1.6m 内，`send_stale` 才划算。",
        "",
        "换句话说：**旧坐标准就用它、不准就别发**。默认 `drop` 是在「不知道目标跑没跑」"
        "时的保守选择，不是最优解；真正的判据是断流期间旧坐标漂到了多远，而这个量"
        "需要速度估计——所以「按速度切换策略」进 backlog，现在不做。",
        "",
    ]
    if crossovers:
        lines += ["各断流时长下 `send_stale` 仍然划算的最大漂移：", "",
                  "| 断流时长 (s) | 最大可用漂移 (m) |", "| --- | --- |"]
        for outage, drift in sorted(crossovers.items()):
            value = "未找到（一直更差）" if drift is None else f"{drift:.2f}"
            lines.append(f"| {outage:g} | {value} |")
        lines.append("")

    lines += ["## 首次到达 100 / 120 的时间", "",
              "时间从第 0 秒的第一拍算起（我们 4.7Hz 发出第一帧就在 0 秒），所以冷启动",
              "`t100` 是 2.77s 而不是研究员那份从 0.2s 起算的 2.98s——两者是同一个结论。", "",
              "对比时刻取「断流结束后 0.5 秒」——两边到那时都已恢复数据；`代价` 是同速度、"
              "无断流基准减掉的实际 P。",
              "",
              "| 断流 (s) | 速度 (m/s) | 漂移 (m) | 策略 | P@对比时刻 | 代价 | 首次 100 (s) | 首次 120 (s) | 错误/中断 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        if row["policy"] == "ideal":
            continue
        lines.append(
            f"| {row['outage_s']:g} | {row['speed_mps']:g} | {row['drift_m']:.2f} | {row['policy']} | "
            f"{row.get('p_at_compare', '')} | {row.get('cost_vs_ideal', '')} "
            f"| {row['first_100_s'] if row['first_100_s'] is not None else '—'} "
            f"| {row['first_120_s'] if row['first_120_s'] is not None else '—'} "
            f"| {row['error']}/{row['interrupt']} |"
        )
    lines += ["", "完整数据见 `results.csv`，曲线见 `curves.svg`、`crossover.svg`。", ""]
    return "\n".join(lines)


def grid_figures(rows, *, send_hz: float) -> dict[str, str]:
    """两张图：代表性场景的 P(t)，以及"首次到 100"随漂移的变化。"""
    # 图里的标签刻意用 ASCII：SVG 转 PNG 时中文常因缺 CJK 字体掉字，
    # 而图是给论文/汇报直接用的，标签得在哪儿都渲染得出来。
    show = [
        ("stationary", Scenario(speed_mps=0.0, outage_s=2.0)),
        ("2m/s 1s outage", Scenario(speed_mps=2.0, outage_s=1.0)),
    ]
    curves = []
    colors = {
        (StalePolicy.DROP, 0): "#c0392b",
        (StalePolicy.SEND_STALE, 0): "#2471a3",
        (StalePolicy.DROP, 1): "#e67e22",
        (StalePolicy.SEND_STALE, 1): "#27ae60",
    }
    for index, (label, scenario) in enumerate(show):
        for policy in StalePolicy:
            result = run(scenario, policy)
            curves.append({
                "label": f"{label} {policy.value}",
                "points": result.series(),
                "color": colors[(policy, index)],
            })
        ideal = run_ideal(scenario)
        curves.append({"label": f"{label} no-outage", "points": ideal.series(), "color": "#7f8c8d"})
    chart1 = svg_line_chart(curves, title="P(t): send (0,0) vs send stale during an outage",
                            x_label="time (s)", y_label="P")

    speed_axis = [0.5 * index for index in range(11)]  # 0 ~ 5 m/s
    series = []
    for label, outage, color in (("outage 0.5s", 0.5, "#2471a3"), ("outage 2.0s", 2.0, "#c0392b")):
        drop_points, stale_points = [], []
        for speed in speed_axis:
            scenario = Scenario(send_hz=send_hz, outage_s=outage, speed_mps=speed)
            drift = scenario.drift_m
            drop = run(scenario, StalePolicy.DROP).first_time_at(100.0)
            stale = run(scenario, StalePolicy.SEND_STALE).first_time_at(100.0)
            if drop is not None:
                drop_points.append((drift, drop))
            if stale is not None:
                stale_points.append((drift, stale))
        series.append({"label": f"{label} / drop", "points": drop_points, "color": color})
        series.append({"label": f"{label} / send_stale", "points": stale_points, "color": color})
    chart2 = svg_line_chart(series, title="Time to P=100 vs drift during the outage",
                            x_label="drift (m)", y_label="time to P=100 (s)")
    return {"curves.svg": chart1, "crossover.svg": chart2}


def write_outputs(out: Path, *, rows, scenarios, send_hz: float, crossovers) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fields = ["policy", "outage_s", "speed_mps", "drift_m", "p_end", "p_at_compare", "ideal_p_at_compare",
              "cost_vs_ideal", "p_max", "p_min",
              "first_100_s", "first_120_s", "account_js", "half", "error", "interrupt"]
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join("" if row.get(name) is None else str(row.get(name)) for name in fields))
    (out / "results.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for name, svg in grid_figures(rows, send_hz=send_hz).items():
        (out / name).write_text(svg, encoding="utf-8")

    (out / "report.md").write_text(
        render_report(rows, scenarios, send_hz=send_hz, crossovers=crossovers), encoding="utf-8"
    )
    (out / "meta.json").write_text(json.dumps({
        "send_hz": send_hz,
        "scenarios": len(scenarios),
        "crossovers": {str(k): v for k, v in crossovers.items()},
        "data_nature": "synthetic-scenario; P model from 2027 rules manual table 5-29",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 真实日志模式：把 --referee-log（run_radio_vision 的 0x0305 落盘）算成 P(t)
# ---------------------------------------------------------------------------
def run_log(path: Path, *, slot: str, speed_mps: float, send_hz_hint: float) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} 是空的")
    tracker = MarkingTracker()
    sent = 0
    for row in rows:
        t = float(row["t"])
        position = (row.get("slots") or {}).get(slot)
        if not position or (position[0] == 0 and position[1] == 0):
            tracker.silence_until(t)
            continue
        # 真值：一条假设的匀速直线轨迹（真实日志没有真值，必须显式声明）
        truth = (speed_mps * max(t - float(rows[0]["t"]), 0.0), 0.0)
        tracker.data(t, distance_m((position[0] / 100.0, position[1] / 100.0), truth))
        sent += 1
    return {
        "frames": len(rows),
        "sent": sent,
        "slot": slot,
        "speed_mps": speed_mps,
        "p_end": tracker.p,
        "first_100_s": tracker.first_time_at(100.0),
        "first_120_s": tracker.first_time_at(120.0),
        "counts": {
            judgement.value: sum(1 for sample in tracker.samples if sample.judgement is judgement)
            for judgement in Judgement
        },
        "truth_assumption": "匀速直线，从第一条帧时刻起算；真实日志没有真值，这个数只验证管线",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("grid", "log"), default="grid")
    parser.add_argument("--send-hz", type=float, default=4.7)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--speeds", default=",".join(str(v) for v in DEFAULT_SPEEDS))
    parser.add_argument("--outages", default=",".join(str(v) for v in DEFAULT_OUTAGES))
    parser.add_argument("--referee-log", default="", help="log 模式：run_radio_vision --referee-log 生成的文件")
    parser.add_argument("--slot", default="B1", help="log 模式：看哪个槽位")
    parser.add_argument("--speed-mps", type=float, default=2.0, help="log 模式：假设的真值速度")
    parser.add_argument("--no-crossover", action="store_true", help="跳过交叉点扫描（省时间）")
    args = parser.parse_args(argv)

    out = Path(args.out)
    if args.mode == "log":
        if not args.referee_log:
            parser.error("log 模式需要 --referee-log")
        summary = run_log(Path(args.referee_log), slot=args.slot, speed_mps=args.speed_mps,
                          send_hz_hint=args.send_hz)
        out.mkdir(parents=True, exist_ok=True)
        (out / "log_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
        print(json.dumps({"mode": "log", "out": str(out), **summary}, ensure_ascii=False))
        return 0

    speeds = [float(item) for item in args.speeds.split(",") if item.strip()]
    outages = [float(item) for item in args.outages.split(",") if item.strip()]
    rows, scenarios = run_grid(speeds, outages, send_hz=args.send_hz)

    crossovers: dict[float, float | None] = {}
    if not args.no_crossover:
        crossovers = {
            outage: crossover_drift(outage_s=outage, send_hz=args.send_hz) for outage in outages
        }
    write_outputs(out, rows=rows, scenarios=scenarios, send_hz=args.send_hz, crossovers=crossovers)

    print(json.dumps({
        "mode": "grid",
        "out": str(out),
        "scenarios": len(scenarios),
        "rows": len(rows),
        "crossovers": crossovers,
        "data_nature": "synthetic-scenario",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
