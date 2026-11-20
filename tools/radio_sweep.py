#!/usr/bin/env python3
"""在回放链路上扫解调参数，输出 CSV、曲线和一份可读报告。

用 `tools/radio_replay.py` 把同一段 IQ 反复送进真实流图，只改 `loop_bw`、
`bt` 这类生成期参数，统计三级解码计数。信号来源两种：

- `--mode synth`：由 `tools/iq_synth.py` 合成 GFSK 基线，按 `--snr` 逐档加
  复高斯白噪声。没有现场信号时用这个。
- `--mode file --iq xxx.c64`：回放真实录制的 IQ，`--snr` 表示在录制上**附加**
  多少噪声（录制本身已含噪声，所以这是上界，不是真实 SNR）。

指标口径（都按消耗的比特归一，与墙钟速度无关）：

- `sync_rate`   空口帧同步率 = 空口帧 / (比特 / 空口帧周期)
- `packet_yield` 业务包产出率 = 业务包 / 期望业务包（合成模式才有的真值）
- `packets_per_kbit` 每千比特解出的业务包，跨数据源可比的主指标

用法::

    source ~/.jg_env.zsh
    python3 tools/radio_sweep.py --mode synth \\
        --snr 0,5,10,15,20,25,30 \\
        --loop-bw 0.001,0.002,0.005,0.01,0.02 \\
        --bt 0.25,0.35,0.45 \\
        --out out/radio_sweep/20261120
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.iq_synth import (  # noqa: E402
    DEFAULT_MOD_INDEX,
    DEFAULT_SAMP_RATE,
    DEFAULT_SPS,
    FRAME_BITS,
    SynthSpec,
    add_noise,
    normalize_power,
    read_iq,
    synthesize,
    write_iq,
)
from tools.radio_replay import replay_once  # noqa: E402

DEFAULT_LOOP_BW = (0.001, 0.002, 0.005, 0.01, 0.02)
DEFAULT_BT = (0.25, 0.35, 0.45)
DEFAULT_SNR = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)

#: 报告里认定「跑通」的产出率下限，用来定门限 SNR。
WORKING_YIELD = 0.9

PALETTE = (
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#17becf", "#8c564b", "#7f7f7f",
)


# ==================== 网格与单点 ====================


@dataclass(frozen=True)
class SweepPoint:
    """一个待测的参数组合。"""

    loop_bw: float
    bt: float
    snr_db: float | None = None

    @property
    def label(self) -> str:
        snr = "无附加噪声" if self.snr_db is None else f"SNR {self.snr_db:g} dB"
        return f"{snr} / loop_bw={self.loop_bw:g} / bt={self.bt:g}"

    def overrides(self) -> dict[str, str]:
        return {"loop_bw": repr(self.loop_bw), "bt": repr(self.bt)}


def expand_grid(
    *,
    snr_values: tuple[float | None, ...] = (None,),
    loop_bw_values: tuple[float, ...] = DEFAULT_LOOP_BW,
    bt_values: tuple[float, ...] = DEFAULT_BT,
) -> list[SweepPoint]:
    """展开成参数网格，顺序稳定（先 SNR，再环路带宽，最后 BT）。"""
    return [
        SweepPoint(loop_bw=loop_bw, bt=bt, snr_db=snr)
        for snr in snr_values
        for loop_bw in loop_bw_values
        for bt in bt_values
    ]


@dataclass
class SweepRow:
    """一个参数点的测量结果。"""

    point: SweepPoint
    bits: int
    sync: int
    frames: int
    decoded: int
    seconds: float
    frame_period_symbols: int = FRAME_BITS
    serial_frames_per_air: float | None = None

    @property
    def expected_air_frames(self) -> float:
        return self.bits / self.frame_period_symbols if self.bits else 0.0

    @property
    def sync_rate(self) -> float:
        expected = self.expected_air_frames
        return self.sync / expected if expected else 0.0

    @property
    def packets_per_kbit(self) -> float:
        return self.decoded / self.bits * 1000.0 if self.bits else 0.0

    @property
    def packet_yield(self) -> float | None:
        """业务包 / 期望业务包。只有知道每空口帧承载多少串口帧时才算得出。"""
        if not self.serial_frames_per_air:
            return None
        expected = self.expected_air_frames * self.serial_frames_per_air
        return self.decoded / expected if expected else 0.0

    @property
    def snr_db(self) -> float | None:
        return self.point.snr_db

    def as_dict(self) -> dict:
        yield_value = self.packet_yield
        return {
            "snr_db": "" if self.snr_db is None else f"{self.snr_db:g}",
            "loop_bw": f"{self.point.loop_bw:g}",
            "bt": f"{self.point.bt:g}",
            "bits": self.bits,
            "sync": self.sync,
            "frames": self.frames,
            "decoded": self.decoded,
            "expected_air_frames": round(self.expected_air_frames, 1),
            "sync_rate": round(self.sync_rate, 4),
            "packet_yield": "" if yield_value is None else round(yield_value, 4),
            "packets_per_kbit": round(self.packets_per_kbit, 4),
            "seconds": round(self.seconds, 2),
        }


def evaluate_point(
    point: SweepPoint,
    iq_file: Path,
    *,
    target_bits: int,
    timeout: float,
    serial_frames_per_air: float | None = None,
    frame_period_symbols: int = FRAME_BITS,
    protocol: str = "2027",
) -> SweepRow:
    """跑一个参数点，返回计数。"""
    result = replay_once(
        iq_file,
        overrides=point.overrides(),
        target_bits=target_bits,
        timeout=timeout,
        protocol=protocol,
    )
    info = result.info
    return SweepRow(
        point=point,
        bits=info.bits,
        sync=info.sync,
        frames=info.frames,
        decoded=info.decoded,
        seconds=result.seconds,
        frame_period_symbols=frame_period_symbols,
        serial_frames_per_air=serial_frames_per_air,
    )


# ==================== 汇总 ====================


def axis_values(rows: list[SweepRow], attribute: str) -> list[float]:
    """取某个轴上的所有取值（去重、排序）。"""
    values = {getattr(row.point, attribute, None) for row in rows}
    return sorted(value for value in values if value is not None)


def axis_means(rows: list[SweepRow], attribute: str) -> dict[float, float]:
    """某个轴上每个取值的平均产出率。"""
    buckets: dict[float, list[float]] = {}
    for row in rows:
        value = getattr(row.point, attribute)
        if value is None:
            continue
        buckets.setdefault(value, []).append(row.packets_per_kbit)
    return {value: sum(items) / len(items) for value, items in sorted(buckets.items())}


def sensitivity(rows: list[SweepRow], attributes: tuple[str, ...] = ("snr_db", "loop_bw", "bt")) -> list[tuple[str, float]]:
    """按「轴上均值的极差 / 全体均值」给参数排序，数值越大越敏感。"""
    overall = sum(row.packets_per_kbit for row in rows) / len(rows) if rows else 0.0
    ranking = []
    for attribute in attributes:
        means = list(axis_means(rows, attribute).values())
        if len(means) < 2 or overall <= 0:
            ranking.append((attribute, 0.0))
            continue
        ranking.append((attribute, (max(means) - min(means)) / overall))
    return sorted(ranking, key=lambda item: item[1], reverse=True)


def best_row(rows: list[SweepRow]) -> SweepRow | None:
    """产出率最高的参数点；并列时取环路带宽更小的（更稳）。"""
    if not rows:
        return None
    return max(rows, key=lambda row: (row.packets_per_kbit, -row.point.loop_bw))


def threshold_snr(rows: list[SweepRow], loop_bw: float, bt: float) -> float | None:
    """给定 (loop_bw, bt) 曲线，产出率达到平台值 90% 的最低 SNR。"""
    curve = sorted(
        (
            row for row in rows
            if row.point.loop_bw == loop_bw and row.point.bt == bt and row.snr_db is not None
        ),
        key=lambda row: row.snr_db,
    )
    if not curve:
        return None
    plateau = max(row.packets_per_kbit for row in curve)
    if plateau <= 0:
        return None
    for row in curve:
        if row.packets_per_kbit >= plateau * WORKING_YIELD:
            return row.snr_db
    return None


# ==================== 输出：CSV / SVG / 报告 ====================

CSV_FIELDS = (
    "snr_db", "loop_bw", "bt", "bits", "sync", "frames", "decoded",
    "expected_air_frames", "sync_rate", "packet_yield", "packets_per_kbit", "seconds",
)


def write_csv(rows: list[SweepRow], path: Path) -> Path:
    """把全部参数点写成 CSV，一行一个点。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    return path


def read_csv(path: Path, *, serial_frames_per_air: float | None = None) -> list[SweepRow]:
    """把 `results.csv` 读回成结果行——重建报告时不必重跑流图。"""
    rows: list[SweepRow] = []
    with open(path, encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            rows.append(
                SweepRow(
                    point=SweepPoint(
                        loop_bw=float(record["loop_bw"]),
                        bt=float(record["bt"]),
                        snr_db=float(record["snr_db"]) if record["snr_db"] else None,
                    ),
                    bits=int(record["bits"]),
                    sync=int(record["sync"]),
                    frames=int(record["frames"]),
                    decoded=int(record["decoded"]),
                    seconds=float(record["seconds"]),
                    serial_frames_per_air=serial_frames_per_air,
                )
            )
    return rows


def _nice_bounds(values: list[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return low - 1.0, high + 1.0
    pad = (high - low) * 0.08
    return low - pad, high + pad


def svg_line_chart(
    series: list[dict],
    *,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    width: int = 760,
    height: int = 420,
) -> str:
    """用纯 SVG 画折线图，不依赖 matplotlib。

    `series` 每项是 ``{"label": str, "points": [(x, y), ...]}``，可选
    ``"color"``。空数据返回占位文本。
    """
    points = [point for item in series if item["points"] for point in item["points"]]
    if not points:
        return "<p>没有可画的数据</p>"

    left, right, top, bottom = 64, 16, 34, 58
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_low, x_high = _nice_bounds([point[0] for point in points])
    y_low, y_high = _nice_bounds([point[1] for point in points])
    y_low = max(y_low, 0.0)

    def project(x: float, y: float) -> tuple[float, float]:
        px = left + (x - x_low) / (x_high - x_low) * plot_width
        py = top + plot_height - (y - y_low) / (y_high - y_low) * plot_height
        return px, py

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    if title:
        parts.append(f'<text x="{left}" y="22" font-size="14" font-weight="bold" fill="#222">{title}</text>')

    for index in range(6):
        value = y_low + (y_high - y_low) * index / 5
        _, py = project(x_low, value)
        parts.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + plot_width}" y2="{py:.1f}" '
            f'stroke="#e6e6e6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{py + 4:.1f}" text-anchor="end" fill="#555">{value:.3g}</text>'
        )

    for index in range(6):
        value = x_low + (x_high - x_low) * index / 5
        px, _ = project(value, y_low)
        parts.append(
            f'<text x="{px:.1f}" y="{top + plot_height + 18}" text-anchor="middle" fill="#555">{value:.3g}</text>'
        )

    parts.append(
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#888" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
        f'stroke="#888" stroke-width="1"/>'
    )

    for order, item in enumerate(series):
        if not item["points"]:
            continue
        color = item.get("color") or PALETTE[order % len(PALETTE)]
        coords = " ".join(
            f"{project(x, y)[0]:.1f},{project(x, y)[1]:.1f}"
            for x, y in sorted(item["points"])
        )
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>'
        )
        for x, y in item["points"]:
            px, py = project(x, y)
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.5" fill="{color}"/>')
        legend_y = top + 12 + order * 16
        parts.append(
            f'<rect x="{left + plot_width - 150}" y="{legend_y - 8}" width="10" height="10" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{left + plot_width - 134}" y="{legend_y + 1}" fill="#333">{item["label"]}</text>'
        )

    if x_label:
        parts.append(
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 14}" text-anchor="middle" fill="#333">{x_label}</text>'
        )
    if y_label:
        parts.append(
            f'<text x="16" y="{top + plot_height / 2:.1f}" text-anchor="middle" fill="#333" '
            f'transform="rotate(-90 16 {top + plot_height / 2:.1f})">{y_label}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def build_report(
    rows: list[SweepRow],
    *,
    meta: dict,
) -> str:
    """生成 markdown 报告：结论、数据、曲线、方法与局限。"""
    lines = [f"# {meta.get('title', '解调参数扫描报告')}", ""]
    lines.append(f"- 数据源：{meta.get('source', '未知')}")
    lines.append(f"- 协议版本：{meta.get('protocol', '未知')}（显式指定，未用 auto）")
    lines.append(f"- 符号率：{meta.get('symbol_rate', 0) / 1000:.2f} kBd，"
                 f"每符号 {meta.get('sps', 0)} 采样")
    lines.append(f"- 每点收集：{meta.get('bits', 0)} 比特，共 {len(rows)} 个参数点")
    if meta.get("mod_index"):
        lines.append(f"- 合成调制指数：h={meta['mod_index']:g}（频偏 "
                     f"{meta.get('deviation_hz', 0) / 1000:.1f} kHz）")
    lines.append("")

    top = best_row(rows)
    if top is None:
        lines.append("没有有效数据。")
        return "\n".join(lines) + "\n"

    lines.append("## 结论")
    lines.append("")
    items: list[str] = []
    snr_text = "无附加噪声" if top.snr_db is None else f"{top.snr_db:g} dB"
    items.append(f"最优参数点：`loop_bw={top.point.loop_bw:g}`、`bt={top.point.bt:g}`、"
                 f"SNR {snr_text}，产出 {top.packets_per_kbit:.2f} 包/kbit，"
                 f"同步率 {top.sync_rate * 100:.1f}%。")

    ranking = sensitivity(rows)
    ranking_text = "、".join(
        f"`{name}`（相对极差 {value:.2f}）" for name, value in ranking
    )
    items.append(f"参数敏感度排序：{ranking_text}。相对极差是「该轴均值的极差 / 全体均值」，"
                 "越大说明这个旋钮越值得调。")

    for attribute, title in (("snr_db", "SNR"), ("loop_bw", "环路带宽 loop_bw"), ("bt", "高斯 BT")):
        means = axis_means(rows, attribute)
        if len(means) < 2:
            continue
        best_value = max(means, key=lambda key: means[key])
        worst_value = min(means, key=lambda key: means[key])
        items.append(
            f"{title}：最优 `{best_value:g}`（{means[best_value]:.2f} 包/kbit），"
            f"最差 `{worst_value:g}`（{means[worst_value]:.2f} 包/kbit）。"
        )
    for index, item in enumerate(items, 1):
        lines.append(f"{index}. {item}")

    thresholds = []
    for loop_bw in axis_values(rows, "loop_bw"):
        for bt in axis_values(rows, "bt"):
            value = threshold_snr(rows, loop_bw, bt)
            if value is not None:
                thresholds.append((loop_bw, bt, value))
    if thresholds:
        lines.append("")
        lines.append(f"{len(items) + 1}. 解波门限（产出达到该曲线平台值 "
                     f"{WORKING_YIELD * 100:.0f}% 的最低 SNR）：")
        lines.append("")
        lines.append("| loop_bw | bt | 门限 SNR |")
        lines.append("| --- | --- | --- |")
        for loop_bw, bt, value in thresholds:
            lines.append(f"| {loop_bw:g} | {bt:g} | {value:g} dB |")
        best_threshold = min(thresholds, key=lambda item: item[2])
        lines.append("")
        lines.append(f"   最低门限出现在 `loop_bw={best_threshold[0]:g}`、`bt={best_threshold[1]:g}`，"
                     f"{best_threshold[2]:g} dB。")

    lines.append("")
    lines.append("## 曲线")
    lines.append("")
    lines.append(f"![参数扫描曲线]({meta.get('chart', 'curves.svg')})")
    lines.append("")

    lines.append("## 明细（按产出率排序，前 15 个点）")
    lines.append("")
    lines.append("| SNR dB | loop_bw | bt | 空口帧 | 序列帧 | 业务包 | 同步率 | 包/kbit | 包产出率 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in sorted(rows, key=lambda item: item.packets_per_kbit, reverse=True)[:15]:
        lines.append(
            f"| {_fmt(row.snr_db, 0)} | {row.point.loop_bw:g} | {row.point.bt:g} | "
            f"{row.sync} | {row.frames} | {row.decoded} | {row.sync_rate * 100:.1f}% | "
            f"{row.packets_per_kbit:.2f} | {_fmt(row.packet_yield, 3)} |"
        )
    lines.append("")
    lines.append(f"完整结果见 `{meta.get('csv', 'results.csv')}`。")
    lines.append("")

    lines.append("## 方法与局限")
    lines.append("")
    # 调用方没给就填默认的：这几条限制比结论更需要每次出现。
    for item in meta.get("caveats") or default_caveats(meta):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def default_caveats(meta: dict) -> list[str]:
    """报告里必须写清楚的边界条件。"""
    caveats = [
        "回放链由 `tools/radio_replay.py` 从签入的 `.grc` 派生：只换源块，DSP 链、"
        "AGC、限幅、SIC、判决器全部沿用实时路径，实时路径本身一字未改。",
        "同一参数点的比特数是固定的，指标按比特归一，因此不同 CPU 速度下结果可比；"
        "但单点只有一次测量，差异小于 1 个百分点的参数点不要当成真的不同。",
        "`rx_gain` 在文件回放里不可观测——它只决定录制时的输入 SNR。增益每降 10 dB，"
        "等效 SNR 降 10 dB，所以本报告的 SNR 轴就是增益轴平移；真正的增益曲线必须在"
        "硬件上实测。",
    ]
    if meta.get("mode") == "synth":
        caveats.append(
            "合成分支是模型信号：符号率取自流图 `samp_rate / sps`，调制指数 "
            f"h={meta.get('mod_index', 0):g} 是按「频偏落在 270 kHz 中频滤波器内」"
            "反推的假设值，规则文档里查不到官方波形参数。绝对门限随这个假设漂移，"
            "跨参数比较才可信。")
    else:
        caveats.append(
            "真实录制已含噪声，`snr_db` 表示附加噪声相对录制总功率的比值，是真实"
            "SNR 的上界；录制里没有参考真值，`packet_yield` 无意义，只看 `packets_per_kbit`"
            " 和 `sync_rate`。")
    return caveats


# ==================== 命令行 ====================


def _parse_floats(text: str) -> tuple[float, ...]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise argparse.ArgumentTypeError("至少要给一个数值")
    return tuple(values)


def prepare_inputs(args: argparse.Namespace, out_dir: Path) -> dict:
    """按模式准备 IQ 文件，返回 (snr 值列表, 每档 IQ 文件, 元数据)。"""
    if args.mode == "file":
        if not args.iq:
            raise SystemExit("--mode file 需要 --iq 指定录制文件")
        clean = normalize_power(read_iq(args.iq, max_samples=args.max_samples))
        spec = None
        source = f"真实录制 `{Path(args.iq).name}`（窗口 {len(clean)} 采样）"
    else:
        # 一帧一遍放完就停，所以帧数按每点要收集的比特数定，避免回放接缝。
        frames = max(1, math.ceil(args.bits / FRAME_BITS))
        spec = SynthSpec(
            samp_rate=args.samp_rate,
            sps=args.sps,
            bt=args.bt_synth,
            mod_index=args.mod_index,
            seconds=args.seconds,
            seed=args.seed,
            frames=frames,
            protocol=args.protocol,
        )
        signal = synthesize(spec)
        clean = signal.samples
        source = (f"合成 GFSK 基线：{signal.frames} 个空口帧 / {signal.serial_frames} 条串口帧，"
                  f"{signal.seconds:.2f} s，单遍回放")

    snr_values = args.snr
    files: dict[float | None, Path] = {}
    work = out_dir / "iq"
    work.mkdir(parents=True, exist_ok=True)
    if not snr_values or list(snr_values) == [None]:
        path = write_iq(work / "input.c64", clean)
        files[None] = path
    else:
        for snr in snr_values:
            name = f"input_snr_{snr:g}dB.c64"
            path = write_iq(work / name, add_noise(clean, snr, seed=args.seed))
            files[snr] = path

    meta = {
        "source": source,
        "mode": args.mode,
        "samp_rate": args.samp_rate,
        "sps": args.sps,
        "symbol_rate": args.samp_rate / args.sps,
        "bits": args.bits,
        "protocol": args.protocol,
        "mod_index": args.mod_index if spec else None,
        "deviation_hz": (args.mod_index * args.samp_rate / args.sps / 2.0) if spec else None,
        "serial_frames_per_air": (signal.serial_frames / signal.frames) if spec else None,
        "frame_period_symbols": FRAME_BITS,
    }
    return {"files": files, "meta": meta}


def build_series(rows: list[SweepRow], *, axis: str = "snr_db") -> list[dict]:
    """把结果按 (loop_bw, bt) 分组画成一族曲线。"""
    grouped: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for row in rows:
        value = getattr(row.point, axis)
        if value is None:
            continue
        grouped.setdefault((row.point.loop_bw, row.point.bt), []).append(
            (float(value), row.packets_per_kbit)
        )
    series = []
    for order, ((loop_bw, bt), points) in enumerate(sorted(grouped.items())):
        series.append({
            "label": f"loop_bw={loop_bw:g} bt={bt:g}",
            "points": points,
            "color": PALETTE[order % len(PALETTE)],
        })
    return series


def write_chart(rows: list[SweepRow], path: Path) -> Path:
    series = build_series(rows)
    if series:
        chart = svg_line_chart(
            series,
            title="解波产出 vs SNR",
            x_label="SNR (dB)",
            y_label="业务包 / kbit",
        )
    else:
        # 没有 SNR 轴时（单档噪声）改画参数对比柱状图会更直观，这里退化成按点列。
        chart = svg_line_chart(
            [{"label": "各参数点", "points": [(index, row.packets_per_kbit) for index, row in enumerate(rows)]}],
            title="各参数点产出",
            x_label="参数点序号",
            y_label="业务包 / kbit",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chart, encoding="utf-8")
    return path


def publish_results(rows: list[SweepRow], out_dir: Path, meta: dict) -> dict:
    """把结果写成 CSV、曲线、报告和 summary；`--rebuild` 也走这一条路。"""
    csv_path = write_csv(rows, out_dir / "results.csv")
    chart_meta = dict(meta, csv=csv_path.name, chart="curves.svg")
    write_chart(rows, out_dir / "curves.svg")
    report_meta = dict(chart_meta, title="RMUC 空口解调参数扫描报告")
    report = build_report(rows, meta=report_meta)
    report_path = out_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    summary = {
        "points": len(rows),
        "best": (best_row(rows).as_dict() if rows else None),
        "csv": str(csv_path),
        "chart": str(out_dir / "curves.svg"),
        "report": str(report_path),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def write_meta(meta: dict, out_dir: Path) -> Path:
    """记下这次测量的口径，供 `--rebuild` 复原报告。"""
    stored = {key: value for key, value in meta.items() if key != "caveats"}
    path = out_dir / "meta.json"
    path.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("synth", "file"), default="synth")
    parser.add_argument("--protocol", required=True, choices=("2026", "2027"),
                        help="按哪个赛季的协议造包与解码；回放工具强制显式传，不接受 auto")
    parser.add_argument("--iq", type=Path, help="真实录制的 IQ 文件（complex64）")
    parser.add_argument("--out", type=Path, default=None, help="输出目录（--rebuild 时不需要）")
    parser.add_argument("--snr", type=_parse_floats, default=DEFAULT_SNR,
                        help="SNR 档位，逗号分隔；文件模式表示附加噪声")
    parser.add_argument("--loop-bw", type=_parse_floats, default=DEFAULT_LOOP_BW)
    parser.add_argument("--bt", type=_parse_floats, default=DEFAULT_BT)
    parser.add_argument("--bits", type=int, default=200_000, help="每点每通道收集的比特数")
    parser.add_argument("--timeout", type=float, default=60.0, help="单点墙钟上限，秒")
    parser.add_argument("--samp-rate", type=float, default=DEFAULT_SAMP_RATE)
    parser.add_argument("--sps", type=int, default=DEFAULT_SPS)
    parser.add_argument("--mod-index", type=float, default=DEFAULT_MOD_INDEX)
    parser.add_argument("--bt-synth", type=float, default=0.35, help="合成信号的 BT（不是被测的 bt）")
    parser.add_argument("--seconds", type=float, default=6.0, help="合成信号时长")
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--max-samples", type=int, default=4_000_000,
                        help="文件模式最多取多少采样")
    parser.add_argument("--set", action="append", default=[], metavar="VAR=VALUE")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个点（调试用）")
    parser.add_argument("--rebuild", type=Path, default=None, metavar="DIR",
                        help="不跑流图，用 DIR 里的 results.csv + meta.json 重建图表和报告")
    args = parser.parse_args(argv)

    if not args.rebuild and args.out is None:
        parser.error("需要 --out 指定输出目录（或用 --rebuild 重建已有结果）")

    if args.rebuild:
        directory = args.rebuild
        meta_path = directory / "meta.json"
        if not meta_path.is_file():
            raise SystemExit(f"{directory} 里没有 meta.json，无法重建")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows = read_csv(directory / "results.csv",
                        serial_frames_per_air=meta.get("serial_frames_per_air"))
        summary = publish_results(rows, directory, meta)
        print(f"已重建 {len(rows)} 个点的报告：{summary['report']}", flush=True)
        return 0

    started = time.monotonic()
    if args.set:
        raise SystemExit("--set 目前只支持调试单点，请用 tools/radio_replay.py")

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_inputs(args, out_dir)
    files: dict[float | None, Path] = prepared["files"]
    meta = prepared["meta"]
    serial_per_air = meta["serial_frames_per_air"]

    points = [
        SweepPoint(loop_bw=loop_bw, bt=bt, snr_db=snr)
        for snr in files
        for loop_bw in args.loop_bw
        for bt in args.bt
    ]
    if args.limit:
        points = points[: args.limit]
    print(f"共 {len(points)} 个参数点，每点 {args.bits} 比特", flush=True)

    rows: list[SweepRow] = []
    for index, point in enumerate(points, 1):
        row = evaluate_point(
            point,
            files[point.snr_db],
            target_bits=args.bits,
            timeout=args.timeout,
            serial_frames_per_air=serial_per_air,
            frame_period_symbols=meta["frame_period_symbols"],
            protocol=args.protocol,
        )
        rows.append(row)
        print(
            f"[{index}/{len(points)}] {point.label} → 同步 {row.sync} / 序列帧 "
            f"{row.frames} / 业务包 {row.decoded}，{row.packets_per_kbit:.2f} 包/kbit"
            f"（{row.seconds:.1f}s）",
            flush=True,
        )

    write_meta(meta, out_dir)
    summary = publish_results(rows, out_dir, meta)
    summary["elapsed_s"] = round(time.monotonic() - started, 1)
    print(f"\n完成：{len(rows)} 个点，用时 {summary['elapsed_s']}s", flush=True)
    print(f"CSV {summary['csv']}\n曲线 {summary['chart']}\n报告 {summary['report']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
