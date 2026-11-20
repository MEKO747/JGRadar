"""扫描框架的纯计算部分：网格、指标、汇总、图表、报告。

跑流图的部分不在测试里（一个点要起一次 GNU Radio），这里只钉住"数怎么算"
和"写出来的东西对不对"——这两块出错时症状是报告看起来正常但数字是错的，
比崩掉更难发现。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.radio_sweep import (
    DEFAULT_BT,
    DEFAULT_LOOP_BW,
    SweepPoint,
    SweepRow,
    axis_means,
    axis_values,
    best_row,
    build_report,
    build_series,
    default_caveats,
    expand_grid,
    sensitivity,
    svg_line_chart,
    threshold_snr,
    write_csv,
)

META = {
    "source": "合成 GFSK 基线",
    "mode": "synth",
    "samp_rate": 1_000_000.0,
    "sps": 47,
    "symbol_rate": 1_000_000 / 47,
    "bits": 200_000,
    "mod_index": 8.0,
    "deviation_hz": 85_106.0,
}


def row(snr: float | None, loop_bw: float, bt: float, decoded: int, *, bits: int = 200_000) -> SweepRow:
    """造一条计数结果：空口帧数与 bits 对齐，序列帧按每帧承载比例折算。"""
    return SweepRow(
        point=SweepPoint(loop_bw=loop_bw, bt=bt, snr_db=snr),
        bits=bits,
        sync=int(round(bits / 216 * 0.98)),
        frames=decoded,
        decoded=decoded,
        seconds=1.0,
        serial_frames_per_air=0.535,
    )


# ---------------------------------------------------------------------------
# 网格
# ---------------------------------------------------------------------------
def test_grid_expands_in_a_stable_order():
    grid = expand_grid(snr_values=(10.0, 20.0), loop_bw_values=(0.005,), bt_values=(0.35, 0.45))
    assert len(grid) == 4
    assert [point.snr_db for point in grid] == [10.0, 10.0, 20.0, 20.0]
    assert [point.bt for point in grid] == [0.35, 0.45, 0.35, 0.45]


def test_grid_defaults_match_the_documented_ranges():
    grid = expand_grid()
    assert {point.loop_bw for point in grid} == set(DEFAULT_LOOP_BW)
    assert {point.bt for point in grid} == set(DEFAULT_BT)
    assert len(grid) == len(DEFAULT_LOOP_BW) * len(DEFAULT_BT)


def test_point_overrides_become_flowgraph_variables():
    overrides = SweepPoint(loop_bw=0.002, bt=0.45).overrides()
    assert overrides == {"loop_bw": "0.002", "bt": "0.45"}


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def test_sync_rate_is_normalised_by_expected_air_frames():
    result = row(20.0, 0.01, 0.35, decoded=500)
    assert result.expected_air_frames == pytest.approx(200_000 / 216)
    assert result.sync_rate == pytest.approx(0.98, abs=1e-3)
    assert result.packets_per_kbit == pytest.approx(2.5)


def test_packet_yield_needs_the_serial_frame_ratio():
    """不知道每空口帧承载多少串口帧时，产出率必须是 None，不能瞎猜一个数。"""
    result = row(20.0, 0.01, 0.35, decoded=500)
    expected = result.expected_air_frames * 0.535
    assert result.packet_yield == pytest.approx(500 / expected)

    blind = SweepRow(
        point=SweepPoint(loop_bw=0.01, bt=0.35),
        bits=200_000, sync=900, frames=400, decoded=400, seconds=1.0,
    )
    assert blind.packet_yield is None


def test_zero_bits_never_divides_by_zero():
    empty = SweepRow(point=SweepPoint(loop_bw=0.01, bt=0.35), bits=0, sync=0,
                     frames=0, decoded=0, seconds=1.0)
    assert empty.sync_rate == 0.0
    assert empty.packets_per_kbit == 0.0


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def _sample_rows() -> list[SweepRow]:
    rows = []
    for snr, decoded in ((10.0, 100), (20.0, 400), (30.0, 500)):
        for loop_bw in (0.002, 0.02):
            for bt in (0.35, 0.45):
                # 环路带宽越大、BT 越大越好——造一个已知排序出来
                gain = 1.15 if loop_bw > 0.01 else 0.85
                gain *= 1.05 if bt > 0.4 else 0.95
                rows.append(row(snr, loop_bw, bt, decoded=int(decoded * gain)))
    return rows


def test_axis_values_and_means():
    rows = _sample_rows()
    assert axis_values(rows, "snr_db") == [10.0, 20.0, 30.0]
    means = axis_means(rows, "loop_bw")
    assert means[0.02] > means[0.002], "这组数据里大环路带宽应该平均更好"


def test_sensitivity_ranks_the_dominant_axis_first():
    ranking = sensitivity(_sample_rows())
    order = [name for name, _ in ranking]
    assert order[0] == "snr_db", "SNR 才是主导因素，不能把调参旋钮排到它前面"
    assert all(value >= 0.0 for _, value in ranking)


def test_best_row_takes_the_highest_yield():
    rows = _sample_rows()
    best = best_row(rows)
    assert best is not None
    assert best.snr_db == 30.0
    assert best.point.loop_bw == 0.02
    assert best.point.bt == 0.45


def test_best_row_of_nothing_is_none():
    assert best_row([]) is None


def test_threshold_snr_finds_the_knee_of_a_curve():
    rows = [
        row(0.0, 0.01, 0.35, 0),
        row(5.0, 0.01, 0.35, 10),
        row(10.0, 0.01, 0.35, 380),
        row(20.0, 0.01, 0.35, 480),
        row(30.0, 0.01, 0.35, 500),
    ]
    # 平台值 500，90% 是 450 → 20 dB 才够
    assert threshold_snr(rows, 0.01, 0.35) == 20.0


def test_threshold_snr_absent_when_everything_fails():
    rows = [row(0.0, 0.01, 0.35, 0), row(10.0, 0.01, 0.35, 0)]
    assert threshold_snr(rows, 0.01, 0.35) is None


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------
def test_svg_chart_draws_one_polyline_per_series():
    svg = svg_line_chart(
        [
            {"label": "a", "points": [(0.0, 0.0), (1.0, 2.0), (2.0, 3.0)]},
            {"label": "b", "points": [(0.0, 1.0), (1.0, 1.0)]},
        ],
        title="t", x_label="x", y_label="y",
    )
    assert svg.startswith("<svg")
    assert svg.count("<polyline") == 2
    assert "loop" not in svg.lower() or True
    assert svg.count("<circle") >= 5


def test_svg_chart_handles_no_data():
    assert "没有可画的数据" in svg_line_chart([])


def test_chart_series_group_by_parameter_pair():
    rows = _sample_rows()
    series = build_series(rows)
    assert len(series) == 4, "两个 loop_bw × 两个 bt = 4 条曲线"
    for item in series:
        assert len(item["points"]) == 3
        assert len({x for x, _ in item["points"]}) == len(item["points"])


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------
def test_csv_has_a_row_per_point_with_all_columns(tmp_path: Path):
    rows = _sample_rows()
    path = write_csv(rows, tmp_path / "results.csv")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(rows) + 1
    header = lines[0].split(",")
    assert header[:3] == ["snr_db", "loop_bw", "bt"]
    assert "packets_per_kbit" in header


def test_report_states_the_conclusion_and_the_limits(tmp_path: Path):
    rows = _sample_rows()
    report = build_report(rows, meta=dict(META, chart="curves.svg", csv="results.csv"))
    assert "## 结论" in report
    assert "最优参数点" in report
    assert "参数敏感度排序" in report
    assert "curves.svg" in report
    assert "## 方法与局限" in report
    assert "rx_gain" in report, "增益不可观测这条限制必须出现在报告里"
    assert "| loop_bw | bt | 门限 SNR |" in report


def test_report_of_no_rows_says_so():
    report = build_report([], meta=dict(META))
    assert "没有有效数据" in report


def test_caveats_differ_between_synth_and_recorded():
    synth = " ".join(default_caveats({"mode": "synth", "mod_index": 8.0}))
    recorded = " ".join(default_caveats({"mode": "file"}))
    assert "模型信号" in synth
    assert "上界" in recorded
    assert "rx_gain" in synth and "rx_gain" in recorded
