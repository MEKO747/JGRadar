"""把实跑的 0x0305 落盘日志接进 P 模型：日志格式一变，这里先红。

`run_radio_vision --referee-tx --referee-log` 每帧写一行 JSON，字段是
`t` / `slots` / `stale_policy` 等。P 模型消费的就是这个格式——将来接真机时，
真实日志直接喂进来，不用改模型。
"""

from __future__ import annotations

import json

import pytest

from tools.marking_sim import run_log


def write_log(path, rows):
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                    encoding="utf-8")
    return path


def test_log_mode_counts_only_actually_sent_frames(tmp_path):
    """(0,0) 是「未发送」，不能当成一次坐标去判准确。"""
    path = write_log(tmp_path / "log.jsonl", [
        {"t": 0.0, "slots": {"B1": [0, 0]}},
        {"t": 0.213, "slots": {"B1": [0, 0]}},
        {"t": 0.426, "slots": {"B1": [0, 0]}},
    ])
    summary = run_log(path, slot="B1", speed_mps=2.0, send_hz_hint=4.7)
    assert summary["frames"] == 3
    assert summary["sent"] == 0
    assert summary["p_end"] == 0.0
    assert summary["first_100_s"] is None


def test_log_mode_accumulates_on_matching_positions(tmp_path):
    """坐标跟着假设真值走 → 连续准确，P 应该按 n(n+1)/2 涨。"""
    rows = []
    for index in range(10):
        t = index * 0.2128
        # 2 m/s 往前走，外加 42cm 常量偏移：第一帧不能是 (0,0)，
        # 因为 (0,0) 在协议里是「未发送」。42cm 仍在准确带内。
        x_cm = 42 + int(round(200.0 * t))  # 2 m/s = 200 cm/s
        rows.append({"t": t, "slots": {"B1": [x_cm, 0]}})
    summary = run_log(write_log(tmp_path / "log.jsonl", rows), slot="B1",
                      speed_mps=2.0, send_hz_hint=4.7)
    assert summary["sent"] == 10
    assert summary["counts"]["accurate"] == 10
    assert summary["p_end"] == pytest.approx(55.0)  # 10 拍 → 1+2+…+10


def test_log_mode_treats_a_gap_as_interruption(tmp_path):
    """中间断 1 秒 → 按规则补两次中断判定，P 掉下来。"""
    rows = [
        {"t": 0.0, "slots": {"B1": [0, 0]}},
        {"t": 0.2128, "slots": {"B1": [42, 0]}},
        {"t": 0.4256, "slots": {"B1": [85, 0]}},
        {"t": 1.4256, "slots": {"B1": [285, 0]}},  # 中间整段 (0,0)
    ]
    rows.insert(3, {"t": 0.6384, "slots": {"B1": [0, 0]}})
    summary = run_log(write_log(tmp_path / "log.jsonl", rows), slot="B1",
                      speed_mps=2.0, send_hz_hint=4.7)
    assert summary["counts"]["interrupt"] >= 1
    assert summary["p_end"] < 6.0


def test_log_mode_rejects_an_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        run_log(path, slot="B1", speed_mps=2.0, send_hz_hint=4.7)
