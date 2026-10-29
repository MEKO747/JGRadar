"""输出契约的回归门。

这是全项目最重要的一组测试：契约破了，后面所有模块的输出都不可信。

三条不变式（见 jg_radar/types.py 模块文档）：
  I1  is_predicted ⇔ state != MEASURED
  I2  position_cm 非空时必在场地范围内
  I3  observed_at 单调不回退
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from golden import ALL_SLOTS, build_golden_frames
from jg_radar.io.sink import JsonlSink, NullSink, read_jsonl
from jg_radar.types import (
    FIELD_MAX_X_CM,
    FIELD_MAX_Y_CM,
    SCHEMA_VERSION,
    FrameOutput,
    RobotSlotOutput,
    RobotState,
    SourceKind,
    default_slot_names,
    is_in_field,
    missing_slot,
)

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "frame_golden.json"


# --------------------------------------------------------------------------
# I1：预测永不伪装成测量
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "state,is_predicted,ok",
    [
        (RobotState.MEASURED, False, True),
        (RobotState.OCCLUSION_HOLD, True, True),
        (RobotState.BLIND_PREDICTION, True, True),
        (RobotState.MISSING, False, True),           # 没有坐标，谈不上预测
        (RobotState.MEASURED, True, False),          # 测量被标成预测 → 拒绝
        (RobotState.OCCLUSION_HOLD, False, False),   # 预测伪装成测量 → 拒绝
        (RobotState.BLIND_PREDICTION, False, False),
        (RobotState.MISSING, True, False),           # 没坐标却声称在预测 → 拒绝
    ],
)
def test_invariant_i1_is_predicted(state, is_predicted, ok):
    slot = RobotSlotOutput(
        name="R1",
        position_cm=None if state is RobotState.MISSING else (1000, 500),
        state=state,
        confidence=0.5,
        observed_at=1.0,
        source=SourceKind.VISION,
        is_predicted=is_predicted,
    )
    if ok:
        slot.validate()
    else:
        with pytest.raises(ValueError, match="不变式 I1"):
            slot.validate()


def test_measured_is_the_only_measurement():
    assert RobotState.MEASURED.is_measurement
    for state in (RobotState.OCCLUSION_HOLD, RobotState.BLIND_PREDICTION, RobotState.MISSING):
        assert not state.is_measurement


def test_prediction_states_are_exactly_the_two_inferred_ones():
    assert [s for s in RobotState if s.is_prediction] == [
        RobotState.OCCLUSION_HOLD,
        RobotState.BLIND_PREDICTION,
    ]
    assert not RobotState.MISSING.is_prediction
    assert not RobotState.MEASURED.is_prediction


# --------------------------------------------------------------------------
# I2：场地边界
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "pos,ok",
    [
        ((0, 0), True),
        ((FIELD_MAX_X_CM, FIELD_MAX_Y_CM), True),
        ((1400, 750), True),
        ((-1, 500), False),
        ((500, -1), False),
        ((FIELD_MAX_X_CM + 1, 500), False),
        ((500, FIELD_MAX_Y_CM + 1), False),
    ],
)
def test_invariant_i2_field_bounds(pos, ok):
    slot = RobotSlotOutput(
        name="B3",
        position_cm=pos,
        state=RobotState.MEASURED,
        confidence=0.8,
        observed_at=2.0,
        source=SourceKind.VISION,
        is_predicted=False,
    )
    if ok:
        slot.validate()
        assert slot.is_valid
    else:
        with pytest.raises(ValueError, match="不变式 I2"):
            slot.validate()


def test_is_in_field_helper():
    assert is_in_field(0, 0)
    assert is_in_field(FIELD_MAX_X_CM, FIELD_MAX_Y_CM)
    assert not is_in_field(-0.1, 0)
    assert not is_in_field(0, FIELD_MAX_Y_CM + 0.1)


# --------------------------------------------------------------------------
# 状态与坐标的一致性
# --------------------------------------------------------------------------
def test_missing_must_not_carry_position():
    slot = missing_slot("R7", 5.0)
    assert slot.position_cm is None
    assert slot.state is RobotState.MISSING
    assert not slot.is_valid
    slot.validate()

    bad = RobotSlotOutput(
        name="R7",
        position_cm=(100, 100),
        state=RobotState.MISSING,
        confidence=0.0,
        observed_at=5.0,
        source=SourceKind.VISION,
        is_predicted=False,
    )
    with pytest.raises(ValueError, match="MISSING"):
        bad.validate()


@pytest.mark.parametrize(
    "state", [RobotState.MEASURED, RobotState.OCCLUSION_HOLD, RobotState.BLIND_PREDICTION]
)
def test_non_missing_must_carry_position(state):
    bad = RobotSlotOutput(
        name="R2",
        position_cm=None,
        state=state,
        confidence=0.5,
        observed_at=5.0,
        source=SourceKind.VISION,
        is_predicted=not state.is_measurement,
    )
    with pytest.raises(ValueError):
        bad.validate()


def test_missing_slot_helper_defaults():
    slot = missing_slot("B6", 12.5)
    assert slot.name == "B6"
    assert slot.observed_at == 12.5
    assert slot.source is SourceKind.VISION
    assert not slot.is_predicted
    assert slot.schema_version == SCHEMA_VERSION


# --------------------------------------------------------------------------
# I3：observed_at 单调不回退
# --------------------------------------------------------------------------
def test_invariant_i3_observed_at_monotonic():
    last: dict[str, float] = {}
    for frame in build_golden_frames():
        for name, slot in frame.robots.items():
            if name in last:
                assert slot.observed_at >= last[name], (
                    f"{name}: observed_at 从 {last[name]} 回退到 {slot.observed_at} "
                    f"（frame {frame.frame_seq}）"
                )
            last[name] = slot.observed_at


def test_frame_capture_time_monotonic():
    frames = build_golden_frames()
    caps = [f.captured_at for f in frames]
    assert caps == sorted(caps)
    assert [f.frame_seq for f in frames] == list(range(len(frames)))


# --------------------------------------------------------------------------
# 槽位命名
# --------------------------------------------------------------------------
def test_default_slot_names_order():
    names = default_slot_names()
    assert names == ("R1", "R2", "R3", "R4", "R6", "R7", "B1", "B2", "B3", "B4", "B6", "B7")
    assert len(set(names)) == 12
    assert ALL_SLOTS == names


# --------------------------------------------------------------------------
# 序列化金标准
# --------------------------------------------------------------------------
def _golden_payload() -> list[dict]:
    return [f.to_dict() for f in build_golden_frames()]


def test_golden_json_roundtrip():
    if os.environ.get("JGRADAR_UPDATE_GOLDEN") == "1" or not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(
            json.dumps(_golden_payload(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    actual = _golden_payload()
    assert actual == expected, (
        "输出契约发生变更。若这是有意的，用 JGRADAR_UPDATE_GOLDEN=1 pytest "
        "tests/test_types_contract.py 更新金标准，并在提交信息里说明原因。"
    )

    # 反序列化必须能原样还原
    for frame, d in zip(build_golden_frames(), expected):
        assert FrameOutput.from_dict(d) == frame


def test_serialization_is_deterministic():
    a = json.dumps(_golden_payload(), ensure_ascii=False, separators=(",", ":"))
    b = json.dumps(_golden_payload(), ensure_ascii=False, separators=(",", ":"))
    assert a == b


def test_slot_to_dict_schema_fields():
    d = build_golden_frames()[0].robots["R1"].to_dict()
    assert set(d) == {
        "name",
        "position_cm",
        "state",
        "confidence",
        "observed_at",
        "source",
        "is_predicted",
        "track_id",
        "position_std_cm",
        "schema_version",
    }
    assert d["schema_version"] == SCHEMA_VERSION == "jg.radar.robot.v1"
    assert d["position_cm"] == [1200, 700]
    assert d["state"] == "measured"
    assert d["is_predicted"] is False


def test_frame_to_dict_schema_fields():
    d = build_golden_frames()[0].to_dict()
    assert set(d) == {"frame_seq", "captured_at", "published_at", "robots", "diagnostics"}
    assert len(d["robots"]) == 12


# --------------------------------------------------------------------------
# JSONL 输出
# --------------------------------------------------------------------------
def test_jsonl_sink_roundtrip(tmp_path):
    out = tmp_path / "frames.jsonl"
    frames = build_golden_frames()
    with JsonlSink(out) as sink:
        for f in frames:
            sink.write(f)
    assert sink.count == len(frames)

    back = read_jsonl(out)
    assert back == frames
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == len(frames)


def test_jsonl_sink_is_byte_deterministic(tmp_path):
    frames = build_golden_frames()
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for path in (a, b):
        with JsonlSink(path) as sink:
            for f in frames:
                sink.write(f)
    assert a.read_bytes() == b.read_bytes()


def test_null_sink_counts():
    sink = NullSink()
    for f in build_golden_frames():
        sink.write(f)
    sink.close()
    assert sink.count == 4
