"""Identity, clock and fallback boundaries at the real radio integration seam."""
from dataclasses import replace
import math

import pytest

from jg_radar.config.schema import RadarConfig
from jg_radar.fuse.contract import LatestValueBuffer
from jg_radar.integration.rm_radio import RadioAdapter
from jg_radar.pipeline.build import build_stage_b
from jg_radar.types import SourceKind, SourceSample, RobotState


def test_enemy_slot_order_and_centimetres_for_both_teams():
    for team, side in (("red", "B"), ("blue", "R")):
        buf = LatestValueBuffer()
        adapter = RadioAdapter(buf, team=team)
        assert adapter.position([100, 200, 300, 400, 600, 700], [500] * 6,
                                received_at=10.0) == 6
        assert [s.name for s in buf.snapshot()] == [side + role for role in ("1", "2", "3", "4", "6", "7")]
        assert buf.get(SourceKind.RADIO, side + "6").sample.position_cm == (600, 500)
        assert buf.get(SourceKind.RADIO, side + "7").sample.position_cm == (700, 500)


def test_team_coordinates_are_only_mirrored_when_explicitly_selected():
    buf = LatestValueBuffer()
    adapter = RadioAdapter(buf, team="blue", coordinate_frame="own_team")
    adapter.position([100] * 6, [200] * 6, received_at=1)
    assert buf.get(SourceKind.RADIO, "R1").sample.position_cm == (2700, 1300)


def test_sentinel_and_out_of_field_withdraw_old_radio_but_keep_edge_points():
    buf = LatestValueBuffer()
    adapter = RadioAdapter(buf, team="red")
    adapter.position([100] * 6, [200] * 6, received_at=1)
    assert adapter.position([0, 2800, 2801, 0, 100, 200],
                            [0, 1500, 500, 100, 0, 200], received_at=2) == 3
    for role in ("1", "2", "3"):
        assert buf.get(SourceKind.RADIO, "B" + role).sample.state is RobotState.MISSING
    assert buf.get(SourceKind.RADIO, "B4").sample.position_cm == (0, 100)


def test_queued_or_out_of_order_packets_cannot_refresh_existing_data():
    buf = LatestValueBuffer()
    adapter = RadioAdapter(buf, team="red", input_latency_s=0.1)
    adapter.position([100] * 6, [200] * 6, received_at=10, transport_age_s=0.1)
    before = buf.snapshot()
    assert before[0].sample.observed_at == pytest.approx(9.8)
    assert adapter.position([500] * 6, [200] * 6, received_at=11, transport_age_s=2) == 0
    assert adapter.position([500] * 6, [200] * 6, received_at=10.1, transport_age_s=0.3) == 0
    assert buf.snapshot() == before


@pytest.mark.parametrize("age", [float("nan"), float("inf"), -1])
def test_bad_dds_clock_is_rejected(age):
    adapter = RadioAdapter(LatestValueBuffer(), team="red")
    assert adapter.position([100] * 6, [200] * 6, received_at=10, transport_age_s=age) == 0
    assert adapter.position_frames == 0


def test_short_arrays_and_non_integer_positions_are_not_mapped_to_valid_targets():
    buf = LatestValueBuffer()
    adapter = RadioAdapter(buf, team="red")
    assert adapter.position([1], [2], received_at=1) == 0
    assert len(buf) == 0
    adapter.position([100.5] * 6, [200] * 6, received_at=2)
    assert all(s.sample.state is RobotState.MISSING for s in buf.snapshot())


def test_radio_wins_then_visual_takes_over_and_idle_tick_expires_both():
    cfg = RadarConfig()
    cfg = replace(cfg, fuse=replace(cfg.fuse, dwell_s=0.0))
    stage = build_stage_b(cfg)
    adapter = RadioAdapter(stage.buffer, team="red")
    adapter.position([100] * 6, [200] * 6, received_at=10.0)
    first = stage.tick(now=10.01)
    assert first.robots["B1"].source is SourceKind.RADIO
    stage.buffer.put(SourceSample(source=SourceKind.VISION, name="B1",
                     position_cm=(110, 210), state=RobotState.MEASURED,
                     confidence=1, observed_at=10.3, revision=1, is_predicted=False), now=10.3)
    stage.tick(now=10.3)
    fallback = stage.tick(now=10.31)
    assert fallback.robots["B1"].source is SourceKind.VISION
    assert stage.tick(now=12).robots["B1"].state is RobotState.MISSING
    assert adapter.position_frames == 1


def test_state_messages_expire_independently_of_position_or_other_state_updates():
    adapter = RadioAdapter(LatestValueBuffer(), team="blue")
    adapter.status("hp", {"hp": [100] * 6}, received_at=1)
    adapter.status("state", {"remaining_coin": 5}, received_at=2)
    snap = adapter.snapshot(now=2.1)
    assert snap["messages"]["hp"]["valid"] is False
    assert snap["messages"]["state"]["valid"] is True
    assert not adapter.status("hp", {"hp": [999] * 6}, received_at=0.5)
    assert adapter.snapshot(now=2.1)["messages"]["hp"]["data"]["hp"] == [100] * 6
