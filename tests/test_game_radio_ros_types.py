"""Regression tests using the receiver's generated ROS 2 message classes."""
from types import SimpleNamespace
import time

import pytest

pytest.importorskip("rclpy")
messages = pytest.importorskip("radar_parse_em_wave.msg")
from jg_radar.fuse.contract import LatestValueBuffer
from jg_radar.integration.game_radio import RadioAdapter
from jg_radar.integration.ros_radio import RadioBridge
from jg_radar.types import RobotState, SourceKind, SourceSample


def make_bridge(team):
    bridge = RadioBridge.__new__(RadioBridge)
    bridge.adapter = RadioAdapter(LatestValueBuffer(), team=team)
    bridge._config = None
    bridge._warn = lambda exc: pytest.fail(str(exc))
    bridge._receive_config(messages.RadarParseEmWaveDemodConfig(team=team, interference_level=3))
    return bridge


def stamp(msg, *, age_s=0):
    stamp_ns = time.time_ns() - round(age_s * 1e9)
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp_ns, 10**9)
    return msg


def test_position_fields_reach_enemy_slots_in_centimetres():
    bridge = make_bridge("blue")
    msg = stamp(messages.RadarParseEmWave0A01RobotPosition(
        hero_x=100, hero_y=500, engineer_x=200, engineer_y=501,
        infantry3_x=300, infantry3_y=502, infantry4_x=400, infantry4_y=503,
        aerial_x=600, aerial_y=504, sentry_x=700, sentry_y=505))
    bridge._position(msg)
    assert [sample.sample.position_cm for sample in bridge.adapter.buffer.snapshot()] == [
        (100, 500), (200, 501), (300, 502), (400, 503), (600, 504), (700, 505)]
    assert bridge.adapter.buffer.get(SourceKind.RADIO, "R7").sample.position_cm == (700, 505)


def test_config_change_clears_positions_and_blocks_wrong_team_or_level():
    bridge = make_bridge("red")
    buffer = bridge.adapter.buffer
    now = time.monotonic()
    vision = SourceSample(source=SourceKind.VISION, name="B1", position_cm=(90, 210),
                          state=RobotState.MEASURED, confidence=1.0,
                          observed_at=now, revision=1, is_predicted=False)
    buffer.put(vision, now=now)
    position = messages.RadarParseEmWave0A01RobotPosition(hero_x=100, hero_y=200)
    bridge._position(stamp(position))
    bridge._receive_config(messages.RadarParseEmWaveDemodConfig(team="red", interference_level=3))
    assert buffer.get(SourceKind.RADIO, "B1") is not None
    for team, level in (("red", 2), ("blue", 3)):
        bridge._receive_config(messages.RadarParseEmWaveDemodConfig(team=team, interference_level=level))
        bridge._position(stamp(position))
        assert buffer.get(SourceKind.RADIO, "B1") is None
        assert buffer.get(SourceKind.VISION, "B1").sample == vision
    assert not bridge.adapter.snapshot(now=time.monotonic())["messages"]


def test_status_payload_retains_native_fields_and_hides_key():
    bridge = make_bridge("red")
    bridge._status_callback("hp")(stamp(messages.RadarParseEmWave0A02RobotHp(hero_hp=100)))
    bridge._status_callback("password")(stamp(messages.RadarParseEmWave0A06InterferenceKey(key="ABC123")))
    payload = bridge.adapter.snapshot(now=time.monotonic())["messages"]
    assert payload["hp"]["data"]["hero_hp"] == 100
    assert "header" not in payload["hp"]["data"]
    assert payload["password"]["valid"]
    assert "data" not in payload["password"]


def test_position_age_uses_header_stamp_and_rejects_stale_frames():
    bridge = make_bridge("red")
    msg = messages.RadarParseEmWave0A01RobotPosition(hero_x=100, hero_y=500)
    _, age = RadioBridge._timing(stamp(msg, age_s=0.2))
    assert age == pytest.approx(0.2, abs=0.05)
    bridge._position(stamp(msg, age_s=1))
    assert bridge.adapter.position_frames == 0
    with pytest.raises(ValueError):
        RadioBridge._timing(SimpleNamespace(header=messages.RadarParseEmWave0A01RobotPosition().header))
