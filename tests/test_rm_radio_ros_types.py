"""Optional real Jazzy type regressions; no ROS context, sockets or devices."""
import time
from types import SimpleNamespace

import pytest

pytest.importorskip('rclpy')
messages = pytest.importorskip('radio_interface.msg')
from jg_radar.fuse.contract import LatestValueBuffer
from jg_radar.integration.rm_radio import RadioAdapter
from jg_radar.integration.ros_radio import RadioBridge
from jg_radar.types import SourceKind


def test_jazzy_metadata_dict_and_uint16_arrays_are_json_serialisable():
    import json
    adapter = RadioAdapter(LatestValueBuffer(), team='red')
    owner = SimpleNamespace(adapter=adapter, _timing=RadioBridge._timing,
                            _warn=lambda exc: pytest.fail(str(exc)))
    msg = messages.Hp()
    msg.hp = [100,200,300,400,500,600]
    RadioBridge._status_callback(owner, 'hp')(msg, {'source_timestamp':time.time_ns()})
    snap = adapter.snapshot(now=time.monotonic())
    assert snap['messages']['hp']['data']['hp'] == [100,200,300,400,500,600]
    json.dumps(snap, allow_nan=False)


def test_real_position_type_reaches_buffer_with_jazzy_metadata():
    buf=LatestValueBuffer()
    adapter=RadioAdapter(buf,team='blue')
    owner=SimpleNamespace(adapter=adapter,_timing=RadioBridge._timing,
                          _warn=lambda exc: pytest.fail(str(exc)))
    msg=messages.Position();msg.x=[100]*6;msg.y=[500]*6
    RadioBridge._position(owner,msg,{'source_timestamp':time.time_ns()})
    assert buf.get(SourceKind.RADIO,'R7').sample.position_cm == (100,500)


def test_metadata_object_compatibility_and_missing_timestamp():
    stamp,age=RadioBridge._timing(SimpleNamespace(source_timestamp=time.time_ns()))
    assert stamp>0 and abs(age)<0.1
    with pytest.raises(ValueError):
        RadioBridge._timing({'source_timestamp':0})
