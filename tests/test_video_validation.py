"""验证报告不能把预测、缺失或轨迹换号算成识别准确率。"""
from types import SimpleNamespace
import pytest

from jg_radar.io.frame_source import Frame
from jg_radar.types import FrameOutput, RobotSlotOutput, RobotState, SourceKind, missing_slot
from tools.validate_video import Stats, analyzed_frames


def _output(state, track_id=None):
    if state is RobotState.MISSING:
        slot = missing_slot('B3', 1.0)
    else:
        slot = RobotSlotOutput('B3', (100, 200), state, .9, 1.0,
                               SourceKind.VISION, state.is_prediction, track_id=track_id)
    return FrameOutput(0, 1.0, 1.0, {'B3': slot})


def test_report_separates_measured_predictions_missing_and_track_handoff():
    stats = Stats()
    result = SimpleNamespace(cars=[None], armors=[None])
    observations = [SimpleNamespace(car_class='B3', world_cm=(100, 200))]
    states = [(RobotState.MISSING, None), (RobotState.MEASURED, 1),
              (RobotState.OCCLUSION_HOLD, 1), (RobotState.MISSING, None),
              (RobotState.MEASURED, 2)]
    for i, (state, track_id) in enumerate(states):
        packet = SimpleNamespace(captured_at=100 + i * .04, observations=observations)
        tracks = [SimpleNamespace(track_id=track_id)] if track_id else []
        stats.add(result, packet, _output(state, track_id), tracks, 10, 12)
    summary = stats.summary()
    assert summary['frames'] == 5
    assert summary['frames_with_published_position'] == 3
    assert summary['per_slot_state_frames']['B3'] == {'missing': 2, 'measured': 2, 'occlusion_hold': 1}
    assert summary['slot_track_handoffs'] == {'B3': 1}
    assert summary['unique_track_ids'] == 2
    assert summary['nonmonotonic_stamps'] == 0


def test_report_exposes_failed_outside_projection_and_bad_timestamps():
    stats = Stats()
    observations = [SimpleNamespace(car_class='', world_cm=None),
                    SimpleNamespace(car_class='B3', world_cm=(3000, 200))]
    result = SimpleNamespace(cars=[None, None], armors=[])
    packet = SimpleNamespace(captured_at=100.0, observations=observations)
    for _ in range(2):
        stats.add(result, packet, _output(RobotState.MISSING), [], 2, 3)
    summary = stats.summary()
    assert summary['projected_observations'] == 2
    assert summary['outside_field_observations'] == 2
    assert summary['identity_observations'] == 2
    assert summary['frames_with_published_position'] == 0
    assert summary['nonmonotonic_stamps'] == 1


def _analysis_fixture(count, *, fail_at=None):
    class Source:
        index = 0

        def read(self):
            i = self.index
            if i >= count:
                return None
            self.index += 1
            return Frame(i, 100 + i * .04)

    class StageB:
        def __init__(self):
            self.tracker = SimpleNamespace(tracks=[SimpleNamespace(label='')])

        def process(self, packet, *, now):
            if packet.frame_seq == fail_at:
                raise ValueError('intentional inference/pipeline failure')
            self.tracker.tracks[0].label = str(packet.frame_seq)
            return FrameOutput(packet.frame_seq, packet.captured_at, now, {})

    cascade = SimpleNamespace(run=lambda image: None, car=SimpleNamespace(),
                              armor=SimpleNamespace(), digit=SimpleNamespace())
    stage_a = SimpleNamespace(cascade=cascade, run=lambda frame: SimpleNamespace(
        frame_seq=frame.seq, captured_at=frame.captured_at))
    return Source(), SimpleNamespace(stage_a=stage_a, stage_b=StageB())


def test_parallel_visualization_preserves_every_frame_timestamp_and_track_snapshot():
    source, loop = _analysis_fixture(12)
    rows = list(analyzed_frames(source, loop, 12))
    assert [r[0].seq for r in rows] == list(range(12))
    assert [r[2].captured_at for r in rows] == [100 + i * .04 for i in range(12)]
    assert [r[4][0].label for r in rows] == [str(i) for i in range(12)]


def test_parallel_visualization_propagates_failure_instead_of_successful_eof():
    source, loop = _analysis_fixture(12, fail_at=4)
    stream = analyzed_frames(source, loop, 12)
    assert [next(stream)[0].seq for _ in range(4)] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match='intentional'):
        next(stream)
