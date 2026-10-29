"""审查报告的跨模块回归：验证结果语义，不依赖相机或真实 GPU。"""
from dataclasses import replace
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from jg_radar.config import FuseConfig, PipelineConfig, RadarConfig, TrackConfig, load_config
from jg_radar.detect.armor_detector import ArmorDetector, ArmorHit
from jg_radar.detect.digit_classifier import DigitClassifier, DigitVote
from jg_radar.detect.factory import CascadeResult
from jg_radar.detect.labels import ClassMap
from jg_radar.detect.protocol import DetectionRecord
from jg_radar.detect.backends import OnnxBackend
from jg_radar.field import Keypoint, KeypointSet, validate_alignment
from jg_radar.fuse.arbiter import Arbiter, SlotDecision, position_error_bound_cm
from jg_radar.fuse.contract import LatestValueBuffer, StampedSample
from jg_radar.fuse.publish import PublishLayer
from jg_radar.fuse.sources.vision_source import VisionSource
from jg_radar.io.frame_source import Frame
from jg_radar.io.frame_source import ListFrameSource
from jg_radar.pipeline.build import build_loop
from jg_radar.pipeline.event_loop import ThreadedEventLoop
from jg_radar.pipeline.mailbox import Mailbox, StageAPacket
from jg_radar.pipeline.stage_a import StageA
from jg_radar.pipeline.stage_b import StageB
from jg_radar.track import MultiTracker
from jg_radar.track.kalman import WorldKalman
from jg_radar.track.types import TrackObservation
from jg_radar.transform import CameraModel, Intrinsics, Pose, look_at, solve_pnp_ransac
from jg_radar.types import RobotState, SourceKind, SourceSample, FrameOutput


def observation(label="B1"):
    return TrackObservation((10., 10., 70., 50.), (100., 700.), .9, label)


def stage_b(buffer=None):
    return StageB(tracker=MultiTracker(TrackConfig()), pipeline=PipelineConfig(),
                  fuse=FuseConfig(), buffer=buffer)


def radio(now, name="B2"):
    return SourceSample(SourceKind.RADIO, name, (100, 700), RobotState.MEASURED,
                        .9, now, 1, False)


class KnownLookup:
    occluded = np.zeros((1, 1), dtype=bool)

    def sample_bbox_ground(self, boxes):
        return np.tile((100., 700.), (len(boxes), 1)), np.ones(len(boxes), dtype=bool)


def test_empty_shared_buffer_keeps_identity_and_receives_later_radio():
    shared = LatestValueBuffer()
    stage = stage_b(shared)
    assert stage.buffer is shared
    shared.put(radio(10.), now=10.)
    frame = stage.process(StageAPacket(0, 10.), now=10.)
    assert frame.robots["B2"].source is SourceKind.RADIO
    assert frame.robots["B2"].state is RobotState.MEASURED


def test_build_loop_uses_configured_mailbox_and_slot_names():
    cfg = RadarConfig()
    cfg = replace(cfg, pipeline=replace(cfg.pipeline, mailbox_size=7),
                  field=replace(cfg.field, sides=("B",), roles=("3",)))
    loop = build_loop(cfg, frame_source=ListFrameSource([]), lookup=KnownLookup(), threaded=True)
    assert loop.mailbox.capacity == 7
    assert loop.stage_b.tracker.slot_names == ("B3",)
    assert tuple(loop.stage_b.tick(now=1.).robots) == ("B3",)


def test_live_source_automatically_gets_an_independent_consumer():
    class LiveSource(ListFrameSource):
        is_live = True

    loop = build_loop(RadarConfig(), frame_source=LiveSource([]), lookup=KnownLookup())
    assert isinstance(loop, ThreadedEventLoop)


def test_stale_or_repeated_packets_do_not_block_radio_ticks_or_mutate_tracker():
    stage = stage_b()
    packet = StageAPacket(1, 10., (observation(),))
    stage.process(packet, now=10.)
    hits = stage.tracker.tracks[0].hits
    assert stage.process(packet, now=10.05) is None
    assert stage.tracker.tracks[0].hits == hits
    stage.buffer.put(radio(10.5), now=10.5)
    assert stage.process(StageAPacket(2, 10.1), now=10.5) is None
    assert stage.tick(now=10.5).robots["B2"].source is SourceKind.RADIO
    assert stage.tracker.tracks[0].hits == hits


def test_source_switch_can_preserve_an_older_observation_time():
    from tools.replay_runner import check_contract
    pub = PublishLayer(slot_names=("B1",))
    visual = replace(radio(10.1, "B1"), source=SourceKind.VISION)
    first = pub.build({"B1": SlotDecision("B1", visual, "vision_measured", False)},
                      frame_seq=0, captured_at=10.1, published_at=10.1)
    second = pub.build({"B1": SlotDecision("B1", radio(10.02, "B1"), "radio_stale", True)},
                       frame_seq=1, captured_at=10.2, published_at=10.2)
    assert second.robots["B1"].observed_at < first.robots["B1"].observed_at
    assert check_contract([first, second]) == []


def test_identity_correction_updates_one_physical_track_and_retracts_old_slot():
    stage = stage_b()
    outputs = []
    for i, label in enumerate(["B1"] * 8 + ["B2"] * 30):
        now = 10. + i * .05
        out = stage.process(StageAPacket(i, now, (observation(label),)), now=now)
        active = [s for s in out.robots.values() if s.is_valid]
        assert len(active) <= 1
        outputs.append(out)
    assert len(stage.tracker.tracks) == 1
    assert stage.tracker.tracks[0].track_id == 1
    assert outputs[-1].robots["B2"].is_valid
    assert outputs[-1].robots["B1"].state is RobotState.MISSING


def test_blind_waypoint_offset_is_carried_to_fusion_and_vetoed():
    tracker = MultiTracker(TrackConfig(), waypoints=(("far", 2500., 700.),))
    for i in range(20):
        tracker.step([observation()], now=10 + i * .05)
    now = 11.85
    slots = tracker.step([], now=now)
    src = VisionSource()
    sample = src.submit(slots, now=now)[0]
    bound = position_error_bound_cm(sample, now=now, cfg=FuseConfig())
    assert sample.position_cm == (2500, 700)
    assert sample.prediction_offset_cm == pytest.approx(2400)
    assert bound >= 2400
    assert not Arbiter(FuseConfig()).decide([StampedSample(sample, now)], now=now)
    encoded = slots["B1"].to_dict()
    assert encoded["prediction_offset_cm"] == 2400
    assert type(slots["B1"]).from_dict(encoded).to_dict() == encoded


def test_near_waypoint_is_still_available_with_honest_bound():
    tracker = MultiTracker(TrackConfig(), waypoints=(("near", 105., 700.),))
    for i in range(20):
        tracker.step([observation()], now=10 + i * .05)
    now = 11.85
    sample = VisionSource().submit(tracker.step([], now=now), now=now)[0]
    assert Arbiter(FuseConfig()).decide([StampedSample(sample, now)], now=now)["B1"].sample is sample
    assert position_error_bound_cm(sample, now=now + .1, cfg=FuseConfig()) == pytest.approx(
        position_error_bound_cm(sample, now=now, cfg=FuseConfig()) + 50.)


def test_reacquisition_cannot_go_through_hold_to_bypass_consecutive_hits():
    tracker = MultiTracker(TrackConfig(), waypoints=(("near", 100., 700.),))
    for i in range(20):
        tracker.step([observation()], now=10 + i * .05)
    for now, hit in ((11.85, False), (11.9, True), (11.95, False), (12., True)):
        slot = tracker.step([observation()] if hit else [], now=now)["B1"]
        assert slot.state is RobotState.BLIND_PREDICTION
        assert slot.observed_at == pytest.approx(10.95)
    slot = tracker.step([observation()], now=12.05)["B1"]
    assert slot.state is RobotState.MEASURED
    assert slot.observed_at == pytest.approx(12.05)


def test_publish_preserves_old_measurement_time_after_missing():
    pub = PublishLayer(slot_names=("B1",))
    missing = pub.build({}, frame_seq=0, captured_at=10.9, published_at=10.9)
    decision = SlotDecision("B1", radio(10.1, "B1"), "radio_stale", False)
    frame = pub.build({"B1": decision}, frame_seq=1, captured_at=11., published_at=11.)
    assert frame.robots["B1"].observed_at == 10.1
    assert frame.published_at - frame.robots["B1"].observed_at == pytest.approx(.9)
    assert frame.publication_seq == missing.publication_seq + 1
    restored = FrameOutput.from_dict(frame.to_dict())
    assert restored.publication_seq == frame.publication_seq


def test_idle_tick_expires_visual_source_without_refreshing_liveness():
    stage = stage_b()
    for i in range(2):
        now = 10. + i * .05
        stage.process(StageAPacket(i, now, (observation(),)), now=now)
    before = stage.buffer.get(SourceKind.VISION, "B1")
    stage.tick(now=10.25)
    stage.buffer.put(radio(10.65), now=10.65)
    frame = stage.tick(now=10.7)
    assert frame.robots["B1"].state is RobotState.MISSING
    assert frame.robots["B2"].source is SourceKind.RADIO
    assert stage.buffer.get(SourceKind.VISION, "B1").received_at == before.received_at
    assert stage.processed == 2
    assert stage.vision.poll(10.7) == []


def test_vision_pending_is_bounded_and_reset_clears_history():
    stage = stage_b()
    for i in range(100):
        now = 10. + i * .05
        stage.process(StageAPacket(i, now, (observation(),)), now=now)
    assert stage.vision.poll(15.) == []
    sample_slot = stage.tracker.step([observation()], now=15.)
    src = VisionSource()
    for i in range(100):
        src.submit(sample_slot, now=15. + i * .05)
    assert len(src.poll(20.)) == 1
    stage.reset()
    assert not stage.tracker.tracks
    assert len(stage.buffer) == 0
    assert stage.tick(now=20.).robots["B1"].state is RobotState.MISSING


@pytest.mark.parametrize("elapsed", [0., 1 / 240, .05, .5, 2.])
def test_kalman_covers_the_actual_elapsed_time(elapsed):
    k = WorldKalman((100., 700.))
    k.x[2] = 300.
    before = k.covariance
    k.predict(elapsed)
    assert k.position_cm[0] == pytest.approx(100. + 300 * elapsed)
    if elapsed == 0:
        np.testing.assert_array_equal(k.covariance, before)


def test_fixed_size_onnx_roi_inference_preserves_both_roi_coordinates():
    backend = OnnxBackend(None, input_size=192, num_classes=3)
    shapes = []

    class Session:
        def run(self, _, values):
            tensor = next(iter(values.values()))
            shapes.append(tensor.shape)
            # Two anchors avoid an ambiguous 7x1 output in the YOLO layout heuristic.
            output = np.zeros((1, 7, 20), dtype=np.float32)
            output[0, :4, 0] = [96, 96, 80, 80]
            output[0, 6, 0] = .9
            return [output]

    backend._session = Session()
    backend._input_name = "images"
    detector = ArmorDetector(backend)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    hits = detector.detect(frame, [[0, 0, 40, 40], [50, 50, 90, 90]])
    assert {h.car_index for h in hits} == {0, 1}
    assert shapes == [(1, 3, 192, 192)] * 2
    assert hits[0].bbox[0] < 40 and hits[1].bbox[0] >= 50
    assert detector.inference_errors == 0


def test_dynamic_onnx_batch_has_a_real_n_dimension():
    backend = OnnxBackend(None, input_size=32, num_classes=1)

    class Session:
        def run(self, _, values):
            tensor = next(iter(values.values()))
            assert tensor.shape == (3, 3, 32, 32)
            return [np.zeros((3, 5, 20), dtype=np.float32)]

    backend._session = Session()
    backend._batch_size = None
    assert backend.infer([np.zeros((32, 32, 3), dtype=np.uint8)] * 3) == [[], [], []]


def test_ega_color_and_digit_resolve_identity_without_a_vehicle_id_label():
    result = CascadeResult([DetectionRecord(0, .9, (10, 10, 70, 50), "item")],
                           np.array([[10, 10, 70, 50]]),
                           [ArmorHit(0, DetectionRecord(2, .9, (20, 20, 35, 30), "blue"))],
                           [DigitVote(2, .95, "3")])
    a = StageA(KnownLookup())
    b = stage_b()
    for i in range(2):
        packet = a.run(Frame.with_cascade(i, 10. + i * .05, result))
        out = b.process(packet, now=packet.captured_at)
    assert packet.observations[0].car_class == "B3"
    assert out.robots["B3"].state is RobotState.MEASURED
    result.armors[0] = ArmorHit(0, replace(result.armors[0].record, label="dead"))
    assert a.run(Frame.with_cascade(2, 10.1, result)).observations[0].car_class == ""
    result.armors[0] = ArmorHit(0, replace(result.armors[0].record, label="red"))
    result.digits[0] = DigitVote(4, .9, "S")
    packet = a.run(Frame.with_cascade(3, 10.15, result))
    assert packet.observations[0].car_class == "R7"
    assert packet.observations[0].digit == "7"
    result.digits[0] = DigitVote(-1, 0., "unknown")
    packet = a.run(Frame.with_cascade(4, 10.2, result))
    assert packet.observations[0].car_class == packet.observations[0].digit == ""


def test_ega_configs_declare_color_and_digit_semantics():
    for name in ("ega_replay.yaml", "vision_check.yaml"):
        cfg = load_config("config/default.yaml", f"config/{name}")
        assert cfg.detect.car_classes == {0: "item"}
        assert cfg.detect.armor_classes == {0: "dead", 1: "red", 2: "blue"}
        assert cfg.detect.digit_classes[4] == "7"
        assert cfg.detect.digit_classes[5] == "Q"
        assert cfg.detect.digit_backend == "mobilenet"
        assert cfg.detect.digit_img_size == 64


def test_raw_classification_crops_keep_their_original_indices():
    class Backend:
        task = "classify"
        input_size = 64
        is_ready = True

        def infer(self, crops):
            assert [im.shape[:2] for im in crops] == [(10, 20), (20, 20)]
            return [[DetectionRecord(index, .9, (0, 0, 20, 10))] for index in (0, 1)]

    classifier = DigitClassifier(Backend(), class_map=ClassMap.from_spec({0: "3", 1: "7"}))
    votes = classifier.classify(np.zeros((50, 50, 3), dtype=np.uint8),
                                [[0, 0, 20, 10], [1, 1, 1, 1], [20, 20, 40, 40]])
    assert [v.label for v in votes] == ["3", "unknown", "7"]


@pytest.mark.parametrize("plane", ["ground", "raised", "tilted"])
def test_planar_pnp_handles_height_and_orientation(plane):
    fixture = json.loads((Path(__file__).parent / "fixtures/keypoints_synth.json").read_text())
    cfg = fixture["camera"]
    intr = Intrinsics.from_matrix(cfg["K"], cfg["dist"])
    camera = CameraModel(intr, cfg["width"], cfg["height"])
    R, t = look_at(fixture["pose"]["eye"], fixture["pose"]["target"])
    world = KeypointSet.from_dict(fixture).world_array()
    world[:, 2] = 0. if plane == "ground" else 50.
    if plane == "tilted":
        world[:, 2] += .03 * world[:, 0]
    pixels, valid = camera.project_world(world, R, t)
    assert valid.all()
    result = solve_pnp_ransac(world, pixels, intr)
    assert result is not None
    assert result.rms_px < .001
    np.testing.assert_allclose(result.pose.R, R, atol=1e-5)
    np.testing.assert_allclose(result.pose.t, t, atol=.01)


def test_alignment_fails_if_one_supplied_point_is_unprojectable():
    intr = Intrinsics(1000., 1000., 1000., 750.)
    camera = CameraModel(intr, 2000, 1500)
    R, t = look_at((1400., 750., 300.), (0., 750., 0.))
    world = np.array([[x, y, 0.] for x in (100., 400., 700.) for y in (650., 850.)])
    pixels, _ = camera.project_world(world, R, t)
    points = [Keypoint(str(i), tuple(w), tuple(p)) for i, (w, p) in enumerate(zip(world, pixels))]
    points.append(Keypoint("bad", (2600., 750., 0.), (1000., 200.)))
    report = validate_alignment(camera, Pose(R, t), KeypointSet(tuple(points)))
    assert not report.acceptable()
    assert np.isinf(report.max_px) and np.isinf(report.max_roundtrip_cm)


def test_live_camera_pause_still_publishes_radio_and_expires_vision():
    first_output = threading.Event()
    measured = threading.Event()
    paused = threading.Event()
    release = threading.Event()
    frames, errors = [], []

    class Source:
        is_live = True
        count = 0

        def read(self):
            if self.count == 1:
                assert first_output.wait(2.)
            if self.count < 2:
                self.count += 1
                return Frame(self.count, time.monotonic())
            paused.set()
            release.wait(3.)
            return None

        def close(self):
            release.set()

    class A:
        def run(self, frame):
            return StageAPacket(frame.seq, frame.captured_at, (observation(),))

    class Sink:
        def write(self, frame):
            frames.append(frame)
            first_output.set()
            if frame.robots["B1"].state is RobotState.MEASURED:
                measured.set()

        def close(self):
            pass

    stage = stage_b()
    mailbox = Mailbox(9)
    loop = ThreadedEventLoop(source=Source(), stage_a=A(), stage_b=stage, sink=Sink(),
                             mailbox=mailbox, poll_timeout=.01)
    assert loop.mailbox is mailbox

    def run():
        try:
            list(loop.run())
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert paused.wait(2.) and measured.wait(2.)
        now = time.monotonic()
        stage.buffer.put(radio(now), now=now)
        before = len(frames)
        release.wait(.65)
        assert len(frames) > before + 2
        assert frames[-1].robots["B1"].state is RobotState.MISSING
        assert frames[-1].robots["B2"].source is SourceKind.RADIO
        assert frames[-1].robots["B2"].state is RobotState.MEASURED
    finally:
        release.set()
        thread.join(2.)
    assert not errors and not thread.is_alive()
