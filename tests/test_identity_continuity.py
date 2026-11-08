from dataclasses import replace

import pytest

from jg_radar.config import TrackConfig, FuseConfig, PipelineConfig
from jg_radar.pipeline.mailbox import StageAPacket
from jg_radar.pipeline.stage_b import StageB
from jg_radar.track import MultiTracker
from jg_radar.track.identity import armor_votes
from jg_radar.track.types import ArmorEvidence, TrackObservation
from jg_radar.types import RobotState


def plate(digit="7", size=40, confidence=.9, time=0.):
    return ArmorEvidence("blue", digit, confidence, confidence, size, size, time)


def obs(x=100, world=500, *, evidence=None, label="B7", size=40):
    return TrackObservation((x-size/2, 100-size/2, x+size/2, 100+size/2),
                            (world, 700), .9, label, "7", None,
                            tuple(evidence or ()))


def test_box_deformation_keeps_track_and_two_cars_do_not_merge():
    tracker = MultiTracker(TrackConfig())
    for i in range(4):
        tracker.step([obs(), obs(300, 1500, label="B3")], now=i*.05)
    tracker.step([obs(size=100), obs(300, 1500, label="B3")], now=.2)
    assert len(tracker.tracks) == 2
    assert tracker.tracks[0].hits == 5
    assert tracker.tracks[1].hits == 5


def test_box_expands_upward_while_vehicle_bottom_stays_put():
    tracker = MultiTracker(TrackConfig())
    original = TrackObservation((1859, 2422, 2173, 2866), (355, 671), .9, 'R3', '3')
    expanded = replace(original, bbox=(1846, 2268, 2174, 2869))
    for i in range(20):
        tracker.step([original], now=i*.05)
    tracker.step([expanded], now=1.)
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].hits == 21


def test_two_cars_cross_without_being_merged():
    tracker = MultiTracker(TrackConfig())
    for i in range(16):
        left = obs(100 + 5*i, 500 + 10*i)
        right = obs(200 - 5*i, 1500 - 10*i, label='R3')
        tracker.step([left, right], now=i*.1)
    assert len(tracker.tracks) == 2
    assert {track.label for track in tracker.tracks} == {'B7', 'R3'}
    assert all(track.hits == 16 for track in tracker.tracks)


def test_armor_evidence_is_grouped_and_preserves_conflict():
    agree = armor_votes((plate(), plate()))
    conflict = armor_votes((plate(), plate("Q")))
    assert len(agree) == 1
    assert {v.label for v in conflict} == {"B7", "BQ"}
    assert next(v.weight for v in conflict if v.label == "B7") < agree[0].weight
    assert armor_votes((plate(size=10),))[0].weight < agree[0].weight


def test_identity_expires_despite_matching_unidentified_vehicle():
    tracker = MultiTracker(TrackConfig(identity_evidence_ttl_s=.2))
    tracker.step([obs()], now=0.)
    assert "B7" in tracker.step([obs()], now=.05)
    for i in range(2, 8):
        slots = tracker.step([obs(label="")], now=i*.05)
    assert "B7" not in slots or slots["B7"].state is RobotState.MISSING
    assert len(tracker.tracks) == 1


def test_far_challenge_holds_then_can_correct_wrong_owner():
    tracker = MultiTracker(TrackConfig(identity_challenge_s=.2))
    for i in range(4):
        slots = tracker.step([obs()], now=i*.05)
    old = slots["B7"]
    assert old.state is RobotState.MEASURED
    for i in range(4, 8):
        slots = tracker.step([obs(400, 2000)], now=i*.05)
    held = slots["B7"]
    assert held.state is RobotState.OCCLUSION_HOLD
    assert held.observed_at == old.observed_at
    for i in range(8, 14):
        slots = tracker.step([obs(400, 2000)], now=i*.05)
    assert slots["B7"].state is RobotState.MEASURED
    assert slots["B7"].position_cm[0] > 1900


def test_far_conflict_expires_hold_and_retracts_in_stage_b():
    cfg = TrackConfig(lost_predict_s=.15, identity_challenge_s=1.)
    stage = StageB(tracker=MultiTracker(cfg), pipeline=PipelineConfig(), fuse=FuseConfig())
    for i in range(4):
        now = 10+i*.05
        out = stage.process(StageAPacket(i, now, (obs(),)), now=now)
    for i in range(4, 12):
        now = 10+i*.05
        out = stage.process(StageAPacket(i, now, (obs(400, 2000),)), now=now)
    assert out.robots["B7"].state is RobotState.MISSING


def test_far_claim_cannot_reacquire_using_only_old_identity():
    tracker = MultiTracker(TrackConfig(lost_predict_s=.15, identity_challenge_s=.1,
                                       identity_evidence_ttl_s=2.))
    for i in range(4):
        tracker.step([obs()], now=i*.05)
    tracker.step([obs(400, 2000)], now=.2)
    tracker.step([obs(400, 2000)], now=.25)
    for i in range(6, 14):
        slots = tracker.step([obs(400, 2000, label="")], now=i*.05)
    assert "B7" not in slots or slots["B7"].state is RobotState.MISSING
