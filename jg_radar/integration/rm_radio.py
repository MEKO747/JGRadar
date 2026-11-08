"""Convert the RM_Radio 2026 ROS messages into JGRadar samples.

No ROS imports here. The six position slots are enemy roles 1,2,3,4,6,7;
raw coordinates and the default output use the red field frame in cm.
The headerless messages cannot provide the RF acquisition timestamp. We use
DDS publisher time (minus optional measured decode latency), never tick time.
"""
from __future__ import annotations

import math
import threading
from numbers import Integral

from jg_radar.fuse.contract import LatestValueBuffer
from jg_radar.types import RobotState, SourceKind, SourceSample

ROLES = ("1", "2", "3", "4", "6", "7")


class RadioAdapter:
    def __init__(self, buffer: LatestValueBuffer, *, team: str,
                 coordinate_frame: str = "red_field", input_latency_s: float = 0.0,
                 max_message_age_s: float = 0.5, status_ttl_s: float = 1.0):
        if team not in ("red", "blue"):
            raise ValueError("team must be red or blue (own team)")
        if coordinate_frame not in ("red_field", "own_team"):
            raise ValueError("coordinate_frame must be red_field or own_team")
        for name, value in (("input_latency_s", input_latency_s),
                            ("max_message_age_s", max_message_age_s),
                            ("status_ttl_s", status_ttl_s)):
            if not math.isfinite(value) or value < 0 or (name != "input_latency_s" and value == 0):
                raise ValueError(f"invalid {name}")
        self.buffer = buffer
        self.team = team
        self.enemy = "B" if team == "red" else "R"
        self.coordinate_frame = coordinate_frame
        self.input_latency_s = float(input_latency_s)
        self.max_message_age_s = float(max_message_age_s)
        self.status_ttl_s = float(status_ttl_s)
        self._lock = threading.Lock()
        self._revision = 0
        self._last_position_time = float("-inf")
        self._latest: dict[str, tuple[float, dict]] = {}
        self.position_frames = 0
        self.rejected_frames = 0
        self.invalid_positions = 0

    def message_time(self, *, received_at: float, transport_age_s: float = 0.0) -> float:
        """DDS send time expressed on this process's monotonic clock."""
        if (not math.isfinite(received_at) or not math.isfinite(transport_age_s)
                or transport_age_s < -0.05 or transport_age_s > self.max_message_age_s):
            raise ValueError("old message or unsynchronised DDS sender clock")
        return received_at - max(0.0, transport_age_s) - self.input_latency_s

    def position(self, x, y, *, received_at: float, transport_age_s: float = 0.0) -> int:
        try:
            if len(x) != 6 or len(y) != 6:
                raise ValueError("Position must contain exactly six x/y slots")
            observed_at = self.message_time(received_at=received_at, transport_age_s=transport_age_s)
        except (ValueError, TypeError):
            with self._lock:
                self.rejected_frames += 1
            return 0
        with self._lock:
            if observed_at <= self._last_position_time:
                self.rejected_frames += 1
                return 0
            samples = []
            valid = 0
            for role, px, py in zip(ROLES, x, y):
                good = (isinstance(px, Integral) and not isinstance(px, bool)
                        and isinstance(py, Integral) and not isinstance(py, bool)
                        and 0 <= px <= 2800 and 0 <= py <= 1500
                        and (px, py) not in ((0, 0), (2800, 1500)))
                pos = (int(px), int(py)) if good else None
                if pos is not None and self.coordinate_frame == "own_team" and self.team == "blue":
                    pos = (2800 - pos[0], 1500 - pos[1])
                self._revision += 1
                samples.append(SourceSample(
                    source=SourceKind.RADIO, name=self.enemy + role,
                    position_cm=pos,
                    state=RobotState.MEASURED if good else RobotState.MISSING,
                    confidence=1.0 if good else 0.0, observed_at=observed_at,
                    revision=self._revision, is_predicted=False))
                valid += int(good)
            # Invalid/sentinel slots explicitly withdraw the old radio reading.
            self.buffer.put_many(samples, now=received_at)
            self._last_position_time = observed_at
            self.position_frames += 1
            self.invalid_positions += 6 - valid
            self._latest["position"] = (observed_at, {"valid_slots": valid})
            return valid

    def status(self, kind: str, data: dict, *, received_at: float,
               transport_age_s: float = 0.0) -> bool:
        try:
            stamp = self.message_time(received_at=received_at, transport_age_s=transport_age_s)
        except ValueError:
            return False
        with self._lock:
            if kind in self._latest and stamp <= self._latest[kind][0]:
                return False
            self._latest[kind] = (stamp, data)
            return True

    def snapshot(self, *, now: float) -> dict:
        with self._lock:
            return {
                "protocol": "tdt_2026", "own_team": self.team,
                "position_frames": self.position_frames,
                "rejected_position_frames": self.rejected_frames,
                "invalid_position_slots": self.invalid_positions,
                "timestamp_basis": "dds_publish_time_minus_configured_latency",
                "input_latency_s": self.input_latency_s,
                "messages": {kind: {"observed_at": stamp, "age_s": max(0.0, now - stamp),
                                    "valid": 0 <= now - stamp <= self.status_ttl_s,
                                    "data": data}
                             for kind, (stamp, data) in self._latest.items()},
            }
