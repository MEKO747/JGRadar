#!/usr/bin/env python3
"""从完整检测记录和原采集时钟重放跟踪、融合和发布；不重新推理或投影。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jg_radar.config import load_config
from jg_radar.io.frame_source import cascade_from_dict
from jg_radar.pipeline.mailbox import StageAPacket
from jg_radar.pipeline.stage_a import StageA
from jg_radar.pipeline.stage_b import StageB
from jg_radar.track import MultiTracker
from jg_radar.track.types import TrackObservation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tracker = MultiTracker(cfg.track)
    stage = StageB(tracker=tracker, pipeline=cfg.pipeline, fuse=cfg.fuse)
    args.out.mkdir(parents=True, exist_ok=True)
    counts = {'old': Counter(), 'new': Counter()}
    jumps = {'old': 0, 'new': 0}
    changes = {'old': 0, 'new': 0}
    last = {'old': {}, 'new': {}}
    n = 0
    with args.input.open() as src, (args.out / 'frames.jsonl').open('w') as dst:
        for line in src:
            record = json.loads(line)
            timestamp = float(record['captured_at'])
            cascade = cascade_from_dict(record['cascade'])
            observations = []
            for saved in record['observations']:
                index = saved.get('car_index')
                if index is None or not (0 <= index < len(cascade.cars)):
                    raise ValueError('检测记录缺少可对应的车辆下标')
                observations.append(TrackObservation(
                    bbox=tuple(saved['bbox']),
                    world_cm=tuple(saved['world_cm']) if saved['world_cm'] is not None else None,
                    confidence=saved['confidence'], car_class=saved['car_class'],
                    digit=saved['digit'], car_index=index,
                    armor_evidence=StageA._armor_evidence(cascade, index, timestamp),
                    direct_identity=cascade.cars[index].label in tracker.slot_names))
            packet = StageAPacket(record['frame_index'], timestamp, tuple(observations))
            result = stage.process(packet, now=timestamp)
            if result is None:
                raise ValueError(f'第 {packet.frame_seq} 帧时间戳无效')
            old = record['output']['robots']
            new = {name: slot.to_dict() for name, slot in result.robots.items()}
            active_boxes = {str(t.track_id): list(t.predicted_bbox) for t in tracker.tracks
                            if t.predicted_bbox is not None}
            row = {'frame': packet.frame_seq, 'video_time_s': record['video_time_s'],
                   'captured_at': timestamp, 'old': old, 'new': new, 'boxes': active_boxes,
                   'identity_conflicts': tracker.last_diagnostics['identity_conflicts']}
            dst.write(json.dumps(row, separators=(',', ':')) + '\n')
            for key, slots in (('old', old), ('new', new)):
                for name, slot in slots.items():
                    counts[key][slot['state']] += 1
                    prior = last[key].get(name)
                    if prior and slot['state'] == prior['state'] == 'measured':
                        dt = timestamp - prior['_time']
                        if 0 < dt <= .1 and slot['position_cm'] and prior['position_cm']:
                            if math.dist(slot['position_cm'], prior['position_cm']) >= 200:
                                jumps[key] += 1
                            if slot['track_id'] != prior['track_id']:
                                changes[key] += 1
                    last[key][name] = dict(slot, _time=timestamp)
            n += 1
            if n % 1000 == 0:
                print(f'已重放 {n} 帧', flush=True)
    summary = {'frames': n, 'states': {k: dict(v) for k, v in counts.items()},
               'measured_adjacent_jumps_ge_2m': jumps,
               'measured_adjacent_track_changes': changes,
               'note': '疑似跳变是逐帧统计，不等于经人工确认的错误；复用检测和投影。'}
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
