#!/usr/bin/env python3
"""逐帧真实模型回放；可视化的是正式 Stage B 输出，不把候选框当成身份。

每个输入帧恰好推理一次，保留录制时间戳供跟踪使用。处理速度不作为
实时相机帧率；覆盖率也不作为准确率。输出完整视频、逐帧 JSONL 和摘要。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path
from queue import Queue, Empty, Full
import subprocess
import sys
from threading import Event, Thread
import time
import traceback

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jg_radar.config import load_config
from jg_radar.io.frame_source import Frame, cascade_to_dict
from jg_radar.io.video_source import VideoFrameSource
from jg_radar.pipeline.build import build_loop
from jg_radar.types import RobotState, FIELD_MAX_X_CM, FIELD_MAX_Y_CM
from tools.render_overlay import _scale_camera, CAM_W, CAM_H, HEADER_H

WIDTH, HEIGHT = 1920, 1080
PX, PW, FH = 1360, 552, 296
WHITE, DIM = (235, 235, 235), (155, 163, 175)
GREEN, YELLOW = (90, 225, 100), (70, 220, 255)
WINDOW = 'JGRadar - FULL VIDEO VALIDATION (Space pause / Esc stop)'


def atom_json(path, data):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def side_color(name):
    return (95, 105, 255) if name.startswith('R') else (255, 185, 65)


def label(canvas, text, origin, color=WHITE, scale=.48):
    x, y = origin
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def dashed_box(canvas, a, b, color):
    x1, y1 = a
    x2, y2 = b
    for x in range(x1, x2, 16):
        cv2.line(canvas, (x, y1), (min(x + 8, x2), y1), color, 1)
        cv2.line(canvas, (x, y2), (min(x + 8, x2), y2), color, 1)
    for y in range(y1, y2, 16):
        cv2.line(canvas, (x1, y), (x1, min(y + 8, y2)), color, 1)
        cv2.line(canvas, (x2, y), (x2, min(y + 8, y2)), color, 1)


def render(image, result, packet, output, tracks, *, total, fps, source_elapsed,
           inference_ms, process_fps, mode, state_totals):
    canvas = np.full((HEIGHT, WIDTH, 3), 20, np.uint8)
    camera, scale, ox, oy = _scale_camera(image)
    canvas[HEADER_H:HEADER_H + CAM_H, :CAM_W] = camera

    def pt(x, y):
        # Bound drawing coordinates only; no coordinates fed back into the pipeline.
        return (int(np.clip(round(x * scale) + ox, 0, CAM_W - 1)),
                int(np.clip(round(y * scale) + oy + HEADER_H, HEADER_H, HEADER_H + CAM_H - 1)))

    label(canvas, f'JGRadar | frame {packet.frame_seq + 1}/{total} | video {packet.frame_seq / fps:.1f}s'
          f' | capture +{source_elapsed:.1f}s | {process_fps:.1f} processed FPS | inference {inference_ms:.0f}ms', (8, 23), scale=.56)
    for index, car in enumerate(result.cars):
        x1, y1, x2, y2 = car.bbox
        a, b = pt(x1, y1), pt(x2, y2)
        cv2.rectangle(canvas, a, b, GREEN, 1, cv2.LINE_AA)
        identity = packet.observations[index].car_class
        label(canvas, f'C{index} candidate {car.confidence:.2f}' + (f' [{identity} vote]' if identity else ''),
              (a[0], max(HEADER_H + 14, a[1] - 7)), GREEN, .39)
        cv2.drawMarker(canvas, pt((x1 + x2) / 2, y2), YELLOW, cv2.MARKER_CROSS, 9, 1)
    for index, hit in enumerate(result.armors):
        x1, y1, x2, y2 = hit.bbox
        color = side_color('R' if hit.record.label == 'red' else 'B')
        if hit.record.label not in ('red', 'blue'):
            color = DIM
        a, b = pt(x1, y1), pt(x2, y2)
        cv2.rectangle(canvas, a, b, color, 2)
        vote = result.digits[index] if index < len(result.digits) else None
        digit = f'{vote.label} {vote.confidence:.2f}' if vote and vote.known else '?'
        label(canvas, f'A{hit.car_index} {hit.record.label} {digit}', (a[0], min(1030, b[1] + 13)), color, .4)

    published = {s.track_id: s for s in output.robots.values() if s.position_cm is not None}
    for track in tracks:
        box = track.predicted_bbox
        if box is None:
            continue
        a, b = pt(*box[:2]), pt(*box[2:])
        slot = published.get(track.track_id)
        color = side_color(slot.name) if slot else DIM
        if slot and slot.is_predicted:
            dashed_box(canvas, a, b, YELLOW)
        elif slot:
            cv2.rectangle(canvas, a, b, color, 2, cv2.LINE_AA)
        text = (f'{slot.name} {slot.state.value}' if slot else f'{track.label or "?"} unreported')
        label(canvas, f'T{track.track_id} {text}', (a[0], max(HEADER_H + 12, a[1] + 14)), color, .45)

    field = canvas[HEADER_H:HEADER_H + FH, PX:PX + PW]
    field[:] = (32, 34, 37)

    def fp(x, y):
        return int(x / FIELD_MAX_X_CM * (PW - 1)), int(y / FIELD_MAX_Y_CM * (FH - 1))

    for x in range(0, FIELD_MAX_X_CM + 1, 500):
        u, _ = fp(x, 0)
        cv2.line(field, (u, 0), (u, FH - 1), (62, 65, 70), 1)
    for y in range(0, FIELD_MAX_Y_CM + 1, 500):
        _, v = fp(0, y)
        cv2.line(field, (0, v), (PW - 1, v), (62, 65, 70), 1)
    cv2.rectangle(field, (0, 0), (PW - 1, FH - 1), DIM, 1)
    for obs in packet.observations:
        if obs.world_cm is not None:
            x, y = obs.world_cm
            if 0 <= x <= FIELD_MAX_X_CM and 0 <= y <= FIELD_MAX_Y_CM:
                cv2.circle(field, fp(x, y), 3, (110, 125, 115), -1)
    for slot in output.robots.values():
        if slot.position_cm is None:
            continue
        x, y = slot.position_cm
        u, v = fp(x, y)
        color = side_color(slot.name)
        cv2.circle(field, (u, v), 7, color, 2 if slot.is_predicted else -1, cv2.LINE_AA)
        label(field, slot.name + ('~' if slot.is_predicted else ''),
              (min(PW - 55, u + 9), max(15, v - 7)), color, .48)
    label(field, 'FIELD 28m x 15m | x ->, y down | grid 5m', (8, 18), DIM, .43)
    y = HEADER_H + FH + 26
    lines = [
        (f'Projection: {mode}  |  VISION ONLY', YELLOW),
        ('Ground plane cannot validate highland accuracy', YELLOW),
        (f'Candidates {len(result.cars)} | armors {len(result.armors)} | tracks {len(tracks)}', WHITE),
        (f'Published {len(published)} | solid=measured  hollow=predicted', WHITE),
        ('SLOT    STATE           X(m)   Y(m)   TRACK', DIM),
    ]
    abbreviations = {'measured': 'MEASURED', 'occlusion_hold': 'HOLD~', 'blind_prediction': 'BLIND~', 'missing': 'MISSING'}
    for name, slot in output.robots.items():
        pos = f'{slot.position_cm[0]/100:5.2f} {slot.position_cm[1]/100:6.2f}' if slot.position_cm else '    -      -'
        lines.append((f'{name:4} {abbreviations[slot.state.value]:12} {pos}   {slot.track_id or "-"}',
                      side_color(name) if slot.position_cm else DIM))
    lines += [('', DIM), ('GREEN: candidate, identity still needs evidence', GREEN),
              ('R/B: armor color + recognized digit', WHITE),
              ('T#: physical track; slot table = final output', WHITE),
              ('Dashed box / ~ : prediction, not observation', YELLOW),
              ('Offline replay: capture timestamps preserved', DIM),
              ('Every input frame is analyzed and saved', DIM),
              ('Space: pause / resume      Esc: stop', WHITE)]
    for text, color in lines:
        label(canvas, text, (PX + 7, y), color, .45)
        y += 24
    cv2.line(canvas, (PX - 8, 0), (PX - 8, HEIGHT), (65, 68, 75), 1)
    label(canvas, 'Full pipeline: detection > armor > digit > projection > tracking > arbitration > publication', (8, 1072), DIM, .48)
    return canvas


class Stats:
    def __init__(self):
        self.frames = 0
        self.candidates = 0
        self.armors = 0
        self.labeled = 0
        self.projected = 0
        self.outside = 0
        self.frames_with_output = 0
        self.states = Counter()
        self.per_slot = defaultdict(Counter)
        self.last_track = {}
        self.handoffs = Counter()
        self.identities = set()
        self.track_ids = set()
        self.inference_ms = []
        self.processing_ms = []
        self.nonmonotonic_stamps = 0
        self.previous_stamp = None

    def add(self, result, packet, output, tracks, infer_ms, process_ms):
        self.frames += 1
        self.candidates += len(result.cars)
        self.armors += len(result.armors)
        self.labeled += sum(bool(o.car_class) for o in packet.observations)
        self.projected += sum(o.world_cm is not None for o in packet.observations)
        self.outside += sum(o.world_cm is not None and not
                            (0 <= o.world_cm[0] <= FIELD_MAX_X_CM and 0 <= o.world_cm[1] <= FIELD_MAX_Y_CM)
                            for o in packet.observations)
        self.inference_ms.append(infer_ms)
        self.processing_ms.append(process_ms)
        self.track_ids.update(t.track_id for t in tracks)
        if self.previous_stamp is not None and packet.captured_at <= self.previous_stamp:
            self.nonmonotonic_stamps += 1
        self.previous_stamp = packet.captured_at
        self.frames_with_output += any(s.position_cm is not None for s in output.robots.values())
        for name, slot in output.robots.items():
            self.states[slot.state.value] += 1
            self.per_slot[name][slot.state.value] += 1
            if slot.position_cm is None:
                continue
            self.identities.add(name)
            if name in self.last_track and self.last_track[name] != slot.track_id:
                self.handoffs[name] += 1
            self.last_track[name] = slot.track_id

    def summary(self):
        def percentiles(values):
            return {name: round(float(np.percentile(values, p)), 2) if values else 0
                    for name, p in [('median', 50), ('p95', 95), ('max', 100)]}
        return dict(frames=self.frames, candidate_detections=self.candidates,
                    armor_detections=self.armors, identity_observations=self.labeled,
                    projected_observations=self.projected, outside_field_observations=self.outside,
                    frames_with_published_position=self.frames_with_output,
                    published_identities=sorted(self.identities), unique_track_ids=len(self.track_ids),
                    state_slot_frames=dict(self.states), per_slot_state_frames=dict(self.per_slot),
                    slot_track_handoffs=dict(self.handoffs), nonmonotonic_stamps=self.nonmonotonic_stamps,
                    inference_ms=percentiles(self.inference_ms), pipeline_ms=percentiles(self.processing_ms))


def analyzed_frames(source, loop, total):
    """顺序推理与画面编码并行；有界队列阻塞写入，绝不丢帧。

    跟踪器只由一个线程推进。轨迹在入队前深拷贝，避免画面显示下一帧
    已经修改过的跟踪状态。异常通过队列回传主线程，不能伪装成正常 EOF。
    """
    pending = Queue(maxsize=2)
    stopped = Event()

    def send(item):
        while not stopped.is_set():
            try:
                pending.put(item, timeout=.1)
                return
            except Full:
                pass

    def produce():
        try:
            for _ in range(total):
                if stopped.is_set():
                    return
                frame_start = time.perf_counter()
                frame = source.read()
                if frame is None:
                    raise RuntimeError('Decoder ended before the expected frame count')
                infer_start = time.perf_counter()
                cascade = loop.stage_a.cascade
                result = cascade.run(frame.image)
                infer_ms = (time.perf_counter() - infer_start) * 1000
                packet = loop.stage_a.run(Frame.with_cascade(frame.seq, frame.captured_at, result))
                output = loop.stage_b.process(packet, now=packet.captured_at)
                if output is None:
                    raise RuntimeError(f'Stage B rejected frame {frame.seq}')
                output.validate()
                errors = {n: getattr(getattr(cascade, n), 'inference_errors', 0)
                          for n in ('car', 'armor', 'digit')}
                if any(errors.values()):
                    raise RuntimeError(f'Model inference errors: {errors}')
                process_ms = (time.perf_counter() - frame_start) * 1000
                send((frame, result, packet, output, deepcopy(loop.stage_b.tracker.tracks), infer_ms, process_ms))
        except BaseException as exc:
            send(exc)
        finally:
            send(None)

    worker = Thread(target=produce, name='sequential-video-analysis', daemon=True)
    worker.start()
    try:
        while True:
            try:
                item = pending.get(timeout=.5)
            except Empty:
                if not worker.is_alive():
                    raise RuntimeError('Video analysis worker ended without EOF')
                continue
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stopped.set()
        worker.join(timeout=10)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', action='append', default=[])
    ap.add_argument('--video', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--require-gpu', action='store_true')
    ap.add_argument('--max-frames', type=int, default=0, help='0 = complete input')
    args = ap.parse_args(argv)
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'frames.jsonl').exists():
        raise FileExistsError(f'Output already contains a run: {out}')
    cfg = load_config(*(args.config or [ROOT / 'config/default.yaml', ROOT / 'config/ega_replay.yaml']))
    stats = Stats()
    atom_json(out / 'status.json', {'status': 'loading', 'frames': 0, 'video': args.video})
    if args.show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1600, 900)
        splash = np.full((HEIGHT, WIDTH, 3), 24, np.uint8)
        label(splash, 'Loading GPU models and replay calibration...', (80, 180), WHITE, 1.1)
        cv2.imshow(WINDOW, splash)
        cv2.waitKey(1)
    source = VideoFrameSource(args.video)
    writer = None
    started = time.perf_counter()
    last_update = started
    status, error, devices = 'error', None, {}
    total = min(source.info.frame_count, args.max_frames) if args.max_frames else source.info.frame_count
    fps = source.info.fps
    first_stamp, last_stamp = None, None
    sample_ids = set(np.linspace(0, max(0, total - 1), 10, dtype=int).tolist())
    ff_log = (out / 'encoder.log').open('w')
    source_meta = dict(video=str(source.path.resolve()), video_info=asdict(source.info), expected_frames=total,
                       timestamps=str(source.timestamps_path), configs=[str(p) for p in (args.config or [ROOT / 'config/default.yaml', ROOT / 'config/ega_replay.yaml'])],
                       projection_mode=cfg.transform.projection_mode, every_frame=True,
                       visualization_queue_capacity=2, drop_frames=False,
                       limitations=['No ground truth: coverage is not accuracy.',
                                    'Ground plane projection does not model highlands.',
                                    'Video only: no lidar, radio, camera USB or real-time scheduling validation.'])
    if source.calibration is not None:
        c = source.calibration
        source_meta['calibration'] = dict(K=c.K.tolist(), dist=c.dist.tolist(), R=c.R.tolist(),
                                          t_cm=c.t.tolist(), camera_origin_cm=(-c.R.T @ c.t).tolist())
    atom_json(out / 'metadata.json', source_meta)
    try:
        loop = build_loop(cfg, frame_source=source)
        cascade = loop.stage_a.cascade
        for name in ('car', 'armor', 'digit'):
            backend = getattr(cascade, name).backend
            session = getattr(backend, '_session', None)
            devices[name] = session.get_providers() if session else getattr(backend, 'device', 'unknown')
        if args.require_gpu:
            if any('CUDAExecutionProvider' not in devices[n] for n in ('car', 'armor')) or devices['digit'] != 'cuda':
                raise RuntimeError(f'GPU requested but not active in all models: {devices}')
        print(json.dumps({'event': 'ready', 'frames': total, 'devices': devices}, ensure_ascii=False), flush=True)
        writer = subprocess.Popen([
            'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-n',
            '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', f'{WIDTH}x{HEIGHT}',
            '-framerate', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
            '-crf', '23', '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(out / 'annotated.mp4')], stdin=subprocess.PIPE, stderr=ff_log)
        stream = analyzed_frames(source, loop, total)
        with (out / 'frames.jsonl').open('w', encoding='utf-8') as fh:
            for frame, result, packet, output, tracks, infer_ms, process_ms in stream:
                if first_stamp is None:
                    first_stamp = frame.captured_at
                last_stamp = frame.captured_at
                stats.add(result, packet, output, tracks, infer_ms, process_ms)
                elapsed = time.perf_counter() - started
                canvas = render(frame.image, result, packet, output, tracks, total=total, fps=fps,
                                source_elapsed=last_stamp-first_stamp, inference_ms=infer_ms,
                                process_fps=stats.frames/max(elapsed, .001), mode=cfg.transform.projection_mode,
                                state_totals=stats.states)
                record = dict(frame_index=frame.seq, video_time_s=frame.seq/fps,
                              captured_at=frame.captured_at, inference_ms=round(infer_ms, 3),
                              pipeline_ms=round(process_ms, 3), cascade=cascade_to_dict(result),
                              observations=[asdict(o) for o in packet.observations],
                              tracks=[dict(track_id=t.track_id, label=t.label, matched=t.matched,
                                           status=t.status.value, state=t.state.value if t.state else None,
                                           hits=t.hits, bbox=t.predicted_bbox, last_observed_at=t.last_observed_at)
                                      for t in tracks], output=output.to_dict())
                fh.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                writer.stdin.write(canvas.tobytes())
                ok, jpeg = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise RuntimeError('JPEG encoding failed')
                temp = out / 'live.tmp.jpg'
                temp.write_bytes(jpeg.tobytes())
                temp.replace(out / 'live.jpg')
                if frame.seq in sample_ids:
                    cv2.imwrite(str(out / f'frame_{frame.seq:05d}.jpg'), canvas)
                if args.show:
                    cv2.imshow(WINDOW, canvas)
                    key = cv2.waitKey(1) & 255
                    if key == 32:
                        print('Paused by user; Space resumes.', flush=True)
                        while key == 32 or key not in (27, ord('q')):
                            key = cv2.waitKey(100) & 255
                            if key == 32:
                                break
                        if key == 32:
                            key = -1
                    if key in (27, ord('q')) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        status = 'cancelled'
                        break
                progress = dict(status='running', frames=stats.frames, total_frames=total,
                                progress=stats.frames/total, video_time_s=frame.seq/fps,
                                source_elapsed_s=last_stamp-first_stamp, elapsed_s=round(elapsed, 2),
                                processing_fps=round(stats.frames/max(elapsed, .001), 2),
                                inference_ms=round(infer_ms, 2), devices=devices,
                                active_identities=[n for n,s in output.robots.items() if s.position_cm is not None],
                                per_slot={n:s.to_dict() for n,s in output.robots.items()})
                atom_json(out / 'status.json', progress)
                if time.perf_counter() - last_update >= 15 or stats.frames == 1:
                    print(json.dumps({k:v for k,v in progress.items() if k != 'per_slot'}, ensure_ascii=False), flush=True)
                    fh.flush()
                    last_update = time.perf_counter()
            else:
                if stats.frames != total:
                    raise RuntimeError(f'Processed {stats.frames} frames, expected {total}')
                status = 'complete'
    except KeyboardInterrupt:
        status = 'cancelled'
    except Exception as exc:
        error = str(exc)
        traceback.print_exc()
    finally:
        if 'stream' in locals():
            stream.close()
        source.close()
        if writer is not None:
            try:
                writer.stdin.close()
            except (OSError, BrokenPipeError):
                pass
            rc = writer.wait(timeout=120)
            if rc:
                status, error = 'error', f'Video encoder exit {rc}; {error or ""}'
        ff_log.close()
        if args.show:
            cv2.destroyAllWindows()
        summary = {**source_meta, **stats.summary(), 'status': status, 'error': error,
                   'devices': devices, 'elapsed_s': round(time.perf_counter()-started, 2),
                   'source_elapsed_s': (last_stamp-first_stamp) if last_stamp is not None else 0,
                   'inference_errors': loop.stage_a.diagnostics() if 'loop' in locals() else {}}
        atom_json(out / 'summary.json', summary)
        atom_json(out / 'status.json', {**summary, 'total_frames': total, 'progress': stats.frames/max(total, 1)})
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if status == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
