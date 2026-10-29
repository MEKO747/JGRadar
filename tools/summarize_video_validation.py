#!/usr/bin/env python3
"""从实际逐帧记录生成可回看的验证报告；不需要重新运行神经网络。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import html
import json
import math
from pathlib import Path

STATES = {'measured': ('实测', '#51c8b0'), 'occlusion_hold': ('短时预测', '#ebc06b'),
          'blind_prediction': ('盲区猜测', '#ad90e8'), 'missing': ('无坐标', '#343d4c')}


def analyze(directory):
    directory = Path(directory)
    meta = json.loads((directory / 'metadata.json').read_text())
    counters = defaultdict(Counter)
    handoffs, observed_ids, last_track = Counter(), set(), {}
    last_slots, transitions, runs = {}, [], defaultdict(list)
    jumps, unknown, no_world, cars, armors, n, detection_frames = [], 0, 0, 0, 0, 0, 0
    previous_index = -1
    gaps = []
    last_stamp = None
    timestamp_errors = 0
    frame_positions, inference = 0, []
    with (directory / 'frames.jsonl').open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break  # 正在写入的最后半行，等待下一次读取。
            seq, stamp = row['frame_index'], row['captured_at']
            if seq != previous_index + 1:
                gaps.append([previous_index, seq])
            if last_stamp is not None and stamp <= last_stamp:
                timestamp_errors += 1
            previous_index, last_stamp = seq, stamp
            n += 1
            cars += len(row['cascade']['cars'])
            armors += len(row['cascade']['armors'])
            detection_frames += bool(row['cascade']['cars'])
            unknown += sum(not o['car_class'] for o in row['observations'])
            no_world += sum(o['world_cm'] is None for o in row['observations'])
            inference.append(row['inference_ms'])
            slots = row['output']['robots']
            frame_positions += any(s['position_cm'] is not None for s in slots.values())
            for name, slot in slots.items():
                state = slot['state']
                counters[name][state] += 1
                if runs[name] and runs[name][-1][2] == state:
                    runs[name][-1][1] = seq + 1
                else:
                    runs[name].append([seq, seq + 1, state])
                    transitions.append(dict(frame=seq, video_time_s=row['video_time_s'], slot=name, state=state))
                if slot['position_cm'] is not None:
                    observed_ids.add(name)
                    if name in last_track and last_track[name] != slot['track_id']:
                        handoffs[name] += 1
                    last_track[name] = slot['track_id']
                if name in last_slots:
                    old_stamp, old = last_slots[name]
                    if old['state'] == state == 'measured':
                        distance = math.dist(old['position_cm'], slot['position_cm'])
                        if distance >= 200 and 0 < stamp - old_stamp <= .1:
                            jumps.append(dict(frame=seq, video_time_s=row['video_time_s'], slot=name,
                                              distance_cm=round(distance, 1), dt_ms=round((stamp-old_stamp)*1000, 2),
                                              previous_track=old['track_id'], track=slot['track_id'],
                                              previous_position_cm=old['position_cm'], position_cm=slot['position_cm']))
                last_slots[name] = (stamp, slot)
    audit = dict(frames=n, expected_frames=meta['expected_frames'], frame_index_gaps=gaps,
                 timestamp_errors=timestamp_errors, frames_with_detection=detection_frames,
                 frames_with_position=frame_positions, car_detections=cars, armor_detections=armors,
                 unidentified_observations=unknown, projection_failed_observations=no_world,
                 published_identities=sorted(observed_ids), per_slot_state_frames=dict(counters),
                 slot_track_handoffs=dict(handoffs), large_measured_jumps=jumps,
                 state_runs=dict(runs), state_transitions=transitions,
                 caveat='No ground truth: handoffs are not ground-truth identity switches; coverage is not accuracy.')
    return audit, meta


def write_report(directory):
    directory = Path(directory)
    audit, meta = analyze(directory)
    (directory / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    n = audit['frames']
    fps = meta['video_info']['fps']
    rows, svg = [], []
    for i, (name, counts) in enumerate(audit['per_slot_state_frames'].items()):
        rows.append('<tr><td>' + name + '</td>' + ''.join(f'<td>{counts.get(k, 0):,}</td>' for k in STATES)
                    + f'<td>{audit["slot_track_handoffs"].get(name, 0):,}</td></tr>')
        y = 20 + i * 26
        svg.append(f'<text x="0" y="{y+13}" fill="#c9d7e8">{name}</text>')
        for start, end, state in audit['state_runs'][name]:
            x = 48 + 1040 * start / max(n, 1)
            width = max(.3, 1040 * (end-start) / max(n, 1))
            desc = f'{name} {STATES[state][0]} {start/fps:.2f}–{end/fps:.2f}秒'
            svg.append(f'<rect x="{x:.3f}" y="{y}" width="{width:.3f}" height="19" fill="{STATES[state][1]}" '
                       f'onclick="seek({start/fps:.3f})"><title>{desc}</title></rect>')
    jump_rows = ''.join(f'<tr><td><button onclick="seek({max(0,j["video_time_s"]-1):.3f})">{j["video_time_s"]:.2f} 秒</button></td>'
                        f'<td>{j["slot"]}</td><td>{j["distance_cm"]/100:.2f} m</td><td>{j["previous_track"]} → {j["track"]}</td></tr>'
                        for j in sorted(audit['large_measured_jumps'], key=lambda j:-j['distance_cm'])[:30])
    samples = ''.join(f'<a href="{p.name}"><img src="{p.name}" loading="lazy"><span>第 {int(p.stem.split("_")[-1])+1:,} 帧</span></a>'
                      for p in sorted(directory.glob('frame_*.jpg')))
    legend = ' · '.join(f'<span style="color:{color}">■ {name}</span>' for name,color in STATES.values())
    status = json.loads((directory/'status.json').read_text())
    complete = status.get('status') == 'complete' and n == meta['expected_frames'] and not audit['frame_index_gaps']
    biggest = max(audit['large_measured_jumps'], key=lambda j:j['distance_cm'], default=None)
    largest_button = (f'<button onclick="seek({max(0,biggest["video_time_s"]-1):.3f})">'
                      f'看最大跳变：{biggest["slot"]}，{biggest["distance_cm"]/100:.2f} 米，'
                      f'{biggest["video_time_s"]:.2f} 秒</button>') if biggest else ''
    handoff_jumps = sum(j['previous_track'] != j['track'] for j in audit['large_measured_jumps'])
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>JGRadar 完整录像验证报告</title>
<style>body{{background:#11161e;color:#e6edf6;font:15px system-ui;margin:0}}main{{max-width:1200px;margin:auto;padding:24px}}h1{{font-size:24px}}p{{line-height:1.8;color:#b7c6d9}}video{{width:100%;background:#000}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #303e50;padding:9px;text-align:left}}a{{color:#8bd5ff}}button{{background:#203e55;color:#d6efff;border:1px solid #456783;padding:5px;cursor:pointer}}svg{{width:100%}}rect{{cursor:pointer}}.samples{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.samples img{{width:100%}}.samples a{{text-decoration:none}}code{{word-break:break-all}}</style>
<main><h1>JGRadar · {'全片验证完成' if complete else '阶段性验证记录'}</h1>
<p>已处理 {n:,} / {meta['expected_frames']:,} 帧。逐帧推理、逐帧保存，源采集时间戳用于跟踪。<br>本次使用 EGA 检测与数字权重，验证修复后的身份、投影、跟踪及发布链。没有人工标注真值，不能据此给出识别准确率或定位误差；平地投影不适用于验证高地定位精度。</p>
<video id="video" src="annotated.mp4?view=complete" poster="frame_00000.jpg" controls preload="metadata"></video>
<p><button onclick="seek(0)">从头播放完整标注录像</button></p>
<p>已复现的问题：① 近处 R3 在车体位置基本不变时反复更换跟踪号；② 不同位置的轨迹保留同一身份，造成上报坐标来回跳变。<br>
<button onclick="seek(1.1)">看 R3 换号：1.1 秒</button>　<button onclick="seek(70.8)">看 B1 跳变：71.6 秒</button>　<button onclick="seek(197.3)">看 B7 跳变：3 分 18 秒</button></p>
<p>{largest_button}<br>符合下述统计条件的坐标跳变中，{handoff_jumps:,} 次伴随跟踪号更换。</p>
<details><summary>展开 R3 逐帧对照</summary><img style="width:100%;max-width:1050px" src="evidence/r3_track_comparison.jpg" loading="lazy"></details>
<p>{legend}<br>点击下面的状态条，可跳到对应录像位置。横轴为保存视频的播放时间。</p>
<svg viewBox="0 0 1100 344" role="img" aria-label="12 个身份的逐帧状态时间轴">{''.join(svg)}</svg>
<p>以下按“每个身份、每一帧”统计。轨迹更换表示同一身份改用了另一条轨迹，不能直接等同于真实车辆身份认错。</p>
<table><tr><th>身份</th><th>实测帧</th><th>短时预测帧</th><th>盲区猜测帧</th><th>无坐标帧</th><th>轨迹更换次数</th></tr>{''.join(rows)}</table>
<p>连续两帧均为实测、间隔不超过 0.1 秒，却移动超过 2 米的片段：{len(audit['large_measured_jumps']):,} 处。这里只列出需回看的异常候选，不将其自动判为误识别。</p>
<table><tr><th>回看片段</th><th>身份</th><th>坐标跳变</th><th>轨迹变化</th></tr>{jump_rows or '<tr><td colspan="4">未发现符合上述条件的跳变</td></tr>'}</table>
<p><a href="annotated.mp4" download>完整标注录像</a> · <a href="frames.jsonl" download>逐帧详细结果</a> · <a href="summary.json">运行统计</a> · <a href="audit.json">完整状态与异常候选</a></p>
<div class="samples">{samples}</div>
<p>输入录像：<code>{html.escape(meta['video'])}</code><br>帧号连续性异常：{len(audit['frame_index_gaps'])}；采集时间倒退或重复：{audit['timestamp_errors']}。</p>
</main><script>function seek(t){{let v=document.getElementById('video');v.currentTime=t;v.play();v.scrollIntoView({{behavior:'smooth',block:'start'}})}}</script></html>'''
    (directory / 'report.html').write_text(page, encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('state_runs','state_transitions','large_measured_jumps')},ensure_ascii=False))
    print('large_measured_jumps',len(audit['large_measured_jumps']))


def save_jump_evidence(directory, limit=3):
    """用原始帧和当时已保存的结果重绘异常；不重复推理。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import cv2
    from types import SimpleNamespace as NS
    from jg_radar.io.frame_source import cascade_from_dict
    from jg_radar.types import FrameOutput
    from tools.validate_video import render

    directory = Path(directory)
    audit, meta = analyze(directory)
    selected = sorted(audit['large_measured_jumps'], key=lambda j:-j['distance_cm'])[:limit]
    wanted = {max(0, j['frame'] + offset) for j in selected for offset in (-1, 0, 1)}
    records = {}
    with (directory/'frames.jsonl').open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if row['frame_index'] in wanted:
                records[row['frame_index']] = row
    out = directory/'evidence'
    out.mkdir(exist_ok=True)
    status = json.loads((directory/'status.json').read_text()).get('status')
    use_saved_overlay = status == 'complete' and (directory/'annotated.mp4').is_file()
    cap = cv2.VideoCapture(str(directory/'annotated.mp4') if use_saved_overlay else meta['video'])
    next_index = 0
    try:
        for seq, row in sorted(records.items()):
            if seq != next_index:
                cap.set(cv2.CAP_PROP_POS_FRAMES, seq)
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot read evidence frame {seq}')
            next_index = seq + 1
            if use_saved_overlay:
                cv2.imwrite(str(out/f'jump_frame_{seq:05d}.jpg'), image)
                continue
            packet = NS(frame_seq=seq, captured_at=row['captured_at'],
                        observations=[NS(**o) for o in row['observations']])
            tracks = [NS(track_id=t['track_id'], predicted_bbox=t['bbox'], label=t['label']) for t in row['tracks']]
            panel = render(image, cascade_from_dict(row['cascade']), packet,
                           FrameOutput.from_dict(row['output']), tracks,
                           total=meta['expected_frames'], fps=meta['video_info']['fps'],
                           source_elapsed=seq/meta['video_info']['fps'], inference_ms=row['inference_ms'],
                           process_fps=0, mode=meta['projection_mode'], state_totals={})
            cv2.imwrite(str(out/f'jump_frame_{seq:05d}.jpg'), panel)
    finally:
        cap.release()
    print('Evidence frames saved:', len(records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--save-jumps', action='store_true')
    args = parser.parse_args()
    write_report(args.directory)
    if args.save_jumps:
        save_jump_evidence(args.directory)
