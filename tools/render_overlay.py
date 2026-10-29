#!/usr/bin/env python3
"""叠层渲染：把一帧的**检测框**与**投影落点**画出来，落成视频与拼图。

    python3 tools/render_overlay.py \
        --config config/default.yaml --config config/ega_replay.yaml \
        --camera "" --field "" \
        --out /tmp/overlay.mp4 --sheet /tmp/overlay_sheet.png --max-frames 24

## 为什么需要它

这是方案里列了、但一直没实现的那个工具（``tools/render_overlay.py``）。
它补的是一类**别的测试补不上**的缺口：

单元测试能证明"这个函数在给定输入下返回什么"，但证明不了
**"这一帧里那个框，投到场地上的那个点，是不是真的在那辆车脚下"**。
那是个视觉问题，只能用眼睛回答。而且它极其便宜：反投影一次的成本
在噪声里，收益是"改了一行变换代码、坐标整体偏了两米"这种事故当场可见。

具体它同时验证三件事：

1. **检测**是不是真的在这些车上（框画在画面上，肉眼可判对错）；
2. **投影锚点**是不是落在车与地面的接触点——这是本项目最贵的几个像素
   （锚点用框中心而不是底边中点，远处会偏 87–182cm，见 ``lookup_map``）；
3. **场地落点**是不是落在"那辆车在场地上确实所在的位置"（右侧场地俯视图）。

## 面板

    左：相机画面（缩放后画框）      右：场地俯视图（2800×1500cm）+ 诊断文字
    - 绿框 = 整车检测，框下缘中点画一个小十字 = **投影锚点**
    - 橙框 = 装甲板检测（挂在哪个车框上，颜色深浅不同）
    - 场地图上的圆点 = 每个车框锚点投影出来的坐标；旁边的编号是车框下标
    - 场地图上的半透明网格 = 500cm 一格；中心线是 y=750

## 关于"全 MISSING"

叠层画的是 **Stage A 的输出**（检测 + 投影），不是对外发布的槽位。
所以在身份还没打通的时候（车的类别名不是合法槽位名），跟踪器按设计
不产出槽位、``FrameOutput`` 全是 MISSING——**而这张图上照样能看到
检测框和投影点**。两者的差别正是本项目契约的核心：看得见 ≠ 可以上报。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from jg_radar.config import load_config  # noqa: E402
from jg_radar.pipeline.build import build_loop, build_source  # noqa: E402
from jg_radar.transform.lookup_map import LookupMap  # noqa: E402
from jg_radar.types import FIELD_MAX_X_CM, FIELD_MAX_Y_CM  # noqa: E402

# 画布布局（1920×1080）
CANVAS_W, CANVAS_H = 1920, 1080
HEADER_H = 34
CAM_W, CAM_H = 1344, 1008          # 4:3，正好装下 4024×3036
PANEL_X = 1360                     # 右栏起点
FIELD_W = 552
FIELD_H = int(FIELD_W * FIELD_MAX_Y_CM / FIELD_MAX_X_CM)   # 保持场地真实比例

# 配色（BGR）
C_CAR = (80, 220, 80)
C_ARMOR = (60, 170, 255)
C_ANCHOR = (0, 255, 255)
C_POINT = (60, 220, 255)
C_GRID = (70, 70, 70)
C_TEXT = (235, 235, 235)
C_DIM = (150, 150, 150)
C_WARN = (80, 200, 255)


def _scale_camera(image: np.ndarray) -> np.ndarray:
    """把原始帧缩放进左侧面板（保持 4:3 的 letterbox，不拉伸）。"""
    h, w = image.shape[:2]
    scale = min(CAM_W / w, CAM_H / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    panel = np.full((CAM_H, CAM_W, 3), 24, dtype=np.uint8)
    ox, oy = (CAM_W - nw) // 2, (CAM_H - nh) // 2
    panel[oy:oy + nh, ox:ox + nw] = resized
    return panel, scale, ox, oy


def _draw_field(points, frame_seq: int) -> np.ndarray:
    """画场地俯视图 + 投影落点。``points`` 是 ``[(编号, x_cm, y_cm, ok)]``。"""
    panel = np.full((FIELD_H, FIELD_W, 3), 30, dtype=np.uint8)

    def to_px(x_cm: float, y_cm: float) -> tuple[int, int]:
        u = int(round(x_cm / FIELD_MAX_X_CM * (FIELD_W - 1)))
        v = int(round(y_cm / FIELD_MAX_Y_CM * (FIELD_H - 1)))
        return u, v

    # 网格：500cm 一格
    for x in range(0, FIELD_MAX_X_CM + 1, 500):
        u, _ = to_px(x, 0)
        cv2.line(panel, (u, 0), (u, FIELD_H - 1), C_GRID, 1)
    for y in range(0, FIELD_MAX_Y_CM + 1, 500):
        _, v = to_px(0, y)
        cv2.line(panel, (0, v), (FIELD_W - 1, v), C_GRID, 1)
    # 中线（y=750）与边框
    _, mid = to_px(0, FIELD_MAX_Y_CM / 2)
    cv2.line(panel, (0, mid), (FIELD_W - 1, mid), (110, 110, 110), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (FIELD_W - 1, FIELD_H - 1), (120, 120, 120), 1)

    for idx, x_cm, y_cm, ok in points:
        if not ok:
            continue
        u, v = to_px(float(np.clip(x_cm, 0, FIELD_MAX_X_CM)),
                     float(np.clip(y_cm, 0, FIELD_MAX_Y_CM)))
        cv2.circle(panel, (u, v), 5, C_POINT, -1, cv2.LINE_AA)
        cv2.circle(panel, (u, v), 5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(panel, str(idx), (u + 7, v - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, C_TEXT, 1, cv2.LINE_AA)

    cv2.putText(panel, f"FIELD  {FIELD_MAX_X_CM}x{FIELD_MAX_Y_CM} cm   frame {frame_seq}",
                (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)
    return panel


def _draw_text_block(lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    """右栏下方的诊断文字块。"""
    h = CANVAS_H - (HEADER_H + FIELD_H) - 8
    block = np.full((max(h, 10), FIELD_W, 3), 24, dtype=np.uint8)
    y = 22
    for text, color in lines:
        cv2.putText(block, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    color, 1, cv2.LINE_AA)
        y += 22
        if y > block.shape[0] - 6:
            break
    return block


def render_frame(
    image: np.ndarray,
    *,
    car_boxes: np.ndarray,
    car_labels: list[str],
    car_confs: list[float],
    anchors: np.ndarray,
    world: np.ndarray,
    valid: np.ndarray,
    armor_boxes: list[tuple[int, np.ndarray, float]],
    frame_seq: int,
    info: list[tuple[str, tuple[int, int, int]]],
) -> np.ndarray:
    """把一帧画成完整的 1920×1080 画布。"""
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 18, dtype=np.uint8)
    cv2.putText(canvas, "JGRadar overlay : camera view", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TEXT, 1, cv2.LINE_AA)

    panel, scale, ox, oy = _scale_camera(image)
    canvas[HEADER_H:HEADER_H + CAM_H, 0:CAM_W] = panel

    def px(x: float, y: float) -> tuple[int, int]:
        return int(round(x * scale)) + ox, int(round(y * scale)) + oy + HEADER_H

    # 车框 + 锚点
    for i, box in enumerate(car_boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        p1, p2 = px(x1, y1), px(x2, y2)
        cv2.rectangle(canvas, p1, p2, C_CAR, 2, cv2.LINE_AA)
        label = car_labels[i] if i < len(car_labels) else ""
        conf = car_confs[i] if i < len(car_confs) else 0.0
        tag = f"#{i} {label or '?'} {conf:.2f}"
        cv2.putText(canvas, tag, (p1[0], max(HEADER_H + 12, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_CAR, 1, cv2.LINE_AA)
        # 锚点：底边中点（投影用的就是它）
        ax, ay = px(anchors[i][0], anchors[i][1])
        cv2.drawMarker(canvas, (ax, ay), C_ANCHOR, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)

    # 装甲板框
    for car_index, box, conf in armor_boxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        p1, p2 = px(x1, y1), px(x2, y2)
        cv2.rectangle(canvas, p1, p2, C_ARMOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"a{car_index} {conf:.2f}", (p1[0], p2[1] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_ARMOR, 1, cv2.LINE_AA)

    # 右栏
    points = [(i, world[i][0], world[i][1], bool(valid[i])) for i in range(len(anchors))]
    canvas[HEADER_H:HEADER_H + FIELD_H, PANEL_X:PANEL_X + FIELD_W] = _draw_field(
        points, frame_seq
    )
    block_y = HEADER_H + FIELD_H + 8
    block = _draw_text_block(info)
    canvas[block_y:block_y + block.shape[0], PANEL_X:PANEL_X + FIELD_W] = block
    cv2.line(canvas, (PANEL_X - 8, 0), (PANEL_X - 8, CANVAS_H), (60, 60, 60), 1)
    return canvas


def write_html_report(path: str | Path, frames: list[tuple[int, np.ndarray]],
                      meta: dict[str, str]) -> Path:
    """把若干帧渲染结果写成一个**自包含** HTML（图片以 data-URI 内嵌）。

    为什么不是只给 mp4：编码器不一定可用（本机 OpenCV 的 mp4v 在某些构建里
    打不开），而"打不开视频"会让整个可视化白做。HTML + data-URI 不依赖任何
    解码器，双击就能在浏览器里翻页看，也方便直接发给别人。
    """
    import base64

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for seq, img in frames:
        small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2),
                           interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if not ok:
            continue
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        cards.append(
            f'<figure><img src="data:image/jpeg;base64,{b64}" alt="frame {seq}">'
            f'<figcaption>frame {seq}</figcaption></figure>'
        )
    rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in meta.items())
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>JGRadar 叠层报告</title>
<style>
 body{{background:#0d1117;color:#e6edf3;font-family:"PingFang SC","Microsoft YaHei",system-ui,sans-serif;margin:0;padding:24px}}
 h1{{font-size:19px;margin:0 0 6px}} p.sub{{color:#8b949e;font-size:13px;margin:0 0 18px}}
 table{{border-collapse:collapse;font-size:13px;margin-bottom:22px}}
 th{{text-align:left;color:#8b949e;font-weight:400;padding:3px 14px 3px 0;white-space:nowrap}}
 td{{padding:3px 0;font-family:ui-monospace,Consolas,monospace}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:16px}}
 figure{{margin:0;border:1px solid #30363d;border-radius:10px;overflow:hidden;background:#161b22}}
 img{{width:100%;display:block}}
 figcaption{{padding:6px 10px;color:#8b949e;font-size:12px}}
</style></head><body>
<h1>JGRadar · 叠层报告（检测 + 投影）</h1>
<p class="sub">绿框=整车 · 橙框=装甲板 · 十字=投影锚点（底边中点） · 右栏场地俯视图圆点=投影落点</p>
<table>{rows}</table>
<div class="grid">{"".join(cards)}</div>
</body></html>
"""
    p.write_text(html, encoding="utf-8")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="JGRadar 叠层渲染（检测 + 投影可视化）")
    ap.add_argument("--config", action="append", default=[],
                    help="配置 YAML，可给多次（后面的覆盖前面的）")
    ap.add_argument("--camera", default="", help="额外相机 YAML（空串跳过）")
    ap.add_argument("--field", default="", help="额外场地 YAML（空串跳过）")
    ap.add_argument("--out", default="", help="输出 mp4（不给就只出拼图）")
    ap.add_argument("--sheet", default="", help="输出拼图 PNG（均匀取样若干帧）")
    ap.add_argument("--html", default="",
                    help="输出**自包含** HTML 报告（图片内嵌，不依赖视频编码器）")
    ap.add_argument("--sheet-frames", type=int, default=6, help="拼图取几帧")
    ap.add_argument("--sheet-cols", type=int, default=3, help="拼图列数")
    ap.add_argument("--max-frames", type=int, default=24, help="最多处理多少帧")
    ap.add_argument("--every", type=int, default=0,
                    help="每 N 帧画一次（0 = 用配置里的 diagnostics.overlay_every_n_frames）")
    ap.add_argument("--show", action="store_true",
                    help="开一个实时窗口边跑边看（按 ESC 提前结束；需要 OpenCV 有 GUI 支持）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    paths = list(args.config)
    for extra in (args.camera, args.field):
        if extra and Path(extra).is_file() and extra not in paths:
            paths.append(extra)
    if not paths:
        ap.error("至少要给一个 --config")
    cfg = load_config(*paths)

    every = args.every or max(1, int(getattr(cfg.diagnostics, "overlay_every_n_frames", 30)))
    # 叠层是给人看的，所以每帧都画（every 只影响"是否处理这一帧"）
    source = build_source(cfg)
    loop = build_loop(cfg, frame_source=source)
    stage_a, stage_b = loop.stage_a, loop.stage_b
    lookup = stage_a.lookup

    fps = float(getattr(getattr(source, "info", None), "fps", 30.0))
    writer = None
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 max(1.0, fps) / max(1, every), (CANVAS_W, CANVAS_H))
        if not writer.isOpened():
            print("✗ 打不开视频写入器（这个 OpenCV 构建没有对应编码器？）。"
                  "先看有没有 ffmpeg：用 --sheet / --html 出图，或改用 .avi+MJPG。",
                  file=sys.stderr)
            writer = None

    sheets: list[np.ndarray] = []
    report_frames: list[tuple[int, np.ndarray]] = []
    n = 0
    rendered = 0
    cascade_obj = getattr(stage_a, "cascade", None)
    if cascade_obj is None:
        print("✗ Stage A 没有级联，画不出检测框（检查配置的 detect 段）", file=sys.stderr)
        return 2

    from jg_radar.io.frame_source import Frame, iterate

    for frame in iterate(source):
        if frame.image is None:
            continue
        if n % max(1, every) != 0:
            n += 1
            continue
        # 级联只跑一次：把结果塞回帧里，Stage A 看到 frame.cascade 就不会再跑一遍。
        # （Stage A 对"没带检测结果"的帧会自己跑级联，那是实时路径的行为。）
        cascade = cascade_obj.run(frame.image)
        packet = stage_a.run(
            Frame(seq=frame.seq, captured_at=frame.captured_at,
                  image=frame.image, cascade=cascade)
        )
        # 照样推进 Stage B：跟踪器/仲裁器的状态要跟真实跑法一致，
        # 否则叠层上的 tracks/reported 就不是运行时的那两个数了。
        stage_b.process(packet, now=loop.clock(packet))

        boxes = np.asarray([c.bbox for c in cascade.cars], dtype=np.float64).reshape(-1, 4)
        if len(boxes):
            world, valid = lookup.sample_bbox_ground(boxes)
            anchors = np.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5, boxes[:, 3]], axis=1)
        else:
            world = np.zeros((0, 2)); valid = np.zeros(0, dtype=bool)
            anchors = np.zeros((0, 2))
        armor_boxes = [(h.car_index, np.asarray(h.bbox, dtype=np.float64),
                        float(h.confidence)) for h in cascade.armors]
        info = [
            (f"frame      {frame.seq}   t={frame.captured_at:.3f}", C_TEXT),
            (f"cars       {len(cascade.cars)}", C_CAR),
            (f"armors     {len(cascade.armors)}", C_ARMOR),
            (f"projected  {int(valid.sum())}/{len(boxes)}", C_POINT),
            ("", C_DIM),
            ("tracks (内部)   " + str(int(stage_b.tracker.last_diagnostics.get("tracks", -1))),
             C_DIM),
            ("reported (对外) " + str(int(stage_b.tracker.last_diagnostics.get("reported", -1))),
             C_WARN),
            ("", C_DIM),
            ("绿框=整车  橙框=装甲板", C_DIM),
            ("十字=投影锚点(底边中点)", C_DIM),
            ("右图圆点=场地落点", C_DIM),
        ]
        canvas = render_frame(
            frame.image, car_boxes=boxes,
            car_labels=[c.label for c in cascade.cars],
            car_confs=[float(c.confidence) for c in cascade.cars],
            anchors=anchors, world=world, valid=valid,
            armor_boxes=armor_boxes, frame_seq=frame.seq, info=info,
        )
        if writer is not None:
            writer.write(canvas)
        if args.show:
            try:
                cv2.imshow("JGRadar overlay (ESC to stop)", canvas)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    print("（收到 ESC，提前结束）")
                    break
            except cv2.error as exc:  # 无 GUI 支持的构建会走到这里
                print(f"✗ 打不开窗口（这个 OpenCV 没有 GUI 支持）：{exc}\n"
                      f"  改用 --html 出报告，或 --out 出视频。", file=sys.stderr)
                args.show = False
        rendered += 1
        if (args.sheet or args.html) and len(report_frames) < args.sheet_frames:
            if rendered % max(1, (args.max_frames // args.sheet_frames)) == 0:
                report_frames.append((frame.seq, canvas))
                sheets.append(cv2.resize(canvas, (CANVAS_W // 2, CANVAS_H // 2),
                                         interpolation=cv2.INTER_AREA))
        n += 1
        if not args.quiet:
            print(f"  帧 {frame.seq:4d}  车 {len(cascade.cars):2d}  "
                  f"甲 {len(cascade.armors):2d}  "
                  f"投影有效 {int(valid.sum()):2d}/{len(boxes)}")
        if n >= args.max_frames:
            break

    if writer is not None:
        writer.release()
        print(f"✓ 已写出视频 {args.out}（{n} 帧）")

    if args.sheet and sheets:
        cols = max(1, args.sheet_cols)
        rows = int(np.ceil(len(sheets) / cols))
        th, tw = sheets[0].shape[:2]
        sheet = np.full((rows * th, cols * tw, 3), 12, dtype=np.uint8)
        for i, img in enumerate(sheets):
            r, c = divmod(i, cols)
            sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = img
        Path(args.sheet).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.sheet, sheet)
        print(f"✓ 已写出拼图 {args.sheet}（{len(sheets)} 帧，{cols} 列）")

    if args.html and report_frames:
        meta = {
            "帧源": str(getattr(source, "path", "?")),
            "分辨率": f"{getattr(getattr(source, 'info', None), 'width', '?')}"
                      f"x{getattr(getattr(source, 'info', None), 'height', '?')}",
            "处理帧数": f"{rendered} / {args.max_frames}",
            "标定": "视频自带" if getattr(source, "calibration", None) is not None
                    else "配置",
            "渲染步长": f"每 {every} 帧",
        }
        out = write_html_report(args.html, report_frames, meta)
        print(f"✓ 已写出 HTML 报告 {out}（{len(report_frames)} 帧内嵌）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
