#!/usr/bin/env python3
"""对齐体检 CLI —— M2 的验收门。

    python3 tools/validate_alignment.py --config config/default.yaml \
        --camera config/camera.yaml --field config/field.yaml \
        --keypoints field/keypoints.json

输出三个数：反投影 RMS(px)、反投影最大(px)、往返最大(cm)。
门限 3px / 8px / 2cm，任一不过就以非零码退出 —— 让 CI 能拦住它。

另有 ``--suggest`` 模式：只扫网格、报告轴向与建议变换，用于第一次接触
一个陌生的 PLY/STP 文件。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jg_radar.config import load_config  # noqa: E402
from jg_radar.field import (  # noqa: E402
    KeypointSet,
    coverage_report,
    detect_up_axis,
    load_mesh,
    suggest_mesh_to_field,
    validate_alignment,
)
from jg_radar.transform import CameraModel, Intrinsics, Pose  # noqa: E402


def build_camera_and_pose(cfg) -> tuple[CameraModel, Pose]:
    intr = Intrinsics.from_matrix(cfg.camera.K, cfg.camera.dist)
    camera = CameraModel(intr, cfg.camera.width, cfg.camera.height)
    pose = Pose(np.asarray(cfg.transform.R, dtype=np.float64),
                np.asarray(cfg.transform.t, dtype=np.float64))
    return camera, pose


def cmd_suggest(mesh_path: str) -> int:
    """报告网格的轴向/单位/包围盒，并给出建议变换。"""
    raw = load_mesh(mesh_path)
    lo, hi = raw.bounds()
    print(f"网格：{mesh_path}")
    print(f"  顶点 {raw.num_vertices} / 面 {raw.num_faces}")
    print(f"  包围盒 x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}] "
          f"z[{lo[2]:.2f},{hi[2]:.2f}]")

    up, conf = detect_up_axis(raw.vertices, raw.faces)
    print(f"  推测朝上轴：{up}（共面法线面积占比 {conf:.2f}）")

    # 第二个独立信号：场地是**扁**的，朝上的那根轴跨度应该最小。
    # 两者都不支持时才值得报警 —— 见下方的说明。
    extents = raw.extent()
    thinnest = "xyz"[int(np.argmin(extents))]
    corroborated = thinnest == up
    print(f"  最扁的轴：{thinnest}"
          f"（跨度 {np.round(extents, 2).tolist()}）——"
          f"{'与推测一致' if corroborated else '与推测不一致'}")
    if conf < 0.4 and not corroborated:
        print("  ⚠ 两个判据都不支持这个结果：这个网格可能不是场地，或地面被"
              "拆得很碎。请人工确认轴向。")
    elif conf < 0.4:
        print("  · 共面占比偏低，但最扁的轴支持它 —— 摄影测量出的场地网格常"
              "把地面拆成许多微小三角面，属正常。")

    place = suggest_mesh_to_field(raw.vertices, raw.faces)
    print("  " + place.describe())
    print("  建议 mesh_to_field（4×4，行主序）：")
    for row in place.matrix:
        print("    [" + ", ".join(f"{v:12.4f}" for v in row) + "]")
    print("\n  ⚠ 这只是起点。必须用关键点残差（不带 --suggest 再跑一次）确认。")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="JGRadar 对齐体检")
    ap.add_argument("--config", action="append", default=[],
                    help="主配置，可重复（后加载的覆盖先加载的）")
    ap.add_argument("--camera", help="相机配置（含 K/dist/R/t）")
    ap.add_argument("--field", help="场地配置")
    ap.add_argument("--keypoints", help="关键点 JSON")
    ap.add_argument("--mesh", help="只检查网格并给建议，不需要关键点")
    ap.add_argument("--suggest", action="store_true", help="只跑网格建议模式")
    ap.add_argument("--json", dest="json_out", help="把报告写成 JSON")
    ap.add_argument("--rms-max-px", type=float, default=3.0)
    ap.add_argument("--max-max-px", type=float, default=8.0)
    ap.add_argument("--roundtrip-max-cm", type=float, default=2.0)
    args = ap.parse_args(argv)

    if args.mesh and (args.suggest or not args.keypoints):
        return cmd_suggest(args.mesh)

    paths = list(args.config)
    for extra in (args.camera, args.field):
        if extra:
            paths.append(extra)
    if not paths:
        ap.error("至少要给一个 --config（或用 --mesh 走建议模式）")

    cfg = load_config(*paths)
    camera, pose = build_camera_and_pose(cfg)

    if not args.keypoints:
        ap.error("缺少 --keypoints")
    kps = KeypointSet.load(args.keypoints)
    print(kps.describe())

    plane_z = cfg.field.ground_plane_z
    report = validate_alignment(camera, pose, kps, plane_z=plane_z,
                                extra_diagnostics=coverage_report(camera, pose,
                                                                  plane_z=plane_z))
    print(report.summary())

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"  报告已写入 {out}")

    ok = report.acceptable(rms_max_px=args.rms_max_px,
                           max_max_px=args.max_max_px,
                           roundtrip_max_cm=args.roundtrip_max_cm)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
