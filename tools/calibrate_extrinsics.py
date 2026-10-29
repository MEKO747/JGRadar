#!/usr/bin/env python3
"""外参标定 CLI：关键点 → R|t。

    python3 tools/calibrate_extrinsics.py \
        --keypoints field/keypoints.json \
        --camera config/camera.yaml \
        --out config/extrinsics.yaml

输入是关键点文件（场地坐标 + 像素），输出是一份只含 ``transform`` 组的 YAML。
**残差不过门限就不写文件** —— 宁可不标定，也不要一个系统性偏了的 R|t 悄悄
进到配置里。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jg_radar.config import load_config  # noqa: E402
from jg_radar.field import KeypointSet, validate_alignment  # noqa: E402
from jg_radar.transform import (  # noqa: E402
    CameraModel,
    Intrinsics,
    camera_center,
    solve_pnp_ransac,
)

HEADER = """# 由 tools/calibrate_extrinsics.py 生成 —— 请勿手改。
# 重新标定请重跑该工具；手改会让残差记录与实际值脱节。
#
# 本次标定质量：
#   参与点 {n_used} / {n_total}（内点 {n_inliers}）
#   反投影 RMS {rms:.4f} px，最大 {max_px:.4f} px
#   往返最大 {trip:.4f} cm
#   相机光心（场地系）{center}
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="JGRadar 外参标定")
    ap.add_argument("--keypoints", required=True)
    ap.add_argument("--camera", required=True, help="含 K/dist 的配置")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", action="append", default=[],
                    help="额外配置（用于地面高度等）")
    ap.add_argument("--thresh-px", type=float, default=6.0, help="RANSAC 内点门限")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="即使残差不过门限也写出结果（仅用于排查）")
    args = ap.parse_args(argv)

    cfg = load_config(args.camera, *args.config)
    intr = Intrinsics.from_matrix(cfg.camera.K, cfg.camera.dist)
    camera = CameraModel(intr, cfg.camera.width, cfg.camera.height)

    kps = KeypointSet.load(args.keypoints)
    usable = kps.calibratable
    if len(usable) < 6:
        print(f"✗ 只有 {len(usable)} 个带像素的关键点，PnP 至少需要 6 个", file=sys.stderr)
        return 2

    world = kps.world_array(only_calibratable=True)
    pixels = kps.pixel_array()

    result = solve_pnp_ransac(
        world, pixels, intr,
        thresh_px=args.thresh_px, refine=not args.no_refine,
    )
    if result is None:
        print("✗ PnP 求解失败（点共线/退化，或内点不足）", file=sys.stderr)
        return 2

    pose = result.pose
    report = validate_alignment(camera, pose, kps, plane_z=cfg.field.ground_plane_z)
    print(report.summary())

    ok = report.acceptable(
        rms_max_px=cfg.transform.reproj_rms_max_px,
        max_max_px=cfg.transform.reproj_max_px,
        roundtrip_max_cm=cfg.transform.roundtrip_max_cm,
    )
    if not ok and not args.force:
        print("\n✗ 残差未过门限，**不写出配置**。请检查：", file=sys.stderr)
        for f in report.failures(
            rms_max_px=cfg.transform.reproj_rms_max_px,
            max_max_px=cfg.transform.reproj_max_px,
            roundtrip_max_cm=cfg.transform.roundtrip_max_cm,
        ):
            print(f"    {f}", file=sys.stderr)
        print("  常见原因：关键点像素点错、场地坐标量错、内参不对、"
              "或关键点几乎共面。", file=sys.stderr)
        return 1

    center = camera_center(pose.R, pose.t)
    text = HEADER.format(
        n_used=len(usable), n_total=len(kps), n_inliers=int(result.inliers.sum()),
        rms=report.rms_px, max_px=report.max_px, trip=report.max_roundtrip_cm,
        center=np.round(center, 2).tolist(),
    )
    text += "transform:\n"
    text += "  R:\n"
    for row in pose.R:
        text += "    - [" + ", ".join(f"{v:.10f}" for v in row) + "]\n"
    text += "  t: [" + ", ".join(f"{v:.6f}" for v in pose.t) + "]\n"
    text += f"  reproj_rms_px: {report.rms_px:.6f}\n"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\n✓ 已写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
