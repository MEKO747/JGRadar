#!/usr/bin/env python3
"""Preview and optionally save frames from a HIKROBOT MVS camera."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from jg_radar.io.hikrobot_camera import CameraError, HikRobotCameraSource


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0, help="MVS 设备序号（默认 0）")
    parser.add_argument("--serial", default="", help="按相机序列号选择设备")
    parser.add_argument("--width", type=int, default=0, help="采集宽度；0 使用相机默认值")
    parser.add_argument("--height", type=int, default=0, help="采集高度；0 使用相机默认值")
    parser.add_argument("--exposure-us", type=float, help="手动曝光时间（微秒），退出时恢复")
    parser.add_argument("--gain-db", type=float, help="手动增益（dB），退出时恢复")
    parser.add_argument("--save-frame", type=Path, help="保存首帧到指定图片路径")
    parser.add_argument("--no-display", action="store_true", help="完成采集测试后退出，不打开预览窗口")
    parser.add_argument("--frames", type=int, default=1, help="无窗口测试的采集帧数（默认 1）")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames 必须大于等于 1")

    try:
        with HikRobotCameraSource(
            camera_index=args.camera_index,
            serial_number=args.serial,
            width=args.width,
            height=args.height,
            exposure_us=args.exposure_us,
            gain_db=args.gain_db,
        ) as camera:
            first = camera.read()
            assert first.image is not None
            height, width = first.image.shape[:2]
            print(f"相机出图正常：{width}×{height}，BGR8，首帧 seq={first.seq}")
            if args.save_frame:
                args.save_frame.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.save_frame), first.image):
                    raise RuntimeError(f"图片保存失败：{args.save_frame}")
                print(f"已保存首帧：{args.save_frame}")
            if args.no_display:
                last = first
                for _ in range(args.frames - 1):
                    last = camera.read()
                if args.frames > 1:
                    elapsed = last.captured_at - first.captured_at
                    print(f"连续取流通过：{args.frames} 帧，{(args.frames - 1) / elapsed:.1f} fps")
                return 0

            title = "MV-CS120-10UC — 按 Q 或 Esc 退出"
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            scale = min(1.0, 1280.0 / width, 900.0 / height)
            display_size = (round(width * scale), round(height * scale))
            cv2.resizeWindow(title, *display_size)
            try:
                frame = first
                while True:
                    assert frame.image is not None
                    preview = cv2.resize(frame.image, display_size) if scale < 1.0 else frame.image.copy()
                    cv2.putText(preview, f"Live | frame {frame.seq}", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow(title, preview)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                    frame = camera.read()
            finally:
                cv2.destroyAllWindows()
    except KeyboardInterrupt:
        return 0
    except (CameraError, ValueError) as exc:
        print(f"相机测试失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
