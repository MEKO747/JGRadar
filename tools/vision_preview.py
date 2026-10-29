#!/usr/bin/env python3
"""Inspect real car/armor detections from a camera or video, without field projection."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from jg_radar.config import load_config
from jg_radar.detect.armor_detector import ArmorHit
from jg_radar.detect.factory import CascadeResult, build_cascade
from jg_radar.io.frame_source import Frame, ListFrameSource, iterate
from jg_radar.io.hikrobot_camera import CameraError, HikRobotCameraSource
from jg_radar.io.video_source import VideoFrameSource


class _CheckedBackend:
    """Expose inference failures that the normal detectors turn into empty results."""

    def __init__(self, backend):
        self.backend = backend
        self.error = None

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def infer(self, images):
        try:
            result = self.backend.infer(images)
            if len(result) != len(images):
                raise RuntimeError("推理结果数量与输入图像数量不一致")
            return result
        except Exception as error:
            self.error = error
            raise


def _put_latest(mailbox, item):
    try:
        mailbox.put_nowait(item)
    except queue.Full:
        try:
            mailbox.get_nowait()
        except queue.Empty:
            pass
        mailbox.put_nowait(item)


def _latest_frames(source):
    """Keep acquiring while inference runs; retain only the latest camera frame."""
    mailbox = queue.Queue(maxsize=1)
    stop = threading.Event()

    def capture():
        try:
            while not stop.is_set():
                frame = source.read()
                if frame is None:
                    raise CameraError("实时相机采集意外结束")
                _put_latest(mailbox, frame)
        except Exception as error:
            _put_latest(mailbox, error)

    worker = threading.Thread(target=capture, name="vision-camera-capture", daemon=True)
    worker.start()
    try:
        while True:
            item = mailbox.get()
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()
        # The SDK read has a finite timeout. Join before the camera is closed.
        worker.join()


def _run_detection(cascade, image, mode, armor_roi=None):
    if mode == "cascade":
        return cascade.run(image)
    if mode == "armor":
        cars, rois = [], np.empty((0, 4))
    else:
        cars, rois = cascade.car.detect_rois(image)
    height, width = image.shape[:2]
    region = np.array(armor_roi or (0, 0, 1, 1)) * (width, height, width, height)
    # 独立检查的装甲没有所属车辆，不能借用车框索引或推断身份。
    armors = [ArmorHit(-1, hit.record) for hit in cascade.armor.detect(image, [region])]
    digit = getattr(cascade, "digit", None)
    digits = digit.classify(image, [hit.bbox for hit in armors]) if digit is not None and armors else []
    return CascadeResult(cars=cars, rois=rois, armors=armors, digits=digits)


def _draw(image, result, fps, inference_ms, age_ms, device="", mode="cascade"):
    height, width = image.shape[:2]
    scale = min(1.0, 1280 / width, 900 / height)
    preview = cv2.resize(image, (round(width * scale), round(height * scale)))

    def box(bounds, label, color):
        x1, y1, x2, y2 = [round(float(v) * scale) for v in bounds]
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        cv2.putText(preview, label, (x1, max(72, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    for car in result.cars:
        box(car.bbox, f"CANDIDATE {car.confidence:.2f}", (60, 220, 60))
    for index, armor in enumerate(result.armors):
        label = armor.record.label.upper() if armor.record.label else f"ARMOR {armor.class_id}"
        if index < len(result.digits) and result.digits[index].known:
            label += f" {result.digits[index].label}"
        color = (255, 120, 40) if armor.record.label == "blue" else (40, 120, 255)
        box(armor.bbox, f"{label} {armor.confidence:.2f}", color)
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 63), (25, 25, 25), -1)
    age = f" | age {age_ms:.0f} ms" if age_ms is not None else ""
    cv2.putText(preview, f"candidates {len(result.cars)} | armors {len(result.armors)} | "
                f"{fps:.1f} FPS | infer {inference_ms:.0f} ms{age} | {device}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    cv2.putText(preview, f"{mode} | green: candidate | colored: armor | identity / position untested | Q / Esc",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 220, 255), 1)
    return preview


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", help="叠加配置，可重复")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--video", type=Path, help="改用录像测试；默认实拍相机")
    inputs.add_argument("--image", type=Path, help="改用单张图片检查")
    parser.add_argument("--mode", choices=("cascade", "independent", "armor"), default="cascade",
                        help="cascade 车框内找装甲；independent 两级分别检查；armor 只测装甲")
    parser.add_argument("--armor-roi", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
                        help="独立装甲测试的区域，坐标为图像宽高的比例 0～1；默认整图")
    parser.add_argument("--serial", default="", help="相机序列号")
    parser.add_argument("--exposure-us", type=float, default=10000, help="曝光微秒，退出时恢复")
    parser.add_argument("--gain-db", type=float, default=23, help="增益 dB，退出时恢复")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理帧数；0 持续预览")
    parser.add_argument("--no-display", action="store_true", help="不打开窗口")
    parser.add_argument("--save-frame", type=Path, help="保存首张带检测框的图片")
    parser.add_argument("--save-input", type=Path, help="保存首张原始输入图片，便于复核")
    parser.add_argument("--log", type=Path, help="保存逐帧检测和耗时 JSONL")
    args = parser.parse_args(argv)
    if args.max_frames < 0 or (args.no_display and not args.max_frames and not args.image):
        parser.error("无窗口测试需指定正数 --max-frames")
    if args.armor_roi:
        x1, y1, x2, y2 = args.armor_roi
        if args.mode == "cascade" or not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            parser.error("--armor-roi 仅用于独立测试，坐标须满足 0≤X1<X2≤1、0≤Y1<Y2≤1")

    source = frames = log = None
    try:
        paths = args.config or [ROOT / "config/default.yaml", ROOT / "config/vision_check.yaml"]
        config = load_config(*paths)
        if config.detect.backend != "onnx":
            raise ValueError("检测检查必须使用真实 ONNX 模型，不能使用模拟后端")
        cascade = build_cascade(config.detect)
        cascade.load()
        if config.detect.digit_model and not cascade.digit.is_ready:
            raise RuntimeError(f"数字模型未加载：{cascade.digit.backend.reason}")
        checked = []
        devices = []
        for name in (("armor",) if args.mode == "armor" else ("car", "armor")):
            detector = getattr(cascade, name)
            if not detector.is_ready:
                raise RuntimeError(f"{name} 模型未加载：{detector.backend.reason}")
            detector.backend = _CheckedBackend(detector.backend)
            checked.append(detector.backend)
            detector.warmup(1)
            if detector.backend.error is not None:
                raise RuntimeError(f"{name} 模型预热失败：{detector.backend.error}")
            providers = detector.backend._session.get_providers()
            devices.append("GPU" if "CUDAExecutionProvider" in providers else "CPU")
            print(f"{name} 模型就绪，推理设备：{providers}", flush=True)
        device = "/".join(devices)
        print("绿色框是一级候选目标；彩色框是二级装甲检测。本窗口不验证机器人编号或场地位置。", flush=True)
        print(f"检查模式：{args.mode}；独立装甲框不代表它属于某辆车。", flush=True)
        if args.image:
            image = cv2.imread(str(args.image))
            if image is None:
                raise ValueError(f"无法读取图片：{args.image}")
            source = ListFrameSource([Frame(seq=0, captured_at=0, image=image)])
            frames = iterate(source)
        elif args.video:
            source = VideoFrameSource(args.video)
            frames = iterate(source)
        else:
            source = HikRobotCameraSource(serial_number=args.serial,
                                          exposure_us=args.exposure_us, gain_db=args.gain_db)
            frames = _latest_frames(source)
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            log = args.log.open("w", encoding="utf-8")
        processed = cars = armors = 0
        inference_total = 0.0
        started = time.monotonic()
        title = "JGRadar vision check - Q / Esc"
        for frame in frames:
            before = time.monotonic()
            result = _run_detection(cascade, frame.image, args.mode, args.armor_roi)
            for backend in checked:
                if backend.error is not None:
                    raise RuntimeError(f"模型推理失败：{backend.error}") from backend.error
            now = time.monotonic()
            inference_ms = (now - before) * 1000
            age_ms = (now - frame.captured_at) * 1000 if source.is_live else None
            processed += 1
            cars += len(result.cars)
            armors += len(result.armors)
            inference_total += inference_ms
            fps = processed / max(now - started, 1e-6)
            preview = _draw(frame.image, result, fps, inference_ms, age_ms, device, args.mode)
            if args.save_input and processed == 1:
                args.save_input.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.save_input), frame.image):
                    raise RuntimeError(f"保存原图失败：{args.save_input}")
            if args.save_frame and processed == 1:
                args.save_frame.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.save_frame), preview):
                    raise RuntimeError(f"保存图片失败：{args.save_frame}")
            if log:
                row = {"frame_seq": frame.seq, "inference_ms": inference_ms,
                       "frame_age_ms": age_ms,
                       "device": device,
                       "mode": args.mode, "processing_fps": fps,
                       "cars": [{"bbox": list(c.bbox), "confidence": c.confidence}
                                for c in result.cars],
                       "armors": [{"bbox": list(a.bbox), "confidence": a.confidence,
                                   "label": a.record.label, "car_index": a.car_index}
                                  for a in result.armors],
                       "digits": [{"label": d.label, "confidence": d.confidence,
                                   "class_id": d.class_id} for d in result.digits]}
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
            if not args.no_display:
                if processed == 1:
                    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(title, preview.shape[1], preview.shape[0])
                cv2.imshow(title, preview)
                if cv2.waitKey(0 if args.image else 1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and processed >= args.max_frames:
                break
        if processed == 0:
            raise RuntimeError("没有读到图像，检测检查未完成")
        print(f"已处理 {processed} 帧，一级候选框累计 {cars}，装甲检测累计 {armors}；"
              f"平均推理 {inference_total / processed:.1f} ms。", flush=True)
        return 0
    except KeyboardInterrupt:
        return 0
    except (CameraError, RuntimeError, ValueError, OSError, cv2.error) as error:
        print(f"视觉检查失败：{error}", file=sys.stderr)
        return 1
    finally:
        if frames is not None:
            frames.close()
        if source is not None:
            source.close()
        if log is not None:
            log.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
