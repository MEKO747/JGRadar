"""ONNX Runtime 后端：优先 CUDA，缺库或缺权重时报告未就绪原因。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from jg_radar.detect.protocol import DetectionRecord

__all__ = ["OnnxBackend"]


class OnnxBackend:
    """YOLO 风格的单阶段检测 ONNX 模型。

    参数
    ----
    path: 模型路径。不存在时保持未就绪。
    input_size: 网络输入边长。
    num_classes / conf_thresh / iou_thresh / max_dets: 解码参数。
    output_layout: ``"xywh_conf_cls"`` （YOLOv8 系，(1, 4+nc, N)）或
        ``"xywh_obj_cls"``（YOLOv5 系，(1, N, 5+nc)）。两种摆法差一个转置，
        转错了不会报错，只会得到一堆乱框——所以必须显式配置，不猜。
    """

    supports_tiling = False

    def __init__(
        self,
        path: str | os.PathLike | None = None,
        *,
        input_size: int = 640,
        num_classes: int = 6,
        class_offset: int = 0,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        max_dets: int = 32,
        output_layout: str = "xywh_conf_cls",
        name: str = "onnx",
    ) -> None:
        self.path = None if path is None else str(path)
        self.input_size = int(input_size)
        self.num_classes = int(num_classes)
        self.class_offset = int(class_offset)
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.max_dets = int(max_dets)
        self.output_layout = output_layout
        self.name = name
        self._session = None
        self._input_name: str | None = None
        self._reason = "未加载"
        self._batch_size: int | None = 1

    # -- DetectorBackend --------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self._session is not None

    @property
    def reason(self) -> str:
        """没就绪的原因。让操作员一眼看出是缺库还是缺权重。"""
        return self._reason

    def load(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError:
            self._reason = "未安装 onnxruntime（pip install onnxruntime 或 onnxruntime-gpu）"
            return
        if not self.path or not os.path.exists(self.path):
            self._reason = f"权重文件不存在：{self.path}"
            return
        try:
            available = ort.get_available_providers()
            # pip 安装的 NVIDIA 运行库未必位于系统动态库搜索路径。
            # 在创建 CUDA 会话前加载它们，否则 ORT 可能退回 CPU。
            if "CUDAExecutionProvider" in available and hasattr(ort, "preload_dlls"):
                cudnn_dir = None
                if sys.platform == "linux":
                    # cuDNN 可以来自用户 site-packages，而 ORT 位于虚拟环境。
                    # ORT 的默认搜索只查看它自己的相邻 nvidia 目录。
                    try:
                        spec = importlib.util.find_spec("nvidia.cudnn")
                    except ModuleNotFoundError:
                        spec = None
                    if spec is not None:
                        for root in spec.submodule_search_locations or ():
                            directory = Path(root) / "lib"
                            if (directory / "libcudnn.so.9").is_file():
                                cudnn_dir = str(directory)
                                break
                ort.preload_dlls(cuda=True, cudnn=cudnn_dir is None, msvc=False)
                if cudnn_dir:
                    ort.preload_dlls(cuda=False, cudnn=True, msvc=False, directory=cudnn_dir)
            providers = [
                p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                if p in available
            ]
            self._session = ort.InferenceSession(self.path, providers=providers or None)
            self._input_name = self._session.get_inputs()[0].name
            shape = getattr(self._session.get_inputs()[0], "shape", [1])
            self._batch_size = shape[0] if isinstance(shape[0], int) else None
            self._reason = ""
        except Exception as exc:  # noqa: BLE001
            self._reason = f"加载失败：{exc}"
            self._session = None

    def warmup(self, n: int = 20) -> None:
        if not self.is_ready:
            return
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self.infer([dummy] * max(1, min(n, 4)))

    def infer(self, images: Sequence[np.ndarray]) -> list[list[DetectionRecord]]:
        if not self.is_ready:
            raise RuntimeError(f"OnnxBackend 未就绪：{self._reason}")
        if not images:
            return []
        if self._batch_size == 1:
            return [self._detect_one(np.asarray(img)) for img in images]
        size = self._batch_size or len(images)
        results = []
        for start in range(0, len(images), size):
            chunk = images[start:start + size]
            blobs = [self._preprocess(np.asarray(img)) for img in chunk]
            if self._batch_size:
                blobs.extend([blobs[-1]] * (size - len(blobs)))
            output = np.asarray(self._session.run(None, {
                self._input_name: np.concatenate(blobs, axis=0)
            })[0])
            if output.ndim != 3 or output.shape[0] != len(blobs):
                raise ValueError(f"批量检测输出形状不匹配：{output.shape}")
            for index, img in enumerate(chunk):
                results.append(self._decode(output[index:index + 1], *img.shape[:2]))
        return results

    # -- 内部 -------------------------------------------------------------
    def _detect_one(self, img: np.ndarray) -> list[DetectionRecord]:
        blob = self._preprocess(img)
        outputs = self._session.run(None, {self._input_name: blob})
        return self._decode(np.asarray(outputs[0]), *img.shape[:2])

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """BGR uint8 ``(H,W,3)`` → NCHW float32 RGB，归一化到 [0,1]。

        这里**不做 letterbox**：调用方（``BaseDetector._infer_canvas``）已经把
        图 letterbox 成正方形了，此处再补一次会缩放两次、坐标就对不上了。
        """
        h, w = img.shape[:2]
        if (w, h) != (self.input_size, self.input_size):
            raise ValueError(
                f"输入尺寸应为 {self.input_size}×{self.input_size}，得到 {w}×{h}；"
                "letterbox 是调用方的责任"
            )
        rgb = img[:, :, ::-1] if img.ndim == 3 else np.repeat(img[:, :, None], 3, axis=2)
        x = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return x[None]

    def _decode(self, out: np.ndarray, src_h: int, src_w: int) -> list[DetectionRecord]:
        """解出画布坐标的检测框（画布已经等于网络输入尺寸，所以直接就是画布坐标）。"""
        from jg_radar.detect.nms import nms

        if out.ndim == 3:
            out = out[0]
        if out.ndim != 2:
            return []
        if self.output_layout == "xywh_conf_cls":
            pred = out.T if out.shape[0] < out.shape[1] else out
        else:
            pred = out if out.shape[1] == 5 + self.num_classes else out.T
        if pred.shape[1] < 5:
            return []

        boxes = pred[:, :4].astype(np.float64)
        scores = pred[:, 4:].astype(np.float64)
        cls = scores.argmax(axis=1)
        conf = scores[np.arange(len(scores)), cls]
        keep = conf >= self.conf_thresh
        if not keep.any():
            return []

        boxes, cls, conf = boxes[keep], cls[keep], conf[keep]
        # xywh(中心) → xyxy(角点)
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0

        idx = nms(xyxy, conf, self.iou_thresh)[: self.max_dets]
        return [
            DetectionRecord(
                class_id=int(cls[i]) + self.class_offset,
                confidence=float(conf[i]),
                bbox=tuple(float(v) for v in xyxy[i]),
            )
            for i in idx
        ]
