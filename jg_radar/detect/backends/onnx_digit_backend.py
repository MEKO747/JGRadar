"""数字分类的 ONNX 运行时：不再依赖 torch。

## 为什么不在运行时用 torch

torch 装在本机**用户 site**，而仓库全程 ``PYTHONNOUSERSITE=1``（用户 site 里有
numpy 2.5.5，会跟 GNU Radio / OpenCV 的 ABI 打架）。为了让 torch 可见去关掉那个
环境变量，等于把部署环境做坏。把 MobileNetV2 导成 ONNX 之后，运行时只需要
onnxruntime——GPU 上 64px 分类器是毫秒级。

导出见 ``tools/export_digit_onnx.py``（需要一次性的 torch + onnx）。

## 预处理必须和训练一致

``MobileNetBackend``（torch 版）做的是：裁图 → **补成正方形（黑边）** → resize 到
``input_size`` → RGB → ImageNet 均值方差归一化。这里逐条对齐，任何一条不一致都会
表现为"模型加载成功但识别率暴跌"，而且不会有异常。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jg_radar.detect.protocol import DetectionRecord

#: ImageNet 的均值/方差，与 torchvision 版逐位一致。
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

__all__ = ["OnnxDigitBackend"]


class OnnxDigitBackend:
    """与 ``MobileNetBackend`` 同接口的分类后端，执行引擎换成 onnxruntime。"""

    name = "onnx-digit"
    task = "classify"
    supports_tiling = False

    def __init__(self, path=None, *, input_size: int = 64, num_classes: int = 6) -> None:
        self.path = Path(path) if path else None
        self.input_size = int(input_size)
        self.num_classes = int(num_classes)
        self._session = None
        self._reason = "未加载"
        self.providers: list[str] = []

    # -- 生命周期 ---------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self._session is not None

    @property
    def reason(self) -> str:
        return self._reason

    def load(self) -> None:
        if self.is_ready:
            return
        if self.path is None or not self.path.is_file():
            self._reason = f"数字 ONNX 模型不存在：{self.path}"
            return
        try:
            import onnxruntime as ort

            from jg_radar.integration.gpu_env import preload_cuda_libs

            # 找不到 cuDNN 时 ORT 会静默退 CPU；先把目录递进去（与检测后端一致）
            self._cuda_preload = preload_cuda_libs(ort)
            available = ort.get_available_providers()
            providers = [name for name in ("CUDAExecutionProvider", "CPUExecutionProvider")
                         if name in available]
            self._session = ort.InferenceSession(str(self.path), providers=providers or None)
            self._input_name = self._session.get_inputs()[0].name
            self.providers = self._session.get_providers()
            self._reason = ""
        except Exception as exc:  # noqa: BLE001 —— 失败原因要能被上层看见
            self._session = None
            self._reason = f"数字 ONNX 模型加载失败：{type(exc).__name__}: {exc}"

    def warmup(self, n: int = 1) -> None:
        if not self.is_ready:
            return
        blank = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        for _ in range(max(1, n)):
            self.infer([blank])

    # -- 预处理 -----------------------------------------------------------
    def _prepare(self, image: np.ndarray) -> np.ndarray:
        """黑边补成正方形 → resize → RGB → 归一化 → CHW。"""
        import cv2

        image = np.asarray(image)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("数字分类输入为空")

        # 与 torch 版一致：先补成正方形（黑边居中），再整体缩放
        size = max(height, width)
        square = np.zeros((size, size, 3), dtype=image.dtype)
        top, left = (size - height) // 2, (size - width) // 2
        square[top:top + height, left:left + width] = image

        side = self.input_size
        if size > side:
            # 由大缩小用 AREA，效果接近 torchvision 的 antialias=True
            resized = cv2.resize(square, (side, side), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(square, (side, side), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - _MEAN) / _STD
        return np.ascontiguousarray(normalized.transpose(2, 0, 1))

    # -- 推理 -------------------------------------------------------------
    def infer(self, images):
        if not self.is_ready:
            raise RuntimeError(self.reason)
        if not images:
            return []
        batch = np.stack([self._prepare(image) for image in images]).astype(np.float32)
        logits = self._session.run(None, {self._input_name: batch})[0]
        logits = np.asarray(logits, dtype=np.float64)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)

        results = []
        for image, row in zip(images, probabilities):
            index = int(row.argmax())
            height, width = np.asarray(image).shape[:2]
            results.append([DetectionRecord(index, float(row[index]),
                                            (0.0, 0.0, float(width), float(height)))])
        return results

