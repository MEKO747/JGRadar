"""EGA MobileNet 数字分类适配；模型只读加载，不下载预训练权重。"""

from pathlib import Path

import numpy as np

from jg_radar.detect.protocol import DetectionRecord


class MobileNetBackend:
    name = "mobilenet"
    task = "classify"
    supports_tiling = False

    def __init__(self, path=None, *, input_size=64, num_classes=6):
        self.path = Path(path) if path else None
        self.input_size = int(input_size)
        self.num_classes = int(num_classes)
        self.model = None
        self._reason = "未加载"
        self.device = "cpu"

    @property
    def is_ready(self):
        return self.model is not None

    @property
    def reason(self):
        return self._reason

    def load(self):
        if self.is_ready:
            return
        if self.path is None or not self.path.is_file():
            self._reason = f"数字权重不存在：{self.path}"
            return
        try:
            import torch
            from torchvision.models import mobilenet_v2
            from torchvision import transforms

            checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
            state = checkpoint.get("model_state_dict", checkpoint)
            state = {key.removeprefix("backbone."): value for key, value in state.items()}
            model = mobilenet_v2(weights=None, num_classes=self.num_classes)
            model.load_state_dict(state, strict=True)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = model.to(self.device).eval()
            self._transform = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self._reason = ""
        except Exception as exc:
            self.model = None
            self._reason = f"数字模型加载失败：{exc}"

    def infer(self, images):
        if not self.is_ready:
            raise RuntimeError(self.reason)
        if not images:
            return []
        import torch
        from PIL import Image

        tensors = []
        for image in images:
            image = np.asarray(image)
            height, width = image.shape[:2]
            if not height or not width:
                raise ValueError("数字分类输入为空")
            rgb = Image.fromarray(np.ascontiguousarray(image[:, :, ::-1])).convert("RGB")
            size = max(width, height)
            square = Image.new("RGB", (size, size), (0, 0, 0))
            square.paste(rgb, ((size - width) // 2, (size - height) // 2))
            tensors.append(self._transform(square))
        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(batch), dim=1).cpu().numpy()
        results = []
        for image, row in zip(images, probabilities):
            index = int(row.argmax())
            results.append([DetectionRecord(index, float(row[index]),
                           (0.0, 0.0, float(image.shape[1]), float(image.shape[0])))])
        return results

    def warmup(self, n=1):
        if self.is_ready:
            self.infer([np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)])
