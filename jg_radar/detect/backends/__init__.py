"""推理后端实现。

四个后端对应四种处境：

* ``BlobBackend``——**没有任何权重**时也能做真实检测，用来验证级联的坐标；
* ``StubBackend``——确定性伪检测，让流水线在无权重时可回放、可比金标准；
* ``OnnxBackend``——可移植的真实模型（首选）；
* ``EmptyBackend``——永远返回空，专门用来测降级链。

TensorRT 后端等硬件到位后加入；它继承同一个 ``DetectorBackend`` 协议，
检测层与跟踪层都不需要改动。
"""

from jg_radar.detect.backends.blob_backend import BlobBackend
from jg_radar.detect.backends.onnx_backend import OnnxBackend
from jg_radar.detect.backends.stub_backend import EmptyBackend, StubBackend

__all__ = ["BlobBackend", "EmptyBackend", "OnnxBackend", "StubBackend"]
