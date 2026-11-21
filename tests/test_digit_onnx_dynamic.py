"""数字分类 ONNX 必须是**动态 batch**。

`digit.onnx` 是派生物（不进 git），用 ``tools/export_digit_onnx.py`` 生成。
第一次导出时 dummy 张量是 ``[1,3,64,64]``，TorchScript 导出器就把 batch 钉死成
1——「导出成功、单张正常」，但同一帧里认出两块装甲板时 ``infer`` 会堆成
``[2,3,64,64]`` 送进去，ORT 直接 ``INVALID_ARGUMENT``，而
``DigitClassifier`` 捕异常后把整批留在 ``unknown``：两块装甲的数字同时丢失，
不报错、不影响别的指标，只是融合永远认不出编号。

这份测试盯住三件事：导出后的图 batch 维是符号维；ORT 真能跑 batch=2；
分类器对两块装甲都给出结论而不是 unknown。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

#: `config/vision_check.yaml` 指向的派生物。换机器要按文档重导。
MODEL = Path(
    "/home/meko/radar/EGA_Radar_Algorithm-main/Main_ws/weights/digit/digit_mobilenetv2.onnx"
)

needs_model = pytest.mark.skipif(not MODEL.is_file(), reason="数字 ONNX 派生物不在本机，需重导")


def test_dynamic_batch_check_rejects_a_non_onnx_file(tmp_path):
    """校验函数不能对坏文件返回 True——否则导出脚本会假装成功。"""
    from tools.export_digit_onnx import _check_dynamic_batch

    bogus = tmp_path / "not_a_model.onnx"
    bogus.write_bytes(b"definitely not onnx")
    assert _check_dynamic_batch(bogus) is False


@needs_model
def test_exported_model_declares_a_symbolic_batch_dimension():
    from tools.export_digit_onnx import _check_dynamic_batch

    assert _check_dynamic_batch(MODEL) is True, "batch 维又被钉死了，重导时别忘了 dynamic_axes"


@needs_model
def test_onnx_runtime_accepts_a_batch_of_two():
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    logits = session.run(None, {name: np.zeros((2, 3, 64, 64), dtype=np.float32)})[0]
    assert logits.shape == (2, 6)


@needs_model
def test_two_armor_plates_on_one_frame_both_get_a_class():
    """故障注入的对应节点：两块装甲同帧出现，数字分类不许整批退化成 unknown。"""
    from jg_radar.detect.backends.onnx_digit_backend import OnnxDigitBackend
    from jg_radar.detect.digit_classifier import DigitClassifier

    backend = OnnxDigitBackend(MODEL, input_size=64, num_classes=6)
    backend.load()
    assert backend.is_ready
    classifier = DigitClassifier(backend, min_confidence=0.0)
    frame = np.zeros((240, 640, 3), dtype=np.uint8)
    votes = classifier.classify(frame, [[10, 10, 80, 80], [300, 120, 380, 190]])
    assert len(votes) == 2
    assert classifier.inference_errors == 0, classifier.last_error
    assert all(vote.known for vote in votes), "同帧两块装甲必须都读出结论"
