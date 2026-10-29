from __future__ import annotations

import ctypes
from types import ModuleType

import numpy as np
import pytest

from jg_radar.io.hikrobot_camera import CameraError, HikRobotCameraSource, _Bindings, _copy_bgr8


class _UsbInfo(ctypes.Structure):
    _fields_ = [("chModelName", ctypes.c_char * 64), ("chSerialNumber", ctypes.c_char * 32)]


class _SpecialInfo(ctypes.Union):
    _fields_ = [("stUsb3VInfo", _UsbInfo), ("stGigEInfo", _UsbInfo)]


class _DeviceInfo(ctypes.Structure):
    _fields_ = [("nTLayerType", ctypes.c_uint), ("SpecialInfo", _SpecialInfo)]


class _DeviceList:
    nDeviceNum = 0
    pDeviceInfo = None


class _FrameInfo:
    nWidth = 0
    nHeight = 0
    nFrameLen = 0
    enPixelType = 0


class _FrameOut:
    def __init__(self):
        self.stFrameInfo = _FrameInfo()
        self.pBufAddr = ctypes.POINTER(ctypes.c_ubyte)()


class _ConvertParam:
    pass


class _IntValue:
    nCurValue = 0


class _EnumValue:
    nCurValue = 0


class _FloatValue:
    fCurValue = 0.0


class _FakeCamera:
    device = _DeviceInfo()
    last = None

    def __init__(self):
        self.calls = []
        self.buffer = None
        type(self).last = self

    @staticmethod
    def MV_CC_Initialize():
        return 0

    @staticmethod
    def MV_CC_Finalize():
        return 0

    def MV_CC_EnumDevices(self, mask, device_list):
        self.device.nTLayerType = 4
        self.device.SpecialInfo.stUsb3VInfo = _UsbInfo(b"MV-CS120-10UC", b"ABC123")
        device_list.nDeviceNum = 1
        device_list.pDeviceInfo = (ctypes.POINTER(_DeviceInfo) * 1)(ctypes.pointer(self.device))
        self.calls.append(("enum", mask))
        return 0

    def MV_CC_CreateHandle(self, info):
        self.calls.append("create")
        return 0

    def MV_CC_OpenDevice(self, access, mode):
        self.calls.append(("open", access, mode))
        return 0

    def MV_CC_SetEnumValue(self, name, value):
        self.calls.append((name, value))
        return 0

    def MV_CC_SetBoolValue(self, name, value):
        self.calls.append((name, value))
        return 0

    def MV_CC_SetFloatValue(self, name, value):
        self.calls.append((name, value))
        return 0

    def MV_CC_GetEnumValue(self, name, value):
        value.nCurValue = {"ExposureAuto": 2, "GainAuto": 1}[name]
        self.calls.append(("get", name))
        return 0

    def MV_CC_GetFloatValue(self, name, value):
        value.fCurValue = {"ExposureTime": 5000.0, "Gain": 0.0}[name]
        self.calls.append(("get", name))
        return 0

    def MV_CC_SetIntValue(self, name, value):
        self.calls.append((name, value))
        return 0

    def MV_CC_GetIntValue(self, name, value):
        value.nCurValue = {"Width": 4024, "Height": 3036}[name]
        self.calls.append(("get", name))
        return 0

    def MV_CC_StartGrabbing(self):
        self.calls.append("start")
        return 0

    def MV_CC_GetImageBuffer(self, frame_out, timeout):
        self.calls.append(("get", timeout))
        pixels = (ctypes.c_ubyte * 4)(1, 2, 3, 4)
        self.buffer = pixels
        frame_out.pBufAddr = ctypes.cast(pixels, ctypes.POINTER(ctypes.c_ubyte))
        frame_out.stFrameInfo.nWidth = 2
        frame_out.stFrameInfo.nHeight = 2
        frame_out.stFrameInfo.nFrameLen = 4
        frame_out.stFrameInfo.enPixelType = 7
        return 0

    def MV_CC_ConvertPixelType(self, convert):
        out = np.ctypeslib.as_array(convert.pDstBuffer, shape=(convert.nDstBufferSize,))
        out[:] = np.arange(convert.nDstBufferSize, dtype=np.uint8)
        self.calls.append("convert")
        return 0

    def MV_CC_FreeImageBuffer(self, frame_out):
        self.calls.append("free")
        return 0

    def MV_CC_StopGrabbing(self):
        self.calls.append("stop")
        return 0

    def MV_CC_CloseDevice(self):
        self.calls.append("close")
        return 0

    def MV_CC_DestroyHandle(self):
        self.calls.append("destroy")
        return 0


def _bindings() -> _Bindings:
    bindings = _Bindings(
        ModuleType("camera"), ModuleType("params"), ModuleType("pixel")
    )
    for module, values in (
        (bindings.camera_module, {"MvCamera": _FakeCamera, "MV_USB_DEVICE": 4,
                                  "MV_GIGE_DEVICE": 1, "MV_ACCESS_Exclusive": 1}),
        (bindings.params_module, {"MV_CC_DEVICE_INFO_LIST": _DeviceList,
                                  "MV_CC_DEVICE_INFO": _DeviceInfo, "MV_FRAME_OUT": _FrameOut,
                                  "MVCC_INTVALUE": _IntValue, "MVCC_ENUMVALUE": _EnumValue,
                                  "MVCC_FLOATVALUE": _FloatValue}),
        (bindings.params_module, {"MV_CC_PIXEL_CONVERT_PARAM": _ConvertParam}),
        (bindings.pixel_module, {"PixelType_Gvsp_BGR8_Packed": 999}),
    ):
        for name, value in values.items():
            setattr(module, name, value)
    return bindings


def test_live_camera_source_converts_and_releases_sdk_frame():
    source = HikRobotCameraSource(_bindings=_bindings())
    frame = source.read()

    assert source.is_live
    assert frame.seq == 0
    assert frame.captured_at > 0
    assert frame.image.shape == (2, 2, 3)
    assert frame.image.dtype == np.uint8
    assert _FakeCamera.last.calls[-2:] == ["convert", "free"]

    source.close()
    source.close()
    assert _FakeCamera.last.calls[-3:] == ["stop", "close", "destroy"]


def test_camera_source_can_request_a_smaller_sensor_image():
    source = HikRobotCameraSource(width=640, height=480, _bindings=_bindings())
    assert ("Width", 640) in _FakeCamera.last.calls
    assert ("Height", 480) in _FakeCamera.last.calls
    source.close()
    assert _FakeCamera.last.calls[-5:] == [
        "stop", ("Width", 4024), ("Height", 3036), "close", "destroy"
    ]


def test_camera_source_requires_width_and_height_together():
    with pytest.raises(ValueError, match="width 和 height"):
        HikRobotCameraSource(width=640, _bindings=_bindings())


def test_manual_brightness_controls_restore_values_and_auto_modes():
    source = HikRobotCameraSource(exposure_us=10000, gain_db=23, _bindings=_bindings())
    calls = _FakeCamera.last.calls
    assert ("ExposureAuto", 0) in calls
    assert ("ExposureTime", 10000.0) in calls
    assert ("GainAuto", 0) in calls
    assert ("Gain", 23.0) in calls
    assert calls[-1] == "start"
    source.close()
    assert calls[-7:] == [
        "stop", ("ExposureTime", 5000.0), ("ExposureAuto", 2),
        ("Gain", 0.0), ("GainAuto", 1), "close", "destroy",
    ]


def test_brightness_controls_restore_on_partial_configuration_failure(monkeypatch):
    setter = _FakeCamera.MV_CC_SetFloatValue

    def fail_requested_gain(self, name, value):
        setter(self, name, value)
        return 0x80000025 if name == "Gain" and value == 23 else 0

    monkeypatch.setattr(_FakeCamera, "MV_CC_SetFloatValue", fail_requested_gain)
    with pytest.raises(CameraError, match="设置 Gain"):
        HikRobotCameraSource(exposure_us=10000, gain_db=23, _bindings=_bindings())
    calls = _FakeCamera.last.calls
    assert "start" not in calls
    assert calls[-6:] == [
        ("ExposureTime", 5000.0), ("ExposureAuto", 2),
        ("Gain", 0.0), ("GainAuto", 1), "close", "destroy",
    ]


@pytest.mark.parametrize("options", [
    {"exposure_us": 0}, {"exposure_us": float("nan")},
    {"exposure_us": float("inf")}, {"gain_db": -1}, {"gain_db": float("nan")},
])
def test_invalid_brightness_controls_are_rejected(options):
    with pytest.raises(ValueError):
        HikRobotCameraSource(**options, _bindings=_bindings())


@pytest.mark.parametrize("name,pattern", [
    ("PixelType_Gvsp_BayerRG8", "RGGB"),
    ("PixelType_Gvsp_BayerGR8", "GRBG"),
    ("PixelType_Gvsp_BayerGB8", "GBRG"),
    ("PixelType_Gvsp_BayerBG8", "BGGR"),
])
def test_bayer_conversion_preserves_colors_and_owns_pixels(name, pattern):
    bindings = _bindings()
    setattr(bindings.pixel_module, name, 12345)
    raw = np.empty((6, 6), dtype=np.uint8)
    for index, color in enumerate(pattern):
        raw[index // 2::2, index % 2::2] = {"R": 201, "G": 100, "B": 50}[color]
    buffer = (ctypes.c_ubyte * raw.size).from_buffer_copy(raw)
    frame = _FrameOut()
    frame.pBufAddr = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
    frame.stFrameInfo.nWidth = frame.stFrameInfo.nHeight = 6
    frame.stFrameInfo.nFrameLen = raw.size
    frame.stFrameInfo.enPixelType = 12345

    image = _copy_bgr8(frame, bindings)
    assert image is not None
    np.testing.assert_array_equal(image[2, 2], [50, 100, 201])
    buffer[:] = [0] * raw.size
    np.testing.assert_array_equal(image[2, 2], [50, 100, 201])


def test_standard_8bit_source_does_not_require_mvs_converter():
    bindings = _bindings()
    bindings.pixel_module.PixelType_Gvsp_Mono8 = 7
    del bindings.params_module.MV_CC_PIXEL_CONVERT_PARAM
    with HikRobotCameraSource(_bindings=bindings) as camera:
        image = camera.read().image
        assert image is not None
        np.testing.assert_array_equal(image[:, :, 0], [[1, 2], [3, 4]])
        np.testing.assert_array_equal(image[:, :, 0], image[:, :, 2])
        assert _FakeCamera.last.calls[-1] == "free"
        assert "convert" not in _FakeCamera.last.calls


def test_bgr_passthrough_copies_sdk_buffer():
    bindings = _bindings()
    buffer = (ctypes.c_ubyte * 3)(50, 100, 201)
    frame = _FrameOut()
    frame.pBufAddr = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))
    frame.stFrameInfo.nWidth = frame.stFrameInfo.nHeight = 1
    frame.stFrameInfo.nFrameLen = 3
    frame.stFrameInfo.enPixelType = 999
    image = _copy_bgr8(frame, bindings)
    buffer[:] = [0, 0, 0]
    np.testing.assert_array_equal(image, [[[50, 100, 201]]])


def test_conversion_rejects_incomplete_frame():
    bindings = _bindings()
    bindings.pixel_module.PixelType_Gvsp_Mono8 = 7
    frame = _FrameOut()
    frame.stFrameInfo.nWidth = frame.stFrameInfo.nHeight = 2
    frame.stFrameInfo.nFrameLen = 3
    frame.stFrameInfo.enPixelType = 7
    with pytest.raises(CameraError, match="图像数据不完整"):
        _copy_bgr8(frame, bindings)
