"""HIKROBOT USB3 camera source backed by the vendor MVS Python SDK.

The SDK is shipped with MVS and is deliberately loaded only when a camera
source is requested. Offline/replay use therefore never depends on MVS.
"""

from __future__ import annotations

import ctypes
import importlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np

from jg_radar.io.frame_source import Frame

__all__ = ["HikRobotCameraSource", "CameraError"]


class CameraError(RuntimeError):
    """MVS SDK unavailable or camera acquisition failed."""


@dataclass(frozen=True)
class _Bindings:
    camera_module: ModuleType
    params_module: ModuleType
    pixel_module: ModuleType

    def value(self, name: str, default=None):
        for module in (
            self.camera_module,
            self.params_module,
            self.pixel_module,
        ):
            if hasattr(module, name):
                return getattr(module, name)
        return default


def _sdk_roots() -> tuple[Path, ...]:
    return tuple(
        Path(p)
        for p in (
            "/opt/MVS",
            "/usr/local/MVS",
            "/usr/local/Hikrobot/MVS",
        )
    )


def _load_bindings() -> _Bindings:
    """Find and import the Python modules installed alongside MVS."""
    sdk_path_candidates = (
        Path("Samples/64/Python/MvImport"),
        Path("Samples/64/Python"),
        Path("Development/Samples/64/Python/MvImport"),
    )
    sdk_root = next((root for root in _sdk_roots() if any(
        (root / candidate).is_dir() for candidate in sdk_path_candidates
    )), None)
    if sdk_root is None:
        raise CameraError(
            "找不到海康 MVS Python SDK。请先安装海康官方 MVS Linux x86_64，"
            "并确认安装目录为 /opt/MVS。"
        )

    # MvCameraControl_class loads libMvCameraControl.so using this variable.
    # Installer profiles export it for interactive shells; set a default here
    # so this source also works when launched from an IDE or service.
    import os

    os.environ.setdefault("MVCAM_SDK_PATH", str(sdk_root))
    os.environ.setdefault("MVCAM_COMMON_RUNENV", str(sdk_root / "lib"))
    module_paths = [
        str(sdk_root / candidate)
        for candidate in sdk_path_candidates
        if (sdk_root / candidate).is_dir()
    ]
    for path in reversed(module_paths):
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        camera_module = importlib.import_module("MvCameraControl_class")
        params_module = importlib.import_module("CameraParams_header")
        pixel_module = importlib.import_module("PixelType_header")
    except (ImportError, OSError) as exc:
        raise CameraError(
            "找不到海康 MVS Python SDK。请先安装海康官方 MVS Linux x86_64，"
            "并确认安装目录为 /opt/MVS。"
        ) from exc
    return _Bindings(camera_module, params_module, pixel_module)


def _check(code, operation: str) -> None:
    value = int(code)
    if value != 0:
        detail = f"{operation} 失败，MVS 错误码 0x{value & 0xFFFFFFFF:08X}"
        if operation == "读取相机图像" and value == 0x80000007:
            detail += "（无完整图像数据；请检查 USB 链路及相机采集状态）"
        if operation == "转换相机图像格式" and value == 0x8000000C:
            detail += "（MVS 转换库加载失败；请从已加载 MVS 环境的终端启动）"
        raise CameraError(detail)


def _copy_bgr8(frame_out, bindings: _Bindings) -> np.ndarray | None:
    """Copy standard 8-bit images without MVS's optional conversion library."""
    formats = (
        ("PixelType_Gvsp_Mono8", 1, "COLOR_GRAY2BGR"),
        ("PixelType_Gvsp_RGB8_Packed", 3, "COLOR_RGB2BGR"),
        ("PixelType_Gvsp_BGR8_Packed", 3, None),
        # Four-letter patterns avoid OpenCV's ambiguous legacy Bayer aliases.
        ("PixelType_Gvsp_BayerRG8", 1, "COLOR_BayerRGGB2BGR"),
        ("PixelType_Gvsp_BayerGR8", 1, "COLOR_BayerGRBG2BGR"),
        ("PixelType_Gvsp_BayerGB8", 1, "COLOR_BayerGBRG2BGR"),
        ("PixelType_Gvsp_BayerBG8", 1, "COLOR_BayerBGGR2BGR"),
    )
    info = frame_out.stFrameInfo
    for name, channels, conversion in formats:
        if bindings.value(name) != int(info.enPixelType):
            continue
        width, height = int(info.nWidth), int(info.nHeight)
        size = width * height * channels
        if int(info.nFrameLen) < size:
            raise CameraError(f"相机图像数据不完整：需要 {size} 字节，收到 {info.nFrameLen} 字节")
        raw = np.ctypeslib.as_array(
            ctypes.cast(frame_out.pBufAddr, ctypes.POINTER(ctypes.c_ubyte)),
            shape=(size,),
        )
        image = raw.reshape((height, width) if channels == 1 else (height, width, channels))
        if conversion is None:
            return image.copy()
        import cv2

        return cv2.cvtColor(image, getattr(cv2, conversion))
    return None


def _device_string(raw) -> str:
    if isinstance(raw, bytes):
        data = raw
    else:
        try:
            data = bytes(raw)
        except (TypeError, ValueError):
            return str(raw)
    return data.split(b"\0", 1)[0].decode("utf-8", errors="replace").strip()


def _device_metadata(device_info, bindings: _Bindings) -> tuple[str, str]:
    usb_type = int(bindings.value("MV_USB_DEVICE", 0x00000004))
    if int(device_info.nTLayerType) & usb_type:
        info = device_info.SpecialInfo.stUsb3VInfo
    else:
        info = device_info.SpecialInfo.stGigEInfo
    return _device_string(info.chModelName), _device_string(info.chSerialNumber)


class HikRobotCameraSource:
    """Live HIKROBOT source yielding BGR8 OpenCV images and monotonic stamps.

    Args:
        camera_index: Zero-based index in the MVS USB/GigE enumeration.
        serial_number: If set, choose the matching camera serial instead.
        read_timeout_ms: Maximum time to wait for a frame before raising.
    """

    is_live = True

    def __init__(
        self,
        *,
        camera_index: int = 0,
        serial_number: str = "",
        width: int = 0,
        height: int = 0,
        exposure_us: float | None = None,
        gain_db: float | None = None,
        read_timeout_ms: int = 5000,
        _bindings: _Bindings | None = None,
    ) -> None:
        if camera_index < 0:
            raise ValueError("camera_index 必须大于等于 0")
        if width < 0 or height < 0 or bool(width) != bool(height):
            raise ValueError("width 和 height 必须同时为正数，或同时为 0 以使用相机默认分辨率")
        if exposure_us is not None and (not math.isfinite(exposure_us) or exposure_us <= 0):
            raise ValueError("exposure_us 必须为有限正数")
        if gain_db is not None and (not math.isfinite(gain_db) or gain_db < 0):
            raise ValueError("gain_db 必须为有限非负数")
        self.camera_index = int(camera_index)
        self.serial_number = str(serial_number).strip()
        self.width = int(width)
        self.height = int(height)
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.read_timeout_ms = int(read_timeout_ms)
        self._bindings = _bindings or _load_bindings()
        self._camera = None
        self._initialized = False
        self._created = False
        self._opened = False
        self._grabbing = False
        self._original_size: tuple[int, int] | None = None
        self._original_controls: dict[str, tuple[str, int | float]] = {}
        self._seq = 0
        try:
            self._open()
        except Exception:
            self.close()
            raise

    def _open(self) -> None:
        bindings = self._bindings
        camera_type = bindings.value("MvCamera")
        device_list_type = bindings.value("MV_CC_DEVICE_INFO_LIST")
        device_info_type = bindings.value("MV_CC_DEVICE_INFO")
        if camera_type is None or device_list_type is None or device_info_type is None:
            raise CameraError("MVS Python SDK 缺少相机枚举类型，无法打开相机")

        _check(camera_type.MV_CC_Initialize(), "初始化 MVS SDK")
        self._initialized = True
        camera = camera_type()
        self._camera = camera
        devices = device_list_type()
        device_mask = int(bindings.value("MV_GIGE_DEVICE", 0x00000001)) | int(
            bindings.value("MV_USB_DEVICE", 0x00000004)
        )
        _check(camera.MV_CC_EnumDevices(device_mask, devices), "枚举相机")
        count = int(devices.nDeviceNum)
        if count == 0:
            raise CameraError("MVS 没有发现相机；请检查 USB3 连接，并关闭正在占用相机的程序")

        candidates = []
        for index in range(count):
            info = ctypes.cast(devices.pDeviceInfo[index], ctypes.POINTER(device_info_type)).contents
            model, serial = _device_metadata(info, bindings)
            candidates.append((info, model, serial))

        if self.serial_number:
            selected = next((item for item in candidates if item[2] == self.serial_number), None)
            if selected is None:
                available = ", ".join(f"{model} ({serial})" for _, model, serial in candidates)
                raise CameraError(
                    f"没有找到序列号为 {self.serial_number!r} 的相机；已发现：{available}"
                )
        elif self.camera_index < count:
            selected = candidates[self.camera_index]
        else:
            raise CameraError(f"camera_index={self.camera_index} 超出范围，共发现 {count} 台相机")

        info, model, serial = selected
        _check(camera.MV_CC_CreateHandle(info), f"创建 {model} 相机句柄")
        self._created = True
        access = int(bindings.value("MV_ACCESS_Exclusive", 1))
        _check(camera.MV_CC_OpenDevice(access, 0), f"打开 {model} ({serial})")
        self._opened = True

        if self.width and self.height:
            int_value_type = bindings.value("MVCC_INTVALUE")
            get_int_value = getattr(camera, "MV_CC_GetIntValue", None)
            if int_value_type is None or get_int_value is None:
                raise CameraError("MVS Python SDK 缺少读取相机分辨率的接口")
            original_width = int_value_type()
            original_height = int_value_type()
            _check(get_int_value("Width", original_width), "读取相机原始宽度")
            _check(get_int_value("Height", original_height), "读取相机原始高度")
            self._original_size = (
                int(original_width.nCurValue),
                int(original_height.nCurValue),
            )
            _check(camera.MV_CC_SetIntValue("Width", self.width), "设置相机图像宽度")
            _check(camera.MV_CC_SetIntValue("Height", self.height), "设置相机图像高度")

        # Free-running continuous mode lets a USB camera produce frames without
        # an external trigger cable.
        _check(camera.MV_CC_SetEnumValue("AcquisitionMode", 2), "设置连续采集")
        _check(camera.MV_CC_SetEnumValue("TriggerMode", 0), "关闭外部触发")

        for key, auto_key, requested in (
            ("ExposureTime", "ExposureAuto", self.exposure_us),
            ("Gain", "GainAuto", self.gain_db),
        ):
            if requested is None:
                continue
            enum_type = bindings.value("MVCC_ENUMVALUE")
            float_type = bindings.value("MVCC_FLOATVALUE")
            if enum_type is None or float_type is None:
                raise CameraError("MVS Python SDK 缺少曝光和增益参数类型")
            auto_value = enum_type()
            float_value = float_type()
            _check(camera.MV_CC_GetEnumValue(auto_key, auto_value), f"读取 {auto_key}")
            _check(camera.MV_CC_GetFloatValue(key, float_value), f"读取 {key}")
            self._original_controls[key] = ("float", float(float_value.fCurValue))
            self._original_controls[auto_key] = ("enum", int(auto_value.nCurValue))
            _check(camera.MV_CC_SetEnumValue(auto_key, 0), f"关闭 {auto_key}")
            _check(camera.MV_CC_SetFloatValue(key, float(requested)), f"设置 {key}")

        _check(camera.MV_CC_StartGrabbing(), "开始采集")
        self._grabbing = True

    def read(self) -> Frame:
        """Return one copied BGR frame; camera read failures never look like EOF."""
        if self._camera is None or not self._grabbing:
            raise CameraError("相机未在采集状态")
        frame_out_type = self._bindings.value("MV_FRAME_OUT")
        if frame_out_type is None:
            raise CameraError("MVS Python SDK 缺少取流类型")

        frame_out = frame_out_type()
        _check(
            self._camera.MV_CC_GetImageBuffer(frame_out, self.read_timeout_ms),
            "读取相机图像",
        )
        captured_at = time.monotonic()
        try:
            info = frame_out.stFrameInfo
            width, height = int(info.nWidth), int(info.nHeight)
            if width <= 0 or height <= 0:
                raise CameraError(f"相机返回了非法图像尺寸 {width}×{height}")
            image = _copy_bgr8(frame_out, self._bindings)
            if image is None:
                convert_type = self._bindings.value("MV_CC_PIXEL_CONVERT_PARAM")
                bgr_type = self._bindings.value("PixelType_Gvsp_BGR8_Packed")
                if convert_type is None or bgr_type is None:
                    raise CameraError("MVS Python SDK 缺少像素格式转换类型")
                buffer_size = width * height * 3
                destination = (ctypes.c_ubyte * buffer_size)()
                convert = convert_type()
                convert.nWidth = width
                convert.nHeight = height
                convert.pSrcData = ctypes.cast(frame_out.pBufAddr, ctypes.POINTER(ctypes.c_ubyte))
                convert.nSrcDataLen = int(info.nFrameLen)
                convert.enSrcPixelType = info.enPixelType
                convert.enDstPixelType = bgr_type
                convert.pDstBuffer = ctypes.cast(destination, ctypes.POINTER(ctypes.c_ubyte))
                convert.nDstBufferSize = buffer_size
                _check(self._camera.MV_CC_ConvertPixelType(convert), "转换相机图像格式")
                image = np.ctypeslib.as_array(destination).reshape(height, width, 3).copy()
        finally:
            self._camera.MV_CC_FreeImageBuffer(frame_out)

        result = Frame(seq=self._seq, captured_at=captured_at, image=image)
        self._seq += 1
        return result

    def close(self) -> None:
        """Stop acquisition and release the device handle; safe to call twice."""
        camera = self._camera
        if camera is None:
            return
        try:
            if self._grabbing:
                camera.MV_CC_StopGrabbing()
                self._grabbing = False
        finally:
            try:
                if self._opened:
                    try:
                        for key, (kind, value) in self._original_controls.items():
                            setter = (camera.MV_CC_SetFloatValue if kind == "float"
                                      else camera.MV_CC_SetEnumValue)
                            setter(key, value)
                        if self._original_size is not None:
                            camera.MV_CC_SetIntValue("Width", self._original_size[0])
                            camera.MV_CC_SetIntValue("Height", self._original_size[1])
                    finally:
                        camera.MV_CC_CloseDevice()
                        self._opened = False
            finally:
                try:
                    if self._created:
                        camera.MV_CC_DestroyHandle()
                        self._created = False
                finally:
                    try:
                        if self._initialized:
                            self._bindings.value("MvCamera").MV_CC_Finalize()
                            self._initialized = False
                    finally:
                        self._camera = None

    def __enter__(self) -> "HikRobotCameraSource":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
