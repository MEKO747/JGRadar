# -*- coding: utf-8 -*-
"""裁判系统串口读取：从 0xA5 帧流里挑出雷达侧接口并回调。

雷达机器人和服务器之间是**常规链路**（裁判系统串口）：服务器 1Hz 下发
``0x020C``/``0x020E``，我们用接收方向同一套 `SerialFrameDecoder` 拆帧。

这个线程只做三件事：读字节、拆帧、按命令字分派。业务语义在
`rmuc.referee` 里，ROS 发布在桥接层——三者分开才能各自离线测。
"""

from __future__ import annotations

import threading
from typing import Callable

from .coder import SerialFrameDecoder
from .referee import (
    CMD_MARKING_PROGRESS,
    CMD_RADAR_INFO,
    MarkingProgress,
    RadarInfo,
    decode_marking,
    decode_radar_info,
)

READ_CHUNK = 4096


class RefereeWorker:
    """串口读取线程。

    ``read`` 可注入，用于离线验证整条拆帧/分派链路（不接设备也能测）。
    ``get_own_team`` 返回 ``R``/``B``，因为 ``0x020C`` 的位含义按「对方/己方」
    定义，解码时要按当前视角翻转槽位。
    """

    def __init__(
        self,
        *,
        read: Callable[[], bytes],
        get_own_team: Callable[[], str] = lambda: "R",
        on_marking: Callable[[MarkingProgress], None] | None = None,
        on_radar_info: Callable[[RadarInfo], None] | None = None,
        warn: Callable[[str], None] | None = None,
        close: Callable[[], None] | None = None,
    ) -> None:
        self._read = read
        self._close = close
        self._get_own_team = get_own_team
        self._on_marking = on_marking
        self._on_radar_info = on_radar_info
        self._warn = warn or (lambda _message: None)
        self._decoder = SerialFrameDecoder(max_payload_len=200)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rmuc-referee", daemon=True)
        #: 计数：分派的命令条数与无法识别/长度不符的条数。
        self.marking_frames = 0
        self.radar_info_frames = 0
        self.rejected_frames = 0

    # -- 生命周期 ---------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        if self._close is not None:
            self._close()

    # -- 主循环 -----------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self._read()
            except Exception as exc:  # 串口掉线
                if not self._stop.is_set():
                    self._warn(f"裁判系统串口读取失败：{exc}")
                return
            if chunk:
                self.feed(chunk)

    def feed(self, data: bytes) -> None:
        """喂入一段字节流，拆帧并分派。测试直接调它，不必起线程。"""
        for frame in self._decoder.feed(data):
            if frame.cmd_id == CMD_MARKING_PROGRESS:
                self._dispatch(decode_marking, frame, self._on_marking, own_team=True)
            elif frame.cmd_id == CMD_RADAR_INFO:
                self._dispatch(decode_radar_info, frame, self._on_radar_info, own_team=False)
            else:
                # 裁判系统上还有大量和我们无关的命令（0x0001~0x020B 等），
                # 静默跳过，别让它们污染计数。
                continue

    def _dispatch(self, decoder, frame, callback, *, own_team: bool) -> None:
        try:
            decoded = (
                decoder(frame.payload, own_team=self._get_own_team())
                if own_team else decoder(frame.payload)
            )
        except ValueError as exc:
            self.rejected_frames += 1
            self._warn(f"裁判系统 {frame.cmd_id:#06x} 解不出来：{exc}")
            return
        if decoder is decode_marking:
            self.marking_frames += 1
        else:
            self.radar_info_frames += 1
        if callback is not None:
            callback(decoded)


def open_serial_reader(port: str, baudrate: int = 115_200, *, timeout: float = 0.5):
    """打开裁判系统串口，返回 ``(read, close)`` 两个回调。"""
    import serial

    handle = serial.Serial(port, baudrate, timeout=timeout)
    return (lambda: handle.read(READ_CHUNK)), handle.close
