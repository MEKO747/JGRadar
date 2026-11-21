# -*- coding: utf-8 -*-
"""裁判系统串口的**单点所有权**：一个 fd，读线程 + 写方法。

联合启动时裁判串口只能有一个进程持有——两个进程抢同一个 tty 在物理上就不成立
（读走字节、写没有同步，谁先打开谁先说了算）。所以所有权归 **runner**：

- 读：``0x020C`` / ``0x020E`` 由这里的读线程拆帧并回调
- 写：``0x0305`` 通过 :meth:`write` 走**同一个**串口句柄

接收节点在联合启动下完全不碰串口，只订阅 ``/rmuc/demod_config``。
单独运行接收节点时它自己开串口，逻辑在 ``rmuc_ros2_node.py`` 里，和这里互不影响。

拔线不能崩：写失败置 ``write_failed`` 并返回 ``False``，读线程收到异常后
只报一次警告就退出——主循环照常跑，统计里看得见。
"""

from __future__ import annotations

import threading
from typing import Callable

from .referee import MarkingProgress, RadarInfo
from .referee_worker import READ_CHUNK, RefereeWorker


class RefereeSerialLink:
    """裁判串口：一个句柄、一条读线程、一个写方法。

    ``port`` 打开失败会**抛异常**（启动阶段就该看得见），运行中掉线则退化。
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115_200,
        *,
        on_marking: Callable[[MarkingProgress], None] | None = None,
        on_radar_info: Callable[[RadarInfo], None] | None = None,
        get_own_team: Callable[[], str] = lambda: "R",
        warn: Callable[[str], None] | None = None,
        serial_factory: Callable[..., object] | None = None,
        timeout: float = 0.2,
    ) -> None:
        if serial_factory is None:
            import serial  # pyserial：只在真接串口时需要

            serial_factory = serial.Serial
        self.port = str(port)
        self.baudrate = int(baudrate)
        self._handle = serial_factory(self.port, self.baudrate, timeout=timeout,
                                      write_timeout=timeout)
        self.write_frames = 0
        self.write_errors = 0
        self.write_failed = False
        self.last_write_error: str | None = None
        self._lock = threading.Lock()
        self.worker = RefereeWorker(
            read=self._read,
            get_own_team=get_own_team,
            on_marking=on_marking,
            on_radar_info=on_radar_info,
            warn=warn or (lambda _message: None),
            close=self.close,
        )

    # -- 写 ---------------------------------------------------------------
    def write(self, frame: bytes) -> bool:
        """把一帧写到串口。**失败返回 False**，调用方据此不把它算成发送。"""
        with self._lock:
            if self.write_failed:
                return False
            try:
                written = self._handle.write(frame)
            except Exception as exc:  # noqa: BLE001 —— 拔线/权限/设备消失
                self.write_failed = True
                self.write_errors += 1
                self.last_write_error = f"{type(exc).__name__}: {exc}"
                return False
            # pyserial 正常返回写入字节数；0 也算没写出去（超时）。
            if written is not None and int(written) != len(frame):
                self.write_errors += 1
                self.last_write_error = f"只写入 {written}/{len(frame)} 字节"
                return False
            self.write_frames += 1
            return True

    # -- 读 ---------------------------------------------------------------
    def _read(self) -> bytes:
        return self._handle.read(READ_CHUNK)

    def start(self) -> None:
        self.worker.start()

    def stop(self) -> None:
        self.worker.stop()

    def close(self) -> None:
        with self._lock:
            try:
                self._handle.close()
            except Exception:  # noqa: BLE001 —— 关一个已经掉的串口不该再抛
                pass

    # -- 统计 -------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "port": self.port,
            "baudrate": self.baudrate,
            "marking_frames": self.worker.marking_frames,
            "radar_info_frames": self.worker.radar_info_frames,
            "rejected_frames": self.worker.rejected_frames,
            "write_frames": self.write_frames,
            "write_errors": self.write_errors,
            "write_failed": self.write_failed,
            "last_write_error": self.last_write_error,
        }
