# -*- coding: utf-8 -*-
"""裁判系统串口帧封装：把业务数据段包成 0xA5 帧头 + CRC 的完整帧。

雷达站往上发 0x0305（雷达→服务器→己方所有选手端，48 字节）走的就是这套
帧格式，和接收方向 0x0A01~0x0A06 完全一样：

    SOF(0xA5) | data_length(2B, 小端) | seq(1B) | CRC8(1B) | cmd_id(2B) | data | CRC16(2B)

CRC8 覆盖 SOF + 长度 + 序号（前 4 字节），CRC16 覆盖从 SOF 到数据结束。
这里直接复用 `coder` 里的查表实现——表只有一份，发送方向不另抄一份。
"""

from __future__ import annotations

from typing import Callable

from .coder import SOF, crc8, crc16

#: 雷达位置数据（雷达 → 服务器 → 选手端小地图），数据段 48 字节。
CMD_RADAR_POSITION = 0x0305


def build_frame(cmd_id: int, payload: bytes, *, seq: int = 0) -> bytes:
    """包一帧裁判系统串口数据，返回完整字节串。"""
    if not 0 <= cmd_id <= 0xFFFF:
        raise ValueError(f"cmd_id 超出 uint16 范围：{cmd_id}")
    if len(payload) > 0xFFFF:
        raise ValueError(f"数据段过长：{len(payload)} 字节")
    head = bytes([SOF]) + len(payload).to_bytes(2, "little") + bytes([seq & 0xFF])
    head += bytes([crc8(head)])
    body = cmd_id.to_bytes(2, "little") + bytes(payload)
    frame = head + body
    return frame + crc16(frame).to_bytes(2, "little")


def build_radar_frame(payload: bytes, *, seq: int = 0) -> bytes:
    """包一帧 0x0305。``payload`` 必须是 48 字节数据段（不含帧头）。"""
    if len(payload) != 48:
        raise ValueError(f"0x0305 数据段必须是 48 字节，收到 {len(payload)}")
    return build_frame(CMD_RADAR_POSITION, payload, seq=seq)


class RadarPositionWriter:
    """把 48 字节数据段包帧、编号、写串口。

    既可以直接当 ``RobotPositionSender`` 的 ``send`` 回调用，
    也可以注入 ``write`` 做离线验证（不装 pyserial、不接硬件）。

    串口不存在时**不抛异常**也不静默吞掉：第一次写失败就记下来，
    之后不再尝试，``failed`` 置位——发送端据此在统计里报出来。现场最怕的
    是"看起来在发、其实设备没连上"。
    """

    def __init__(
        self,
        port: str | None = None,
        *,
        baudrate: int = 115_200,
        write: Callable[[bytes], None] | None = None,
        seq_start: int = 0,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.port = port
        self.baudrate = int(baudrate)
        self._seq = int(seq_start) & 0xFF
        self._write = write
        self._serial = None
        self._logger = logger or (lambda message: None)
        self.frames_written = 0
        self.failed = False
        self.last_error: str | None = None
        if self._write is None:
            self._open(port)

    def _open(self, port: str | None) -> None:
        if not port:
            raise ValueError("没有给串口设备，也没有注入 write 回调")
        try:
            import serial  # pyserial，只在真正要开串口时才需要
        except ImportError as exc:  # pragma: no cover - 环境缺 pyserial 时
            raise RuntimeError("缺少 pyserial：pip install pyserial") from exc
        self._serial = serial.Serial(port, self.baudrate, timeout=0.2)

    def __call__(self, payload: bytes) -> bool:
        """``RobotPositionSender`` 的发送回调：数据段进，串口出。

        必须返回「写成功了没有」：发送端只有拿到 ``True`` 才把它算进
        ``frames_sent``。返回 ``None`` 的旧写法会让写失败被当成发送。
        """
        return self.write(payload) is not None

    def write(self, payload: bytes) -> bytes | None:
        """包帧并写出，返回写出的完整帧（离线验证时方便断言）。"""
        if self.failed:
            return None
        frame = build_radar_frame(payload, seq=self._seq)
        try:
            if self._write is not None:
                self._write(frame)
            else:
                assert self._serial is not None
                self._serial.write(frame)
        except Exception as exc:  # 串口掉线 / 权限 / 拔线
            self.failed = True
            self.last_error = str(exc)
            self._logger(f"0x0305 发送失败，已停止尝试：{exc}")
            return None
        self.frames_written += 1
        self._seq = (self._seq + 1) & 0xFF
        return frame

    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None
