#!/usr/bin/env python3
"""合成 RMUC 空口 GFSK 信号，用作没有现场信号时的回放基线。

回放框架（`tools/radio_replay.py`）需要一个 IQ 文件才能跑。现场没有真实
信号时，用本模块按接收流图假定的波形造一段：

- 符号率 = ``samp_rate / sps``，即流图里 ``sps`` 的含义；
- 二进制 GFSK，高斯预调制滤波 BT 与流图的 ``bt`` 同义；
- 空口帧 = Access Code(8B) + Header(4B) + Payload(15B)，整帧 27 字节；
- 串口业务帧按字节流切开，每 15 字节塞进一个空口帧，跨帧重组由接收端
  ``SerialFrameDecoder`` 负责，和真实电台一致。

生成的是**模型信号**，不是实测空口。它能回答「解调链对 SNR / 环路带宽 /
BT 怎么变」，不能回答「真实电台实际占多少带宽」——后者只能靠录制。
调制指数 ``mod_index`` 在规则里查不到，默认值见 `DEFAULT_MOD_INDEX` 的注释。

用法::

    python3 tools/iq_synth.py --out out/iq/synth.c64 --seconds 6
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RADIO_PACKAGE = ROOT / "radio" / "radar_parse_em_wave"
if str(RADIO_PACKAGE) not in sys.path:
    sys.path.insert(0, str(RADIO_PACKAGE))

from rmuc.coder import PROTOCOL_2026, PROTOCOL_2027, SOF, crc8, crc16  # noqa: E402
from rmuc.constants import (  # noqa: E402
    ACCESS_CODES,
    AIR_FRAME_LEN,
    AIR_HEADER,
    AIR_PAYLOAD_LEN,
    CMD_FIELD_STATUS,
    CMD_ROBOT_AMMO,
    CMD_ROBOT_BUFF,
    CMD_ROBOT_HP,
    CMD_ROBOT_POSITION,
    WaveType,
)

#: 接收流图的默认采样率与每符号采样数。符号率 = 1e6 / 47 ≈ 21.28 kBd。
DEFAULT_SAMP_RATE = 1_000_000.0
DEFAULT_SPS = 47

#: 高斯预调制滤波 BT，和流图里 `bt` 同名同义。
DEFAULT_BT = 0.35

#: 调制指数 h（= 2×频偏/符号率）。规则文档里查不到官方波形参数，这里取
#: h=8（频偏约 85 kHz），依据是它能落在流图的 270 kHz 中频滤波器内、且
#: 判决器输出不至于贴住 ±1.2 的限幅。绝对误帧率门槛依赖这个假设，
#: 有真实录制后应改用 `--iq` 回放真实信号。
DEFAULT_MOD_INDEX = 8.0

#: 高斯预调制滤波每个方向的符号跨度。
DEFAULT_SPAN_SYMBOLS = 4

FRAME_BITS = AIR_FRAME_LEN * 8


def serial_frame(cmd_id: int, payload: bytes) -> bytes:
    """拼一条 RMUC 串口帧：SOF + 长度 + 序号 + CRC8 + 命令字 + 数据 + CRC16。"""
    head = bytes([SOF]) + len(payload).to_bytes(2, "little") + b"\x00"
    head += bytes([crc8(head)])
    body = cmd_id.to_bytes(2, "little") + payload
    frame = head + body
    return frame + crc16(frame).to_bytes(2, "little")


def default_packets(
    rng: np.random.Generator,
    count: int,
    *,
    protocol: str = PROTOCOL_2027,
) -> list[tuple[int, bytes]]:
    """造一批合法的业务包，循环覆盖各类命令字。

    载荷长度**必须和赛季一致**：2027 的 `0x0A04` 是 12 字节、`0x0A05` 是
    47 字节，拿 2026 的 8/41 字节去跑显式 2027 解码会被判不合法，统计直接
    归零。所以合成信号的版本和回放解码的版本是同一个参数。

    数值按场地边界（2800 × 1500）和常见血量/弹量范围取，保证下游桥接不会
    把坐标当成越界值丢弃；保留位一律写 0。
    """
    if protocol not in (PROTOCOL_2026, PROTOCOL_2027):
        raise ValueError(f"未知协议版本 {protocol!r}；应为 {PROTOCOL_2026}/{PROTOCOL_2027}")

    def positions() -> bytes:
        values = rng.integers(0, 1500, 12)
        if protocol == PROTOCOL_2027:
            # 2027 只下发重装/3/4/哨兵，工程(2,3)与空中(8,9)是保留位
            for index in (2, 3, 8, 9):
                values[index] = 0
        return struct.pack("<12h", *values)

    def hp() -> bytes:
        values = rng.integers(0, 500, 6)
        if protocol == PROTOCOL_2027:
            values[1] = 0                 # 工程变保留位
        values[4] = 0                     # 两代的 5 号槽都未命名
        return struct.pack("<6H", *values)

    def ammo() -> bytes:
        values = rng.integers(0, 350, 5)
        if protocol == PROTOCOL_2027:
            values[3] = 0                 # 空中变保留位
        return struct.pack("<5H", *values)

    def field() -> bytes:
        if protocol == PROTOCOL_2027:
            return struct.pack("<HHII", *rng.integers(0, 900, 2), 0x0ABC, 0x03FF)
        return struct.pack("<HHI", *rng.integers(0, 900, 3))

    def buff() -> bytes:
        if protocol == PROTOCOL_2027:
            payload = bytearray(47)
            for index, offset in enumerate((0, 14, 21, 28)):
                struct.pack_into("<BHBBH", payload, offset,
                                 int(rng.integers(0, 4)), int(rng.integers(0, 40)),
                                 int(rng.integers(0, 3)), int(rng.integers(0, 3)),
                                 int(rng.integers(0, 40)))
            payload[35] = int(rng.integers(1, 7))      # 哨兵姿态
            payload[36] = int(rng.integers(0, 4))      # 重装状态
            payload[38] = int(rng.integers(0, 4))      # 3 号步兵状态
            payload[39] = int(rng.integers(0, 4))      # 4 号步兵状态
            payload[40] = int(rng.integers(0, 4))      # 哨兵状态
            struct.pack_into("<HHH", payload, 41, 800, 600, 250)  # 加固区域 mm
            return bytes(payload)
        # 2026 的 41 字节版本：5×(B H B B H) + 哨兵姿态(1B) + 5 个机器人状态(5B)
        return struct.pack("<" + "BHBBH" * 5 + "B" + "B" * 5,
                           *rng.integers(0, 4, 31).tolist())

    builders = (
        lambda: (CMD_ROBOT_POSITION, positions()),
        lambda: (CMD_ROBOT_HP, hp()),
        lambda: (CMD_ROBOT_AMMO, ammo()),
        lambda: (CMD_FIELD_STATUS, field()),
        lambda: (CMD_ROBOT_BUFF, buff()),
    )
    return [builders[index % len(builders)]() for index in range(count)]


def air_frames(stream: bytes, wave: WaveType = WaveType.INFO) -> list[bytes]:
    """把串口字节流切成 15 字节一段，每段包成一个空口帧。

    最后一段不足 15 字节时补零：接收端只按 Access Code 找帧，补的零字节
    会作为串口帧之间的垃圾被跳过，不影响前一条帧的 CRC。
    """
    access = ACCESS_CODES[wave]
    frames = []
    for offset in range(0, len(stream), AIR_PAYLOAD_LEN):
        chunk = stream[offset:offset + AIR_PAYLOAD_LEN]
        frames.append(access + AIR_HEADER + chunk.ljust(AIR_PAYLOAD_LEN, b"\x00"))
    return frames


def unpacked_bits(data: bytes) -> np.ndarray:
    """按 MSB 优先把字节展开成 0/1 数组，和流图 `binary_slicer_fb` 的输出同格式。"""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def modulate(
    bits: np.ndarray,
    *,
    samp_rate: float = DEFAULT_SAMP_RATE,
    sps: int = DEFAULT_SPS,
    bt: float = DEFAULT_BT,
    mod_index: float = DEFAULT_MOD_INDEX,
    span_symbols: int = DEFAULT_SPAN_SYMBOLS,
    amplitude: float = 1.0,
) -> np.ndarray:
    """把比特调制成 GFSK 复基带。bit=1 对应正频偏，与判决器的极性一致。"""
    if samp_rate <= 0 or sps <= 0:
        raise ValueError("samp_rate 和 sps 必须为正")
    symbol_rate = samp_rate / sps
    deviation = mod_index * symbol_rate / 2.0

    nrz = 2.0 * np.asarray(bits, dtype=np.float64) - 1.0
    baseband = np.repeat(nrz, sps)

    # 高斯预调制滤波：σ 由 BT 和每符号采样数决定，核归一化保持频偏幅度不变。
    sigma = np.sqrt(np.log(2.0)) / (2.0 * np.pi * bt) * sps
    taps_offset = np.arange(-int(span_symbols * sps), int(span_symbols * sps) + 1)
    kernel = np.exp(-taps_offset ** 2 / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    # 用 full 卷积再裁掉两侧各 span 个符号：'same' 在输入短于核长时会返回
    # 核长个样点，长度就错了。
    span = int(span_symbols * sps)
    filtered = np.convolve(baseband, kernel, mode="full")[span:span + len(baseband)]

    phase = 2.0 * np.pi * deviation * np.cumsum(filtered) / samp_rate
    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


@dataclass(frozen=True)
class SynthSpec:
    """一段合成信号的参数。"""

    samp_rate: float = DEFAULT_SAMP_RATE
    sps: int = DEFAULT_SPS
    bt: float = DEFAULT_BT
    mod_index: float = DEFAULT_MOD_INDEX
    seconds: float = 6.0
    wave: WaveType = WaveType.INFO
    #: 业务包按哪个赛季的布局造；回放解码必须用同一个值。
    protocol: str = PROTOCOL_2027
    seed: int = 20261120
    #: 指定空口帧数时优先于 `seconds`，用于让样本数正好覆盖一个测量窗口。
    frames: int | None = None

    @property
    def symbol_rate(self) -> float:
        return self.samp_rate / self.sps

    @property
    def frame_period_symbols(self) -> int:
        """相邻两帧起始符号的间隔。合成信号是帧接帧，没有空闲间隔。"""
        return FRAME_BITS

    def samples_for_seconds(self) -> int:
        return int(round(self.seconds * self.samp_rate))


@dataclass(frozen=True)
class SynthSignal:
    """合成结果：IQ 样本加上可供报告引用的参数。"""

    samples: np.ndarray
    spec: SynthSpec
    frames: int
    frame_bytes: int
    #: 完整落在 IQ 里的串口帧数，即回放时理论上能解出的业务包上限。
    serial_frames: int = 0

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.spec.samp_rate

    @property
    def serial_frames_per_air_frame(self) -> float:
        """一个空口帧平均承载多少条串口帧，用于把期望值换算到消耗的比特上。"""
        return self.serial_frames / self.frames if self.frames else 0.0

    @property
    def metadata(self) -> dict:
        return {
            "samp_rate": self.spec.samp_rate,
            "sps": self.spec.sps,
            "bt": self.spec.bt,
            "mod_index": self.spec.mod_index,
            "symbol_rate": self.spec.symbol_rate,
            "deviation_hz": self.spec.mod_index * self.spec.symbol_rate / 2.0,
            "seconds": self.seconds,
            "samples": int(len(self.samples)),
            "frames": self.frames,
            "serial_frames": self.serial_frames,
            "wave": self.spec.wave.value,
            "protocol": self.spec.protocol,
            "frame_period_symbols": self.spec.frame_period_symbols,
        }


def synthesize(spec: SynthSpec | None = None) -> SynthSignal:
    """按参数造一段连续空口帧的 IQ。

    样本数正好是空口帧周期的整数倍，这样「消耗的比特 / 216」就是消耗的
    空口帧数，回放统计的期望值没有余数偏差。
    """
    spec = spec or SynthSpec()
    rng = np.random.default_rng(spec.seed)
    period_samples = FRAME_BITS * spec.sps
    if spec.frames:
        frame_count = max(1, int(spec.frames))
    else:
        frame_count = max(1, int(round(spec.samples_for_seconds() / period_samples)))
    usable = frame_count * AIR_PAYLOAD_LEN

    packets = default_packets(rng, frame_count * 2 + 2, protocol=spec.protocol)
    stream = bytearray()
    complete = 0
    for cmd_id, payload in packets:
        stream += serial_frame(cmd_id, payload)
        if len(stream) <= usable:
            complete += 1
        if len(stream) >= usable:
            break
    stream = bytes(stream[:usable])

    frames = air_frames(stream, spec.wave)
    bits = unpacked_bits(b"".join(frames))
    samples = modulate(
        bits,
        samp_rate=spec.samp_rate,
        sps=spec.sps,
        bt=spec.bt,
        mod_index=spec.mod_index,
    )
    return SynthSignal(
        samples=samples,
        spec=spec,
        frames=len(frames),
        frame_bytes=len(frames) * AIR_FRAME_LEN,
        serial_frames=complete,
    )


def normalize_power(samples: np.ndarray, *, target_power: float = 1.0) -> np.ndarray:
    """把 IQ 归一到指定平均功率，消掉录制设备的绝对电平差异。"""
    data = np.asarray(samples, dtype=np.complex64)
    power = float(np.mean(np.abs(data) ** 2)) if len(data) else 0.0
    if power <= 0.0:
        raise ValueError("IQ 全是零，无法归一化")
    return (data * np.sqrt(target_power / power)).astype(np.complex64)


def add_noise(
    samples: np.ndarray,
    snr_db: float,
    *,
    seed: int = 0,
) -> np.ndarray:
    """按「信号平均功率 / 附加噪声功率」加复高斯白噪声。

    σ² 在 I/Q 两路均分，与 GNU Radio 里 `noise_source_c(amp)` 的复噪声口径一致。
    同一个 seed 下各档噪声用同一组随机数缩放，因此不同 SNR 点之间只差噪声幅度。
    """
    data = np.asarray(samples, dtype=np.complex64)
    power = float(np.mean(np.abs(data) ** 2)) if len(data) else 0.0
    if power <= 0.0:
        return data.copy()
    sigma_squared = power / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(data)) + 1j * rng.standard_normal(len(data))
    noise *= np.sqrt(sigma_squared / 2.0)
    return (data + noise).astype(np.complex64)


def write_iq(path: str | Path, samples: np.ndarray) -> Path:
    """写成 GNU Radio `blocks_file_source` 能直接读的 complex64 交错二进制。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(samples, dtype=np.complex64).tofile(path)
    return path


def read_iq(path: str | Path, max_samples: int | None = None) -> np.ndarray:
    """读取 complex64 IQ 文件。"""
    data = np.fromfile(Path(path), dtype=np.complex64)
    if max_samples is not None:
        data = data[:max_samples]
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="输出 IQ 文件（complex64）")
    parser.add_argument("--seconds", type=float, default=SynthSpec.seconds, help="时长，默认 6 秒")
    parser.add_argument("--samp-rate", type=float, default=DEFAULT_SAMP_RATE)
    parser.add_argument("--sps", type=int, default=DEFAULT_SPS)
    parser.add_argument("--bt", type=float, default=DEFAULT_BT)
    parser.add_argument("--mod-index", type=float, default=DEFAULT_MOD_INDEX,
                        help=f"调制指数 h，默认 {DEFAULT_MOD_INDEX}")
    parser.add_argument("--wave", choices=("info", "interfere"), default="info")
    parser.add_argument("--protocol", choices=(PROTOCOL_2026, PROTOCOL_2027),
                        default=PROTOCOL_2027,
                        help=f"业务包按哪个赛季的布局造，默认 {PROTOCOL_2027}")
    args = parser.parse_args(argv)

    spec = SynthSpec(
        samp_rate=args.samp_rate,
        sps=args.sps,
        bt=args.bt,
        mod_index=args.mod_index,
        seconds=args.seconds,
        wave=WaveType(args.wave),
        protocol=args.protocol,
    )
    signal = synthesize(spec)
    path = write_iq(args.out, signal.samples)
    print(json.dumps(signal.metadata, ensure_ascii=False, indent=2))
    print(f"已写入 {path}（{path.stat().st_size / 1e6:.1f} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
