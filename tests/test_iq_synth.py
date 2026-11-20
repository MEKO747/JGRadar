"""合成 GFSK 信号：帧结构、调制极性、噪声口径。

这一层不碰 GNU Radio，所以测试在任何环境都能跑。真正「解调链能不能解出来」
由 `tools/radio_replay.py` 的回放验证，这里只保证喂进去的东西是对的：
帧 CRC 合法、比特到频偏的极性没接反、噪声功率就是标称的那一档。
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.iq_synth import (
    DEFAULT_SPS,
    FRAME_BITS,
    SynthSpec,
    add_noise,
    air_frames,
    default_packets,
    modulate,
    normalize_power,
    read_iq,
    serial_frame,
    synthesize,
    unpacked_bits,
    write_iq,
)

from rmuc.coder import AirFrameDecoder, BusinessDecoder, SerialFrameDecoder
from rmuc.constants import ACCESS_CODES, AIR_HEADER, AIR_PAYLOAD_LEN, WaveType


def _discriminate_bits(iq: np.ndarray, *, sps: int = DEFAULT_SPS, skip_symbols: int = 8) -> np.ndarray:
    """用最简单的 FM 鉴频 + 逐符号取中点还原比特。

    和流图里 ``analog.quadrature_demod_cf`` 的数学一致（相邻样点相位差），
    但省掉了符号同步：合成信号的符号边界是已知的。
    """
    phase_delta = np.angle(iq[1:] * np.conj(iq[:-1]))
    centers = np.arange(skip_symbols * sps + sps // 2, len(phase_delta), sps)
    return (phase_delta[centers] > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 帧结构
# ---------------------------------------------------------------------------
def test_serial_frame_passes_the_receivers_own_decoder():
    payload = bytes(range(12))
    frame = serial_frame(0x0A02, payload)
    decoded = SerialFrameDecoder().feed(frame)
    assert len(decoded) == 1
    assert decoded[0].cmd_id == 0x0A02
    assert decoded[0].payload == payload


def test_air_frames_are_the_documented_length():
    stream = b"\x01" * (AIR_PAYLOAD_LEN + 5)
    frames = air_frames(stream)
    assert len(frames) == 2
    for frame in frames:
        assert len(frame) == FRAME_BITS // 8 == 27
        assert frame[:8] == ACCESS_CODES[WaveType.INFO]
        assert frame[8:12] == AIR_HEADER


def test_short_tail_is_zero_padded_not_dropped():
    frames = air_frames(b"\x01" * 3)
    assert len(frames) == 1
    assert frames[0][12:] == b"\x01\x01\x01" + b"\x00" * (AIR_PAYLOAD_LEN - 3)


def test_generated_frames_survive_the_full_decode_chain():
    """端到端：合成的字节流经空口帧 + 串口帧两级解码能回到业务包。"""
    rng = np.random.default_rng(1)
    packets = default_packets(rng, 12)
    stream = b"".join(serial_frame(cmd, payload) for cmd, payload in packets)
    bits = unpacked_bits(b"".join(air_frames(stream)))

    air = AirFrameDecoder(WaveType.INFO)
    serial = SerialFrameDecoder()
    decoded = []
    for payload in air.feed_unpacked_bytes(bytes(bits)):
        for frame in serial.feed(payload):
            packet = BusinessDecoder.decode(frame)
            assert packet is not None, "合成的业务包必须能被业务层认出来"
            decoded.append(packet["cmd"])
    assert len(decoded) == len(packets), "每条串口帧都应该被还原成一条业务包"


# ---------------------------------------------------------------------------
# 调制
# ---------------------------------------------------------------------------
def test_modulation_is_constant_envelope_and_right_length():
    bits = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    iq = modulate(bits, sps=47)
    assert len(iq) == 5 * 47
    assert np.allclose(np.abs(iq), 1.0, atol=1e-6)


def test_bits_round_trip_through_a_discriminator():
    """bit=1 → 正频偏。极性反了整帧会反相，虽然解码器容错，但那是另一回事。"""
    rng = np.random.default_rng(7)
    bits = rng.integers(0, 2, 64).astype(np.uint8)
    iq = modulate(bits, sps=47)
    recovered = _discriminate_bits(iq)
    assert np.array_equal(recovered, bits[8:])


def test_larger_modulation_index_produces_wider_deviation():
    bits = np.zeros(16, dtype=np.uint8)
    bits[::2] = 1
    narrow = np.mean(np.abs(np.angle(modulate(bits, sps=47, mod_index=1.0)[1:]
                                     * np.conj(modulate(bits, sps=47, mod_index=1.0)[:-1]))))
    wide = np.mean(np.abs(np.angle(modulate(bits, sps=47, mod_index=8.0)[1:]
                                   * np.conj(modulate(bits, sps=47, mod_index=8.0)[:-1]))))
    assert wide > narrow * 4


# ---------------------------------------------------------------------------
# 噪声口径
# ---------------------------------------------------------------------------
def test_added_noise_matches_the_requested_snr():
    samples = np.ones(200_000, dtype=np.complex64)
    noisy = add_noise(samples, 10.0, seed=3)
    noise_power = float(np.mean(np.abs(noisy - samples) ** 2))
    # 10 dB → 噪声功率应该约等于信号功率的 1/10
    assert noise_power == pytest.approx(0.1, rel=0.1)


def test_normalize_power_is_used_by_the_sweep_for_real_recordings():
    data = (np.arange(1000, dtype=np.complex64) * (3 + 1j)).astype(np.complex64)
    normalized = normalize_power(data)
    assert float(np.mean(np.abs(normalized) ** 2)) == pytest.approx(1.0, rel=1e-6)


def test_normalize_power_rejects_silence():
    with pytest.raises(ValueError):
        normalize_power(np.zeros(16, dtype=np.complex64))


# ---------------------------------------------------------------------------
# 样本数与真值
# ---------------------------------------------------------------------------
def test_synthesized_length_is_an_integer_number_of_air_frames():
    """样本数正好是整数个空口帧，回放统计的期望值才没有余数偏差。"""
    signal = synthesize(SynthSpec(frames=40, sps=47))
    assert signal.frames == 40
    assert len(signal.samples) == 40 * FRAME_BITS * 47
    # 每条串口帧至少 15 字节，所以串口帧数不可能超过空口帧数
    assert 0 < signal.serial_frames <= signal.frames


def test_seconds_are_rounded_to_whole_frames():
    signal = synthesize(SynthSpec(seconds=1.0, sps=47))
    assert signal.frames == round(1_000_000 / (FRAME_BITS * 47))
    assert signal.seconds == pytest.approx(1.0, abs=0.02)


def test_iq_file_round_trip(tmp_path):
    signal = synthesize(SynthSpec(frames=4, sps=47))
    path = write_iq(tmp_path / "iq.c64", signal.samples)
    loaded = read_iq(path)
    assert np.array_equal(loaded, signal.samples)
    assert loaded.dtype == np.complex64
