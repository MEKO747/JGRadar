"""IQ 回放框架：从签入流图派生回放变体，且不碰实时路径。

这里的核心承诺只有一条：**实时路径一个字节都不改**。派生的 .grc 写在临时
目录，源块、输出块和变量替换都在内存里做，签入的 `.grc` 前后哈希必须一致。
其余测试盯住派生结果对不对：源块换没换、连线还通不通、变量覆盖生效没生效。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tools.radio_replay import (
    EFFECTIVE_VARIABLES,
    FLOWGRAPH_SOURCE,
    INTERFERE_SOURCE_PORT,
    INFO_SOURCE_PORT,
    REPLAY_SOURCE_NAME,
    SOURCE_BLOCK_NAME,
    TELEMETRY_SOURCE_PORT,
    ReplayError,
    compile_flowgraph,
    decode_bit_files,
    derive_record_flowgraph,
    derive_replay_flowgraph,
    load_flowgraph,
    variable_values,
)

from rmuc.coder import SOF, crc8, crc16
from rmuc.constants import ACCESS_CODES, AIR_HEADER, WaveType

WAVES = (WaveType.INFO.value, WaveType.INTERFERE.value)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def iq_file(tmp_path: Path) -> Path:
    path = tmp_path / "iq.c64"
    np.zeros(2048, dtype=np.complex64).tofile(path)
    return path


def _blocks_by_name(document: dict) -> dict[str, dict]:
    return {block["name"]: block for block in document["blocks"]}


def _port_of(block: dict) -> int:
    return int(str(block["parameters"]["address"]).rsplit(":", 1)[-1])


# ---------------------------------------------------------------------------
# 派生：不碰原文件
# ---------------------------------------------------------------------------
def test_deriving_does_not_touch_the_checked_in_flowgraph(tmp_path, iq_file):
    before = _digest(FLOWGRAPH_SOURCE)
    derive_replay_flowgraph(tmp_path / "replay.grc", iq_file=iq_file)
    assert _digest(FLOWGRAPH_SOURCE) == before, "派生回放变体改到了签入的实时流图"


def test_source_block_is_swapped_and_rewired(tmp_path, iq_file):
    report = derive_replay_flowgraph(tmp_path / "replay.grc", iq_file=iq_file)
    document = load_flowgraph(report.derived_grc)
    blocks = _blocks_by_name(document)

    assert SOURCE_BLOCK_NAME not in blocks
    source = blocks[REPLAY_SOURCE_NAME]
    assert source["id"] == "blocks_file_source"
    assert source["parameters"]["type"] == "complex"
    assert source["parameters"]["repeat"] == "False", "默认单遍回放，避免接缝丢帧"
    assert str(iq_file.resolve()) in source["parameters"]["file"]

    downstream = [c for c in document["connections"] if c[0] == REPLAY_SOURCE_NAME]
    assert len(downstream) == 1, "文件源必须接回原来的下游块"
    assert downstream[0][2] == "rx_raw_baseband_offset"


def test_repeat_can_be_switched_on(tmp_path, iq_file):
    report = derive_replay_flowgraph(tmp_path / "replay.grc", iq_file=iq_file, repeat=True)
    blocks = _blocks_by_name(load_flowgraph(report.derived_grc))
    assert blocks[REPLAY_SOURCE_NAME]["parameters"]["repeat"] == "True"


def test_all_original_dsp_blocks_survive(tmp_path, iq_file):
    """只准换源和输出，DSP 链上的东西一个都不许少。"""
    original = _blocks_by_name(load_flowgraph())
    report = derive_replay_flowgraph(tmp_path / "replay.grc", iq_file=iq_file)
    derived = _blocks_by_name(load_flowgraph(report.derived_grc))
    missing = set(original) - set(derived) - {SOURCE_BLOCK_NAME, "xmlrpc_server"}
    assert not missing, f"派生时丢了这些块：{sorted(missing)}"


def test_xmlrpc_server_is_removed_and_its_connections_dropped(tmp_path, iq_file):
    report = derive_replay_flowgraph(tmp_path / "replay.grc", iq_file=iq_file)
    document = load_flowgraph(report.derived_grc)
    names = {block["name"] for block in document["blocks"]}
    assert "xmlrpc_server" in report.removed_blocks
    assert "xmlrpc_server" not in names
    assert not any("xmlrpc_server" in (c[0], c[2]) for c in document["connections"])


def test_live_mode_rewrites_zmq_ports(tmp_path, iq_file):
    report = derive_replay_flowgraph(
        tmp_path / "replay.grc", iq_file=iq_file, info_port=53803, interfere_port=53804,
        telemetry_port=53805,
    )
    blocks = _blocks_by_name(load_flowgraph(report.derived_grc))
    assert _port_of(blocks["zmq_info_bits_out"]) == 53803
    assert _port_of(blocks["zmq_interfere_bits_out"]) == 53804
    assert _port_of(blocks["rx_power_zmq_pub"]) == 53805
    # 实时链的端口不能被占用，否则回放跑不起来
    for port in (INFO_SOURCE_PORT, INTERFERE_SOURCE_PORT, TELEMETRY_SOURCE_PORT):
        assert port not in report.port_map.values() or report.port_map.get(port) != port


def test_bit_files_mode_swaps_sinks_to_disk_and_keeps_names(tmp_path, iq_file):
    """落盘模式：块名保持不变，这样连线不用动，只换输出方式。"""
    bit_files = {wave: tmp_path / f"bits_{wave}.bin" for wave in WAVES}
    report = derive_replay_flowgraph(
        tmp_path / "replay.grc", iq_file=iq_file, bit_files=bit_files
    )
    document = load_flowgraph(report.derived_grc)
    blocks = _blocks_by_name(document)

    for name in ("zmq_info_bits_out", "zmq_interfere_bits_out"):
        assert blocks[name]["id"] == "blocks_file_sink"
        assert blocks[name]["parameters"]["type"] == "byte"
    assert set(report.replaced_blocks) == {"zmq_info_bits_out", "zmq_interfere_bits_out"}
    assert str(bit_files[WaveType.INFO.value].resolve()) in (
        blocks["zmq_info_bits_out"]["parameters"]["file"]
    )
    # 连线仍然指向原来的块名
    assert any(c[2] == "zmq_info_bits_out" for c in document["connections"])


def test_bit_files_mode_requires_both_channels(tmp_path, iq_file):
    with pytest.raises(ReplayError):
        derive_replay_flowgraph(
            tmp_path / "replay.grc", iq_file=iq_file,
            bit_files={WaveType.INFO.value: tmp_path / "only_info.bin"},
        )


def test_missing_iq_file_is_reported(tmp_path):
    with pytest.raises(ReplayError, match="IQ 文件不存在"):
        derive_replay_flowgraph(tmp_path / "replay.grc", iq_file=tmp_path / "nope.c64")


# ---------------------------------------------------------------------------
# 变量覆盖
# ---------------------------------------------------------------------------
def test_variable_overrides_land_in_the_derived_graph(tmp_path, iq_file):
    report = derive_replay_flowgraph(
        tmp_path / "replay.grc", iq_file=iq_file,
        overrides={"loop_bw": "0.002", "bt": "0.45"},
    )
    assert report.overrides == {"loop_bw": "0.002", "bt": "0.45"}
    derived = variable_values(load_flowgraph(report.derived_grc))
    assert derived["loop_bw"] == "0.002"
    assert derived["bt"] == "0.45"
    # 没覆盖的变量必须保持原样
    assert derived["samp_rate"] == variable_values(load_flowgraph())["samp_rate"]


def test_unknown_variable_is_rejected(tmp_path, iq_file):
    with pytest.raises(ReplayError, match="没有这些变量"):
        derive_replay_flowgraph(
            tmp_path / "replay.grc", iq_file=iq_file, overrides={"no_such_knob": "1"}
        )


def test_gain_is_not_effective_in_replay():
    """硬件增益在文件回放里不可观测——这条要写死在代码里，不能靠文档记住。"""
    assert "loop_bw" in EFFECTIVE_VARIABLES
    assert "bt" in EFFECTIVE_VARIABLES
    assert "rx_gain" not in EFFECTIVE_VARIABLES


# ---------------------------------------------------------------------------
# 离线解码
# ---------------------------------------------------------------------------
def _key_frame_bits() -> bytes:
    """一条正好填满空口帧的干扰波密钥包，展开成 unpacked bits。"""
    payload = bytes([SOF]) + (6).to_bytes(2, "little") + b"\x00"
    payload += bytes([crc8(payload)])
    body = (0x0A06).to_bytes(2, "little") + b"ABC123"
    frame = payload + body
    frame += crc16(frame).to_bytes(2, "little")
    air = ACCESS_CODES[WaveType.INFO] + AIR_HEADER + frame
    assert len(air) == 27
    return bytes((byte >> bit) & 1 for byte in air for bit in range(7, -1, -1))


def test_decode_bit_files_counts_all_three_stages(tmp_path):
    path = tmp_path / "bits_info.bin"
    path.write_bytes(b"\x00" * 50 + _key_frame_bits() + b"\x01" * 50)
    stats = decode_bit_files({WaveType.INFO.value: path}, channels=(WaveType.INFO.value,))
    info = stats[WaveType.INFO.value]
    assert (info.sync, info.frames, info.decoded) == (1, 1, 1)


def test_decode_bit_files_tolerates_missing_file(tmp_path):
    stats = decode_bit_files({WaveType.INFO.value: tmp_path / "absent.bin"},
                             channels=(WaveType.INFO.value,))
    assert stats[WaveType.INFO.value].bits == 0


# ---------------------------------------------------------------------------
# 编译（需要 grcc）
# ---------------------------------------------------------------------------
def test_compile_replay_graph_with_grcc(tmp_path, iq_file):
    pytest.importorskip("gnuradio")
    try:
        compile_flowgraph(FLOWGRAPH_SOURCE, tmp_path / "probe")
    except ReplayError as exc:
        pytest.skip(f"本机 grcc 不可用：{exc}")

    report = derive_replay_flowgraph(tmp_path / "replay.grc", iq_file=iq_file)
    generated = compile_flowgraph(report.derived_grc, tmp_path / "build")
    code = generated.read_text(encoding="utf-8")
    assert "blocks.file_source" in code
    assert "iio.pluto_source" not in code
    assert "xmlrpc" not in code


# ---------------------------------------------------------------------------
# 录制模式：保留实时源，把原始复基带落盘
# ---------------------------------------------------------------------------
def test_record_mode_keeps_the_iio_source(tmp_path):
    report = derive_record_flowgraph(
        tmp_path / "record.grc", record_file=tmp_path / "out.c64", seconds=2.0
    )
    document = load_flowgraph(report.derived_grc)
    blocks = _blocks_by_name(document)

    assert blocks[SOURCE_BLOCK_NAME]["id"] == "iio_pluto_source", "录制不能把实时源换掉"
    assert "xmlrpc_server" not in blocks


def test_record_mode_taps_the_raw_baseband_with_a_known_length(tmp_path):
    report = derive_record_flowgraph(
        tmp_path / "record.grc", record_file=tmp_path / "out.c64", seconds=2.5
    )
    document = load_flowgraph(report.derived_grc)
    blocks = _blocks_by_name(document)

    head = blocks["record_head"]
    assert head["id"] == "blocks_head"
    assert int(head["parameters"]["num_items"]) == 2_500_000, "2500000 个样点 = 2.5 秒"

    sink = blocks["record_file_sink"]
    assert sink["id"] == "blocks_file_sink"
    assert sink["parameters"]["type"] == "complex"
    assert str((tmp_path / "out.c64").resolve()) in sink["parameters"]["file"]

    connections = {(c[0], c[2]) for c in document["connections"]}
    assert (SOURCE_BLOCK_NAME, "record_head") in connections
    assert ("record_head", "rx_raw_baseband_offset") in connections, "录制不能把 DSP 链断掉"
    assert ("record_head", "record_file_sink") in connections


def test_record_mode_honours_a_different_sample_rate(tmp_path):
    report = derive_record_flowgraph(
        tmp_path / "record.grc", record_file=tmp_path / "out.c64",
        seconds=1.0, samp_rate=2_000_000.0,
    )
    document = load_flowgraph(report.derived_grc)
    blocks = _blocks_by_name(document)
    assert int(blocks["record_head"]["parameters"]["num_items"]) == 2_000_000
    assert variable_values(document)["samp_rate"] == "2000000.0"


def test_record_graph_compiles_with_the_real_iio_source(tmp_path):
    pytest.importorskip("gnuradio")
    report = derive_record_flowgraph(
        tmp_path / "record.grc", record_file=tmp_path / "out.c64", seconds=1.0
    )
    try:
        generated = compile_flowgraph(report.derived_grc, tmp_path / "build")
    except ReplayError as exc:
        pytest.skip(f"本机 grcc 不可用：{exc}")
    code = generated.read_text(encoding="utf-8")
    assert "iio." in code and "iio_pluto_source" not in code
    assert "blocks.file_sink" in code
    assert "blocks.head" in code
