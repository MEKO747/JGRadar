#!/usr/bin/env python3
"""把接收流图改成 IQ 文件回放，跑完统计三级解码的通过率。

仓库里签入的流图只有 PlutoSDR 实时源（`iio_pluto_source`），没法对同一段
信号反复跑不同参数。本模块不复制一份流图——复制出来的第二条 DSP 链迟早
和真实链漂移——而是**读原 `.grc`、只替换源块和输出块**，把变体写到临时
目录再用 `grcc` 编译。原文件一个字节都不改，所以实时路径的行为天然不变。

派生出来的回放链和实时链共用同一份 DSP 与解码器：

    file_source → rx_raw_baseband_offset → … → 比特流（文件或 ZMQ）

两种取比特方式：

- **落盘（默认）**：把 `zeromq_pub_sink` 换成 `blocks_file_sink`，流图满速
  跑完文件，再离线解码。不受解码速度限制，也没有尾部丢消息的问题，扫描
  结果可复现。
- **旁听（`--live`）**：保留 ZMQ 输出，边跑边订阅解码。适合交互式观察，
  但流图比 Python 解码快，收满预定比特数会在流图退出时丢掉尾巴。

用法::

    source ~/.jg_env.zsh
    python3 tools/radio_replay.py --iq out/iq/synth.c64
    python3 tools/radio_replay.py --iq rec.c64 --bits 400000 --set loop_bw=0.002
    python3 tools/radio_replay.py --iq rec.c64 --live --info-port 53203
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
RADIO_PACKAGE = ROOT / "radio" / "radar_parse_em_wave"
for _path in (ROOT, RADIO_PACKAGE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rmuc.constants import WaveType  # noqa: E402
from tools.radio_probe import ChannelStats  # noqa: E402

#: 签入仓库的流图，是唯一事实来源。
FLOWGRAPH_SOURCE = RADIO_PACKAGE / "grc" / "RMUC2026_NanoSDR_RX.grc"

#: 实时流图里 PlutoSDR 源块的实例名。
SOURCE_BLOCK_NAME = "nanosdr_source"

#: 派生流图里文件源的实例名。
REPLAY_SOURCE_NAME = "iq_file_source"

#: 回放默认端口。刻意和实时链的 53103/53104/53105 错开，方便和正在运行的
#: 接收节点并存。
DEFAULT_INFO_PORT = 53203
DEFAULT_INTERFERE_PORT = 53204
DEFAULT_TELEMETRY_PORT = 53205

#: 签入流图里两路比特流的 ZMQ 端口。
INFO_SOURCE_PORT = 53103
INTERFERE_SOURCE_PORT = 53104
TELEMETRY_SOURCE_PORT = 53105

#: 只有这些变量在回放里真的会改变 DSP 行为。`rx_gain` 不在其中：硬件增益
#: 在文件回放里不可观测，它只决定录制时的输入 SNR。
EFFECTIVE_VARIABLES = frozenset({
    "samp_rate", "sps", "loop_bw", "bt", "bw",
    "rx_info_bw", "rx_int_bw", "rx_info_fir_taps", "rx_int_fir_taps",
    "leak_beta_re", "leak_beta_im",
    "rx_center_freq", "rx_info_channel_freq", "rx_interfere_channel_freq",
})

WAVES = (WaveType.INFO.value, WaveType.INTERFERE.value)


class ReplayError(RuntimeError):
    """回放链路在派生、编译或运行阶段失败。"""


@dataclass(frozen=True)
class DeriveReport:
    """派生流图做了哪些改动，供日志和测试核对。"""

    source_grc: Path
    derived_grc: Path
    removed_blocks: tuple[str, ...]
    replaced_blocks: tuple[str, ...]
    port_map: dict[int, int]
    overrides: dict[str, str]


def load_flowgraph(path: str | Path = FLOWGRAPH_SOURCE) -> dict:
    """读取 GRC 流图（GNU Radio 3.10 起 .grc 是 YAML）。"""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def variable_values(document: dict) -> dict[str, str]:
    """取流图里所有 `variable` 块的取值，键是变量名。"""
    return {
        block["name"]: str(block["parameters"].get("value"))
        for block in document["blocks"]
        if block.get("id") == "variable"
    }


def _sink_port(block: dict) -> int | None:
    """取 ZMQ 输出块绑定的端口号。"""
    address = str(block.get("parameters", {}).get("address", ""))
    tail = address.rsplit(":", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _rewrite_zmq_ports(document: dict, port_map: dict[int, int]) -> None:
    for block in document["blocks"]:
        if block.get("id") != "zeromq_pub_sink":
            continue
        port = _sink_port(block)
        if port in port_map:
            address = str(block["parameters"]["address"])
            block["parameters"]["address"] = address[: -len(str(port))] + str(port_map[port])


def _drop_block(document: dict, name: str) -> None:
    """删掉一个块以及所有连到它的连线。"""
    document["blocks"] = [block for block in document["blocks"] if block["name"] != name]
    document["connections"] = [
        connection for connection in document["connections"]
        if connection[0] != name and connection[2] != name
    ]


def derive_replay_flowgraph(
    dest: str | Path,
    *,
    iq_file: str | Path,
    source_grc: str | Path = FLOWGRAPH_SOURCE,
    overrides: dict[str, str] | None = None,
    info_port: int = DEFAULT_INFO_PORT,
    interfere_port: int = DEFAULT_INTERFERE_PORT,
    telemetry_port: int = DEFAULT_TELEMETRY_PORT,
    repeat: bool = False,
    bit_files: dict[str, str | Path] | None = None,
) -> DeriveReport:
    """生成只换了源（和输出）的流图变体，写到 `dest`。

    除了源块、输出块、三个 ZMQ 端口和显式覆盖的变量，其余块和连线与原图
    完全一致：DSP 链、限幅、AGC、SIC、判决器全部沿用。

    `bit_files` 给出时把两路比特流改成文件输出（键为 `info` / `interfere`），
    并删掉供增益调节用的功率遥测输出。
    """
    source_grc = Path(source_grc)
    dest = Path(dest)
    document = load_flowgraph(source_grc)

    blocks = document["blocks"]
    names = {block["name"] for block in blocks}
    if SOURCE_BLOCK_NAME not in names:
        raise ReplayError(f"流图里没有 {SOURCE_BLOCK_NAME} 源块：{source_grc}")

    iq_path = Path(iq_file).resolve()
    if not iq_path.is_file():
        raise ReplayError(f"IQ 文件不存在：{iq_path}")

    for block in blocks:
        if block["name"] != SOURCE_BLOCK_NAME:
            continue
        block["id"] = "blocks_file_source"
        block["name"] = REPLAY_SOURCE_NAME
        block["parameters"] = {
            "affinity": "",
            "alias": "",
            "begin_tag": "pmt.PMT_NIL",
            "comment": "IQ file replay source, derived from the iio path",
            "file": json.dumps(str(iq_path)),
            "maxoutbuf": "0",
            "minoutbuf": "0",
            "offset": "0",
            "repeat": "True" if repeat else "False",
            "type": "complex",
            "vlen": "1",
        }

    for connection in document["connections"]:
        if connection[0] == SOURCE_BLOCK_NAME:
            connection[0] = REPLAY_SOURCE_NAME

    # 实时调参用的 XMLRPC 服务在回放里没有用途，留着只会抢占 53106。
    removed = tuple(block["name"] for block in blocks if block.get("id") == "xmlrpc_server")
    for name in removed:
        _drop_block(document, name)

    replaced: list[str] = []
    port_map = {INFO_SOURCE_PORT: info_port, INTERFERE_SOURCE_PORT: interfere_port,
                TELEMETRY_SOURCE_PORT: telemetry_port}

    if bit_files:
        missing = sorted(set(WAVES) - set(bit_files))
        if missing:
            raise ReplayError(f"bit_files 缺少这些通道：{', '.join(missing)}")
        port_to_wave = {INFO_SOURCE_PORT: WaveType.INFO.value,
                        INTERFERE_SOURCE_PORT: WaveType.INTERFERE.value}
        for block in document["blocks"]:
            if block.get("id") != "zeromq_pub_sink":
                continue
            wave = port_to_wave.pop(_sink_port(block), None)
            if wave is None:
                continue
            # 保持块名不变，这样连线不用动。
            block["id"] = "blocks_file_sink"
            block["parameters"] = {
                "affinity": "",
                "alias": "",
                "append": "False",
                "comment": f"{wave} 解调比特落盘",
                "file": json.dumps(str(Path(bit_files[wave]).resolve())),
                "maxoutbuf": "0",
                "minoutbuf": "0",
                "type": "byte",
                "unbuffered": "False",
                "vlen": "1",
            }
            replaced.append(block["name"])
        # 功率遥测输出留着不动（删掉会让上游向量块悬空），只把它挪到回放端口。
        _rewrite_zmq_ports(document, port_map)
    else:
        _rewrite_zmq_ports(document, port_map)

    applied: dict[str, str] = {}
    if overrides:
        known = variable_values(document)
        unknown = sorted(set(overrides) - set(known))
        if unknown:
            raise ReplayError(f"流图里没有这些变量：{', '.join(unknown)}")
        for name, value in overrides.items():
            for block in document["blocks"]:
                if block.get("id") == "variable" and block["name"] == name:
                    block["parameters"]["value"] = str(value)
                    applied[name] = str(value)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle, allow_unicode=True, sort_keys=False)

    return DeriveReport(
        source_grc=source_grc,
        derived_grc=dest,
        removed_blocks=removed,
        replaced_blocks=tuple(replaced),
        port_map=port_map,
        overrides=applied,
    )


def derive_record_flowgraph(
    dest: str | Path,
    *,
    record_file: str | Path,
    seconds: float,
    source_grc: str | Path = FLOWGRAPH_SOURCE,
    samp_rate: float | None = None,
    overrides: dict[str, str] | None = None,
) -> DeriveReport:
    """生成**保留实时源**的录制流图：原始复基带原样落盘，其余链路照旧。

    录制点是 ``rx_raw_baseband_offset`` 的输入，也就是回放时喂回去的那一点：
    录下来什么，回放就重放什么。同时把解调链照旧连着，所以录制的时候还能
    用 `tools/radio_probe.py` 看接收状态。

    ``blocks_head`` 收满 ``seconds × samp_rate`` 个样点后发 done，整张图随之
    收尾退出——不需要人工掐表，文件长度也是确定的。
    """
    source_grc = Path(source_grc)
    dest = Path(dest)
    document = load_flowgraph(source_grc)

    values = variable_values(document)
    rate = float(samp_rate if samp_rate is not None else values["samp_rate"])
    overrides = dict(overrides or {})
    if samp_rate is not None:
        overrides["samp_rate"] = repr(float(samp_rate))
        rate = float(samp_rate)
    items = int(round(seconds * rate))
    if items <= 0:
        raise ReplayError("录制时长换算出的样点数不是正数")

    names = {block["name"] for block in document["blocks"]}
    if SOURCE_BLOCK_NAME not in names:
        raise ReplayError(f"流图里没有 {SOURCE_BLOCK_NAME} 源块：{source_grc}")

    document["blocks"].extend([
        {
            "name": "record_head",
            "id": "blocks_head",
            "parameters": {
                "affinity": "", "alias": "", "comment": "",
                "maxoutbuf": "0", "minoutbuf": "0",
                "num_items": str(items), "type": "complex", "vlen": "1",
            },
            "states": {"bus_sink": False, "bus_source": False, "bus_structure": None,
                       "coordinate": [400, 1720], "rotation": 0, "state": "enabled"},
        },
        {
            "name": "record_file_sink",
            "id": "blocks_file_sink",
            "parameters": {
                "affinity": "", "alias": "", "append": "False",
                "comment": "原始复基带录制",
                "file": json.dumps(str(Path(record_file).resolve())),
                "maxoutbuf": "0", "minoutbuf": "0",
                "type": "complex", "unbuffered": "False", "vlen": "1",
            },
            "states": {"bus_sink": False, "bus_source": False, "bus_structure": None,
                       "coordinate": [400, 1820], "rotation": 0, "state": "enabled"},
        },
    ])

    rewired = 0
    connections = []
    for connection in document["connections"]:
        if connection[:4] == [SOURCE_BLOCK_NAME, "0", "rx_raw_baseband_offset", "0"]:
            connections.append([SOURCE_BLOCK_NAME, "0", "record_head", "0"])
            connections.append(["record_head", "0", "rx_raw_baseband_offset", "0"])
            connections.append(["record_head", "0", "record_file_sink", "0"])
            rewired += 1
            continue
        connections.append(connection)
    if not rewired:
        raise ReplayError("没找到源块到 rx_raw_baseband_offset 的连线，录制点插不进去")
    document["connections"] = connections

    removed = tuple(block["name"] for block in document["blocks"]
                    if block.get("id") == "xmlrpc_server")
    for name in removed:
        _drop_block(document, name)

    for name, value in overrides.items():
        found = False
        for block in document["blocks"]:
            if block.get("id") == "variable" and block["name"] == name:
                block["parameters"]["value"] = str(value)
                found = True
        if not found:
            raise ReplayError(f"流图里没有这个变量：{name}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle, allow_unicode=True, sort_keys=False)

    return DeriveReport(
        source_grc=source_grc,
        derived_grc=dest,
        removed_blocks=removed,
        replaced_blocks=("record_head", "record_file_sink"),
        port_map={},
        overrides=dict(overrides),
    )


def compile_flowgraph(
    grc_path: str | Path,
    out_dir: str | Path,
    *,
    grcc: str | None = None,
) -> Path:
    """用 `grcc` 把 .grc 编译成可直接运行的 Python 流图。"""
    grcc = grcc or shutil.which("grcc")
    if not grcc:
        raise ReplayError("找不到 grcc，请安装 gnuradio 包")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [grcc, "-o", str(out_dir), str(Path(grc_path).resolve())],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if process.returncode != 0:
        raise ReplayError(f"grcc 编译失败：{process.stderr.strip() or process.stdout.strip()}")
    generated = sorted(out_dir.glob("*.py"))
    if not generated:
        raise ReplayError(f"grcc 没有生成 Python 流图：{out_dir}")
    return generated[0]


@dataclass
class CaptureResult:
    """一次回放的统计结果。"""

    stats: dict[str, ChannelStats]
    seconds: float
    exit_code: int | None
    timed_out: bool
    stderr: str

    @property
    def info(self) -> ChannelStats:
        return self.stats[WaveType.INFO.value]

    @property
    def interfere(self) -> ChannelStats:
        return self.stats[WaveType.INTERFERE.value]


_CODEC_ENV = {"PYTHONNOUSERSITE": "1"}


def _run_flowgraph(
    flowgraph_python: str | Path,
    *,
    python: str | None,
    timeout: float,
    extra_args: tuple[str, ...] = (),
) -> tuple[subprocess.Popen, float]:
    environment = dict(os.environ, **_CODEC_ENV)
    started = time.monotonic()
    process = subprocess.Popen(
        [python or sys.executable, str(flowgraph_python), *extra_args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return process, time.monotonic() - started
    return process, time.monotonic() - started


def decode_bit_files(
    bit_files: dict[str, str | Path],
    *,
    channels: tuple[str, ...] = (WaveType.INFO.value,),
    chunk_bytes: int = 65536,
) -> dict[str, ChannelStats]:
    """把落盘的比特流喂进接收包自己的解码器，统计三级计数。"""
    stats = {}
    for wave in channels:
        counter = ChannelStats(WaveType(wave))
        path = Path(bit_files[wave])
        if path.is_file():
            with open(path, "rb") as handle:
                while True:
                    chunk = handle.read(chunk_bytes)
                    if not chunk:
                        break
                    counter.feed(chunk)
        stats[wave] = counter
    return stats


def run_replay_file(
    iq_file: str | Path,
    *,
    overrides: dict[str, str] | None = None,
    timeout: float = 120.0,
    work_dir: str | Path | None = None,
    keep_artifacts: bool = False,
    source_grc: str | Path = FLOWGRAPH_SOURCE,
    python: str | None = None,
    repeat: bool = False,
    channels: tuple[str, ...] = (WaveType.INFO.value,),
) -> CaptureResult:
    """派生 → 编译 → 满速跑完 → 离线解码。默认的单点流程。"""
    created = work_dir is None
    work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="jg_radio_replay_"))
    try:
        bit_files = {wave: work_dir / f"bits_{wave}.bin" for wave in WAVES}
        report = derive_replay_flowgraph(
            work_dir / "replay.grc",
            iq_file=iq_file,
            source_grc=source_grc,
            overrides=overrides,
            repeat=repeat,
            bit_files=bit_files,
        )
        flowgraph = compile_flowgraph(report.derived_grc, work_dir / "build")
        process, seconds = _run_flowgraph(flowgraph, python=python, timeout=timeout)
        _, stderr = process.communicate()
        if process.returncode not in (0, None):
            raise ReplayError(
                f"回放流图异常退出（code={process.returncode}）：{(stderr or '')[-800:]}"
            )
        stats = decode_bit_files(bit_files, channels=channels)
        return CaptureResult(
            stats=stats,
            seconds=seconds,
            exit_code=process.returncode,
            timed_out=False,
            stderr=(stderr or "")[-4000:],
        )
    finally:
        if keep_artifacts:
            print(f"派生产物保留在 {work_dir}", file=sys.stderr)
        elif created:
            shutil.rmtree(work_dir, ignore_errors=True)


def record_iq(
    record_file: str | Path,
    *,
    seconds: float = 10.0,
    samp_rate: float | None = None,
    sdr_uri: str | None = None,
    overrides: dict[str, str] | None = None,
    source_grc: str | Path = FLOWGRAPH_SOURCE,
    python: str | None = None,
    work_dir: str | Path | None = None,
    keep_artifacts: bool = False,
) -> Path:
    """录一段真实空口 IQ（需要 PlutoSDR 在线）。"""
    created = work_dir is None
    work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="jg_radio_record_"))
    record_file = Path(record_file)
    try:
        report = derive_record_flowgraph(
            work_dir / "record.grc",
            record_file=record_file,
            seconds=seconds,
            source_grc=source_grc,
            samp_rate=samp_rate,
            overrides=overrides,
        )
        flowgraph = compile_flowgraph(report.derived_grc, work_dir / "build")
        extra = ("--sdr-rx-uri", sdr_uri) if sdr_uri else ()
        process, elapsed = _run_flowgraph(
            flowgraph, python=python, timeout=seconds * 4.0 + 30.0, extra_args=extra
        )
        _, stderr = process.communicate()
        if process.returncode not in (0, None):
            raise ReplayError(
                f"录制流图异常退出（code={process.returncode}）：{(stderr or '')[-800:]}"
            )
        if not record_file.is_file() or record_file.stat().st_size == 0:
            raise ReplayError(f"录制没有产出数据：{record_file}")
        return record_file
    finally:
        if keep_artifacts:
            print(f"派生产物保留在 {work_dir}", file=sys.stderr)
        elif created:
            shutil.rmtree(work_dir, ignore_errors=True)


def capture_bits_live(
    flowgraph_python: str | Path,
    *,
    info_port: int = DEFAULT_INFO_PORT,
    interfere_port: int = DEFAULT_INTERFERE_PORT,
    target_bits: int = 400_000,
    timeout: float = 60.0,
    python: str | None = None,
    poll_interval: float = 0.1,
) -> CaptureResult:
    """起回放流图，边跑边旁听两路比特流。

    只用于交互观察：流图比 Python 解码快，收满 `target_bits` 就杀掉流图，
    队列里没发完的尾部消息会丢。要可复现的测量请用 `run_replay_file`。
    """
    import zmq

    ports = {WaveType.INFO.value: info_port, WaveType.INTERFERE.value: interfere_port}
    stats = {wave: ChannelStats(WaveType(wave)) for wave in ports}

    context = zmq.Context()
    poller = zmq.Poller()
    socket_to_wave: dict = {}
    for wave, port in ports.items():
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.connect(f"tcp://127.0.0.1:{port}")
        poller.register(socket, zmq.POLLIN)
        socket_to_wave[socket] = wave

    environment = dict(os.environ, **_CODEC_ENV)
    started = time.monotonic()
    process = subprocess.Popen(
        [python or sys.executable, str(flowgraph_python)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    timed_out = False

    def drain(quiet_period: float = 0.3) -> None:
        deadline = time.monotonic() + quiet_period
        while time.monotonic() < deadline:
            ready = poller.poll(0.05)
            if not ready:
                continue
            deadline = time.monotonic() + quiet_period
            for socket, _ in ready:
                stats[socket_to_wave[socket]].feed(socket.recv())

    try:
        while time.monotonic() - started < timeout:
            for socket, _ in poller.poll(poll_interval):
                stats[socket_to_wave[socket]].feed(socket.recv())
            if max(item.bits for item in stats.values()) >= target_bits:
                break
            if process.poll() is not None:
                drain()
                break
        else:
            timed_out = True
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for socket in socket_to_wave:
            socket.close()
        context.term()
        _, stderr = process.communicate()

    return CaptureResult(
        stats=stats,
        seconds=time.monotonic() - started,
        exit_code=process.returncode,
        timed_out=timed_out,
        stderr=(stderr or "")[-4000:],
    )


def replay_once(
    iq_file: str | Path,
    *,
    overrides: dict[str, str] | None = None,
    target_bits: int = 400_000,
    timeout: float = 120.0,
    work_dir: str | Path | None = None,
    keep_artifacts: bool = False,
    source_grc: str | Path = FLOWGRAPH_SOURCE,
    python: str | None = None,
    repeat: bool = False,
    channels: tuple[str, ...] = (WaveType.INFO.value,),
    live: bool = False,
) -> CaptureResult:
    """单点回放。`live=True` 走 ZMQ 旁听，否则满速跑完再离线解码。"""
    if not live:
        return run_replay_file(
            iq_file,
            overrides=overrides,
            timeout=timeout,
            work_dir=work_dir,
            keep_artifacts=keep_artifacts,
            source_grc=source_grc,
            python=python,
            repeat=repeat,
            channels=channels,
        )

    created = work_dir is None
    work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="jg_radio_replay_"))
    try:
        report = derive_replay_flowgraph(
            work_dir / "replay.grc",
            iq_file=iq_file,
            source_grc=source_grc,
            overrides=overrides,
            repeat=repeat,
        )
        flowgraph = compile_flowgraph(report.derived_grc, work_dir / "build")
        return capture_bits_live(
            flowgraph,
            target_bits=target_bits,
            timeout=timeout,
            python=python,
        )
    finally:
        if keep_artifacts:
            print(f"派生产物保留在 {work_dir}", file=sys.stderr)
        elif created:
            shutil.rmtree(work_dir, ignore_errors=True)


def format_counts(result: CaptureResult) -> str:
    """把一次回放的结果整理成几行文本。"""
    lines = []
    for wave, item in result.stats.items():
        sync_rate = item.frames / item.sync if item.sync else 0.0
        lines.append(
            f"{wave:8s} 比特 {item.bits:8d} | 空口帧 {item.sync:6d} | "
            f"序列帧 {item.frames:6d} | 业务包 {item.decoded:6d} | "
            f"包/kbit {item.decoded / max(item.bits, 1) * 1000:.2f} | "
            f"帧/同步 {sync_rate:.3f}"
        )
    lines.append(f"耗时 {result.seconds:.1f}s，流图退出码 {result.exit_code}"
                 + ("（超时）" if result.timed_out else ""))
    if result.stderr.strip():
        lines.append("流图输出：" + result.stderr.strip()[-400:])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iq", help="复基带 IQ 文件（complex64 交错）；录制模式下不用填")
    parser.add_argument("--record", metavar="OUT.c64",
                        help="录制模式：保留 PlutoSDR 源，把原始复基带录到文件")
    parser.add_argument("--seconds", type=float, default=10.0, help="录制时长，秒")
    parser.add_argument("--samp-rate", type=float, default=None, help="录制采样率，默认沿用流图")
    parser.add_argument("--sdr-uri", default=None, help="PlutoSDR 地址，默认沿用流图")
    parser.add_argument("--bits", type=int, default=400_000,
                        help="旁听模式下的停止条件（落盘模式忽略，按文件长度算）")
    parser.add_argument("--timeout", type=float, default=120.0, help="单点墙钟上限，秒")
    parser.add_argument("--set", action="append", default=[], metavar="VAR=VALUE",
                        help="覆盖流图变量，可重复，例如 --set loop_bw=0.002")
    parser.add_argument("--repeat", action="store_true",
                        help="循环播放 IQ（默认播一遍就停，避免回放接缝丢帧）")
    parser.add_argument("--live", action="store_true", help="走 ZMQ 旁听，不落盘")
    parser.add_argument("--interfere", action="store_true", help="同时统计干扰波通道")
    parser.add_argument("--keep", action="store_true", help="保留派生的 .grc 和编译产物")
    parser.add_argument("--work-dir", type=Path, default=None, help="派生产物目录")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = parser.parse_args(argv)

    overrides: dict[str, str] = {}
    for item in args.set:
        if "=" not in item:
            parser.error(f"--set 需要 VAR=VALUE 形式：{item}")
        name, value = item.split("=", 1)
        overrides[name.strip()] = value.strip()

    if args.record:
        path = record_iq(
            args.record,
            seconds=args.seconds,
            samp_rate=args.samp_rate,
            sdr_uri=args.sdr_uri,
            overrides=overrides,
            keep_artifacts=args.keep,
            work_dir=args.work_dir,
        )
        size = path.stat().st_size
        rate = args.samp_rate or float(variable_values(load_flowgraph())["samp_rate"])
        print(f"已录制 {path}（{size / 1e6:.1f} MB，约 {size / 8 / rate:.1f} 秒）")
        return 0

    if not args.iq:
        parser.error("需要 --iq 指定回放文件，或 --record 走录制模式")

    ignored = sorted(set(overrides) - EFFECTIVE_VARIABLES)
    if ignored:
        print(f"警告：{', '.join(ignored)} 在文件回放里不改变 DSP 行为", file=sys.stderr)

    channels = (WaveType.INFO.value,) + ((WaveType.INTERFERE.value,) if args.interfere else ())
    result = replay_once(
        args.iq,
        overrides=overrides,
        target_bits=args.bits,
        timeout=args.timeout,
        work_dir=args.work_dir,
        keep_artifacts=args.keep,
        repeat=args.repeat,
        channels=channels,
        live=args.live,
    )

    if args.json:
        print(json.dumps({
            "seconds": round(result.seconds, 3),
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "channels": {
                wave: {
                    "bits": item.bits,
                    "sync": item.sync,
                    "frames": item.frames,
                    "decoded": item.decoded,
                }
                for wave, item in result.stats.items()
            },
        }, ensure_ascii=False, indent=2))
    else:
        print(format_counts(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
