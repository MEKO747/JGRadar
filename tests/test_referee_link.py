"""裁判串口**单点所有权**：一个 fd 既读 0x020C/0x020E，又写 0x0305。

联合启动时串口只能有一个进程持有——旧实现是接收节点开读、runner 开写，
两个进程抢同一个 tty；更糟的是等级改了也不重发配置，融合侧一直用旧值。

这里用 pty 当**真串口**来验（不接 Pluto 也能测到底）：
pty 就是内核 tty，行为和 `/dev/ttyUSB0` 一样，区别只有物理层。
"""

from __future__ import annotations

import os
import pty
import time

from rmuc.coder import SerialFrameDecoder
from rmuc.referee_frame import CMD_RADAR_POSITION, RadarPositionWriter, build_frame
from rmuc.referee_link import RefereeSerialLink

PAYLOAD = bytes(range(48))


def radar_info_frame(level: int) -> bytes:
    """0x020E：byte0 的 bit3~4 是加密等级。"""
    first = (int(level) & 0x03) << 3
    return build_frame(0x020E, bytes([first, 0, 0, 0, 0, 0, 0, 0]), seq=0)


def marking_frame(raw: int) -> bytes:
    return build_frame(0x020C, int(raw).to_bytes(2, "little"), seq=1)


def wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def read_available(fd: int, timeout: float = 3.0) -> bytes:
    os.set_blocking(fd, False)
    data = b""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            chunk = os.read(fd, 4096)
        except BlockingIOError:
            time.sleep(0.005)
            continue
        if chunk:
            data += chunk
            if data:
                break
    return data


def test_same_fd_reads_referee_frames_and_writes_0x0305():
    master, slave = pty.openpty()
    events = {"marking": [], "info": []}
    link = RefereeSerialLink(
        os.ttyname(slave),
        on_marking=events["marking"].append,
        on_radar_info=events["info"].append,
    )
    writer = RadarPositionWriter(write=link.write)
    try:
        link.start()
        os.write(master, marking_frame(0b0000_0000_0000_0001))
        os.write(master, radar_info_frame(3))
        assert wait_for(lambda: events["marking"] and events["info"]), link.snapshot()
        assert events["info"][-1].encryption_level == 3

        # 写方向走的是同一个句柄
        assert writer(PAYLOAD) is True
        frames = SerialFrameDecoder().feed(read_available(master))
        assert [frame.cmd_id for frame in frames] == [CMD_RADAR_POSITION]
        assert frames[0].payload == PAYLOAD
        assert link.snapshot()["write_frames"] == 1
        assert link.snapshot()["radar_info_frames"] == 1
    finally:
        link.stop()
        os.close(master)
        os.close(slave)


def test_unplugged_cable_reports_failure_instead_of_raising():
    """拔线：写失败返回 False、不抛异常，主循环照常跑。"""
    master, slave = pty.openpty()
    link = RefereeSerialLink(os.ttyname(slave))
    writer = RadarPositionWriter(write=link.write)
    try:
        link.start()
        os.close(master)                      # 模拟拔线：master 端消失
        time.sleep(0.1)
        assert link.write(b"\x00" * 48) is False
        assert link.snapshot()["write_failed"] is True
        assert link.snapshot()["last_write_error"]
        assert writer(PAYLOAD) is False       # writer 也不许把失败报成成功
        assert writer.frames_written == 0
    finally:
        link.stop()
        os.close(slave)


def test_level_change_from_0x020e_is_published_once_through_the_runner_wiring():
    """联合启动时 runner 读 0x020E，等级变化才发配置（重复等级不刷镜头）。"""
    from types import SimpleNamespace

    from jg_radar.integration.level_sync import LevelSync
    from tools.run_radio_vision import _open_referee_link

    class FakeBridge:
        def __init__(self):
            self.markings = []
            self.infos = []
            self.published_levels = []
            self._sync = LevelSync(self.published_levels.append, initial=1)

        def publish_marking(self, marking, *, own_team):
            self.markings.append((marking, own_team))

        def publish_radar_info(self, info):
            self.infos.append(info)

        def set_level(self, level):
            return self._sync.set(level)

    master, slave = pty.openpty()
    bridge = FakeBridge()
    args = SimpleNamespace(referee_serial=os.ttyname(slave), referee_baud=115_200,
                           referee_level=True, team="red")
    link = _open_referee_link(args, bridge)
    try:
        os.write(master, radar_info_frame(3))
        os.write(master, radar_info_frame(3))   # 重复等级：不该再发
        os.write(master, radar_info_frame(2))
        assert wait_for(lambda: bridge.published_levels == [3, 2]), bridge.published_levels
        assert len(bridge.infos) == 3
    finally:
        link.stop()
        os.close(master)
        os.close(slave)
