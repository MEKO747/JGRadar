"""等级同步：``0x020E`` 的加密等级只在**变化时**发布。

以前两个坑：runner 每秒重发 1 级把接收侧改过的等级顶回去；接收节点改了内部值
却不重发。判定逻辑收进这个纯类，不依赖 rclpy，方便离线断言。
"""

from __future__ import annotations

import pytest

from jg_radar.integration.level_sync import LevelSync


def test_publishes_only_when_the_level_changes():
    published: list[int] = []
    sync = LevelSync(published.append, initial=1)
    assert sync.set(1) is False            # 和启动值一样，不发
    assert published == []
    assert sync.set(3) is True
    assert published == [3]
    assert sync.set(3) is False            # 重复等级不刷话题
    assert published == [3]
    assert sync.set(2) is True
    assert published == [3, 2]
    assert sync.snapshot() == {"interference_level": 2, "level_publishes": 2}


def test_external_mode_starts_empty_and_publishes_the_first_level():
    published: list[int] = []
    sync = LevelSync(published.append)      # initial=None：还没人说过等级
    assert sync.set(2) is True
    assert published == [2]


def test_invalid_levels_are_ignored():
    published: list[int] = []
    sync = LevelSync(published.append, initial=1)
    for level in (0, 4, 255):
        assert sync.set(level) is False
    assert published == []
    with pytest.raises(ValueError):
        LevelSync(published.append, initial=7)
