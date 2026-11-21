"""等级同步：``0x020E`` 的加密等级 → ``/rmuc/demod_config``，**变化时才发**。

以前这里是两个坑叠在一起：

1. 联合启动时 runner 每秒固定重发一次 1 级配置，把接收节点刚从 ``0x020E``
   切到的 3 级顶回去；
2. 接收节点内部改了等级却不重发，融合侧只能继续用旧值。

现在只有一个来源：谁读到 ``0x020E``，谁就在**等级变化时**发一次。重复的等级
不刷话题，也不会覆盖别人的配置。这个类是纯逻辑（不 import rclpy），
方便离线断言「变了才发、不变不发、非法值不发」。
"""

from __future__ import annotations

from typing import Callable

__all__ = ["LevelSync", "VALID_LEVELS"]

VALID_LEVELS = (1, 2, 3)


class LevelSync:
    """记住「上次发出去的等级」，只在变化时调用 ``emit``。"""

    def __init__(self, emit: Callable[[int], None], initial: int | None = None) -> None:
        if initial is not None and initial not in VALID_LEVELS:
            raise ValueError(f"干扰等级只能是 {VALID_LEVELS}，收到 {initial}")
        self._emit = emit
        self.level = initial
        #: 已经发出去的次数，现场用来判断「等级到底有没有跟着变」。
        self.published = 0

    def set(self, level: int) -> bool:
        """设置等级；只有真的变了才发布，返回是否发布了。"""
        level = int(level)
        if level not in VALID_LEVELS:
            return False
        if level == self.level:
            return False
        self.level = level
        self._emit(level)
        self.published += 1
        return True

    def snapshot(self) -> dict:
        return {"interference_level": self.level, "level_publishes": self.published}
