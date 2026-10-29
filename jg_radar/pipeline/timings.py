"""分段计时埋点 + EMA。**只测量，不参与任何判断。**

方案 §十一 把 50ms 的预算拆成了八个环节。拆完之后必须能**看见**每一段实际
花了多少，否则预算表就只是一张好看的表格——真到了现场发现掉帧，你并不知道
该去看哪一段。

EGA 的做法是给每个环节编号（01–12）打时间戳，这里取它的形状：命名的分段 +
段间耗时，配一个指数滑动平均把单帧抖动抹平。单帧耗时是没有意义的（一次 GC
就能让它翻十倍），要看的是**趋势**。

## 一条硬规矩

本模块读的是 ``time.perf_counter``——**墙上时钟**。而流水线里的正确性判断
（新鲜度、看门狗、帧龄）读的必须是**帧时间线**，两者绝不能混用。

区别在回放时最明显：回放跑得比实时快得多，一帧的处理耗时可能是 0.3ms 而
它的拍摄间隔是 50ms。拿墙上时钟去算帧龄，回放里的每一帧都会被判成"超前"，
而实时里同样的代码又会因为一次卡顿把整批帧判成超龄。所以计时归计时
（``Timings``），判断归判断（事件循环的 clock），中间没有桥。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Iterator

__all__ = ["Timings"]


class Timings:
    """命名分段的 EMA 计时器。

    典型用法::

        timings = Timings(alpha=0.1)
        with timings.measure("stage_a"):
            ...
        with timings.measure("stage_b"):
            ...
        frame.diagnostics.update(timings.snapshot())

    同一段被多次 ``measure`` 时逐次并入 EMA——一帧里跑两次的段（比如每车
    一次 ROI 裁切）反映的是"每次的平均"，不是总和。要总和就自己在外面数。
    """

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("alpha 必须落在 (0, 1] 内；取 1 表示不平均")
        self.alpha = float(alpha)
        self._clock = clock
        self._ema: dict[str, float] = {}
        self._last: dict[str, float] = {}
        self._count: dict[str, int] = {}

    # -- 记录 -------------------------------------------------------------
    def observe(self, name: str, seconds: float) -> float:
        """记一段耗时，返回更新后的 EMA。"""
        value = max(0.0, float(seconds))
        self._count[name] = self._count.get(name, 0) + 1
        if name not in self._ema:
            # 第一帧直接就是观测值：从 0 开始爬的话，启动后头十几帧的均值
            # 会被严重低估，而那正是最需要看真实数字的时候。
            self._ema[name] = value
        else:
            a = self.alpha
            self._ema[name] = (1.0 - a) * self._ema[name] + a * value
        self._last[name] = value
        return self._ema[name]

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """给一段代码计时。异常路径也照样记录——卡住的那一段最需要被看见。"""
        start = self._clock()
        try:
            yield
        finally:
            self.observe(name, self._clock() - start)

    # -- 查询 -------------------------------------------------------------
    def ema(self, name: str) -> float:
        return self._ema.get(name, 0.0)

    def last(self, name: str) -> float:
        """最近一次的耗时（秒）。诊断抖动看这个。"""
        return self._last.get(name, 0.0)

    def count(self, name: str) -> int:
        return self._count.get(name, 0)

    def snapshot(self, *, unit: str = "ms") -> dict[str, float]:
        """全部段的 EMA，键形如 ``t_stage_a_ms``，可直接并进 diagnostics。

        用带单位的键名，而不是一个全局约定：``FrameOutput.diagnostics`` 是
        一个扁平的数字字典，混进一个"秒"进去就再也分不清了。
        """
        scales = {"us": 1e6, "ms": 1e3, "s": 1.0}
        if unit not in scales:
            raise ValueError(f"未知单位 {unit!r}；可用：{sorted(scales)}")
        scale = scales[unit]
        return {f"t_{name}_{unit}": v * scale for name, v in self._ema.items()}

    def total(self, *, unit: str = "ms") -> float:
        """各段 EMA 之和。注意这**不等于**帧周期：段与段之间还有不计时的开销。"""
        return sum(self.snapshot(unit=unit).values())

    def names(self) -> tuple[str, ...]:
        return tuple(self._ema)

    def reset(self) -> None:
        self._ema.clear()
        self._last.clear()
        self._count.clear()
