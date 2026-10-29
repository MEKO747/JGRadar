"""输出落盘：JSONL。每行一个 ``FrameOutput``。

线程安全：写操作在一把锁内完成，便于将来 Stage B 线程边跑边落盘。
``NullSink`` 用于测试与性能基线（不落盘）。

回放/金标准比对依赖**字节级确定性**：键顺序由 dataclass 字段顺序决定，
浮点统一 round，``sort_keys`` 不开启（否则槽位顺序会被打乱，掩盖 bug）。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import TracebackType
from typing import IO, Protocol

from jg_radar.types import FrameOutput


class Sink(Protocol):
    """输出终端协议。"""

    def write(self, frame: FrameOutput) -> None: ...

    def close(self) -> None: ...


class JsonlSink:
    """把 ``FrameOutput`` 逐行写成 JSON。"""

    def __init__(self, path: str | Path, *, flush_every: int = 1) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] = self.path.open("w", encoding="utf-8")
        self._lock = threading.Lock()
        self._flush_every = max(1, int(flush_every))
        self._since_flush = 0
        self._count = 0

    def write(self, frame: FrameOutput) -> None:
        line = json.dumps(frame.to_dict(), ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")
            self._count += 1
            self._since_flush += 1
            if self._since_flush >= self._flush_every:
                self._fh.flush()
                self._since_flush = 0

    @property
    def count(self) -> int:
        return self._count

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()
                self._fh.close()

    def __enter__(self) -> "JsonlSink":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class NullSink:
    """不落盘，只计数。"""

    def __init__(self) -> None:
        self._count = 0

    def write(self, frame: FrameOutput) -> None:
        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def close(self) -> None:  # noqa: D102
        pass


def read_jsonl(path: str | Path) -> list[FrameOutput]:
    """读回 JSONL，用于回放与金标准比对。"""
    out: list[FrameOutput] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(FrameOutput.from_dict(json.loads(line)))
    return out
