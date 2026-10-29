"""帧源：把"一帧"喂给事件循环。

三种来源，同一个形状：

- ``ListFrameSource``      测试用，帧已经在内存里
- ``ReplayFrameSource``    回放目录（manifest + cascade.jsonl）
- ``SyntheticFrameSource`` 按固定节奏造空帧，给线程化路径与冒烟测试用

## 回放里装的是什么

一帧要么带 ``image``（真跑检测级联），要么带 ``cascade``（检测结果已经算好）。

回放目录装的是后者——**级联的输出**，不是世界坐标。这个区分很要紧：

- 存**世界坐标**等于把检测、投影两段都跳过，回放测的就只剩跟踪和仲裁；
- 存**级联输出**（车框、装甲板框、数字）则把投影那段留在流水线里**真跑**，
  只有"神经网络"那一步被替换掉。本项目没有权重，能真跑的就要真跑。

回放目录里的相机与位姿写在 ``manifest.json`` 里，fixture 因此是**自描述**的：
测试不必在代码里重复一遍常量，换一台相机也不用改测试。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from jg_radar.detect.factory import CascadeResult
from jg_radar.detect.protocol import DetectionRecord

__all__ = [
    "Frame",
    "FrameSource",
    "ListFrameSource",
    "ReplayCalibration",
    "ReplayFrameSource",
    "SyntheticFrameSource",
    "cascade_from_dict",
    "cascade_to_dict",
    "iterate",
]


@dataclass(frozen=True, eq=False)
class Frame:
    """一帧输入。

    ``eq=False`` 是**故意的**：``image`` 是 ndarray，dataclass 自动生成的
    ``__eq__`` 会退化成逐元素比较并返回一个数组，于是 ``assert a == b``
    对数组求布尔值会直接抛 "truth value of an array is ambiguous"。
    一个只在测试里炸、且报错信息完全指不到原因的比较，不如不要。
    """

    seq: int
    captured_at: float
    """**采集时刻**，单调时钟。下游所有新鲜度判断都以它为基准，
    所以它必须来自采集那一步，不能是"处理完的时刻"。"""
    image: np.ndarray | None = None
    cascade: CascadeResult | None = None
    """预先算好的级联输出（回放路径）。``None`` 表示这一帧没带，
    调用方得拿 ``image`` 自己去跑检测级联。"""

    @property
    def has_detections(self) -> bool:
        """这一帧是否**带着**检测结果。

        注意这和"检测结果为空"是两回事：``cascade`` 存在但 ``cars`` 为空，
        表示"这一帧画面里确实没有车"——那是最需要被稳定复现的降级场景，
        不能被当成"还没检测，去跑一遍模型"。
        """
        return self.cascade is not None

    @classmethod
    def with_cascade(
        cls, seq: int, captured_at: float, cascade: CascadeResult
    ) -> "Frame":
        return cls(
            seq=int(seq), captured_at=float(captured_at), image=None, cascade=cascade
        )


@runtime_checkable
class FrameSource(Protocol):
    """帧来源。"""

    is_live: bool
    """``captured_at`` 是不是**墙上时钟**（``time.monotonic()``）打的。

    回放、合成、内存列表都是 ``False``：它们的 ``captured_at`` 取自**帧时间
    线**（第 0 帧落在 100.0 秒这种任意起点上），跟开机多久没有关系。

    这个标记决定事件循环用哪个 ``now``——见 ``event_loop.default_clock``。
    接相机驱动时**必须**置 ``True`` 并用 ``time.monotonic()`` 打戳。

    **忘了声明不会得到默认值，会直接报错**（``default_clock`` 抛
    ``TypeError``）。因为猜错有两个方向，而它们差着量级：把回放当成实时，
    帧龄是几十万秒，``StageB`` 当场炸——吵，但安全；把实时当成回放，帧龄
    恒为 0，``max_commit_age_s`` 永远不触发，超龄数据照常上报——**全程没有
    任何异常**，只是位置偶尔旧一点。一个忘了写这行的相机驱动会落进后者。

    包一层别人的帧源时也要自己声明：它是 Protocol 的一部分，不会自动继承
    （``tests/test_pipeline_stages.py::ClosingSource`` 就是这么个例子）。
    """

    def read(self) -> Frame | None:
        """取下一帧。没有更多帧时返回 ``None``（**不是**抛异常）。"""
        ...

    def close(self) -> None: ...


def iterate(source: FrameSource) -> Iterator[Frame]:
    """把 ``read`` 风格的源变成可 for 的迭代器。"""
    while True:
        frame = source.read()
        if frame is None:
            return
        yield frame


# ---------------------------------------------------------------------------
# 级联输出的序列化
# ---------------------------------------------------------------------------
def _record_to_dict(r: DetectionRecord) -> dict[str, Any]:
    return {
        "class_id": int(r.class_id),
        "confidence": float(r.confidence),
        "bbox": [float(v) for v in r.bbox],
        "label": r.label,
    }


def _record_from_dict(d: Mapping[str, Any]) -> DetectionRecord:
    return DetectionRecord(
        class_id=int(d["class_id"]),
        confidence=float(d["confidence"]),
        bbox=tuple(float(v) for v in d["bbox"]),  # type: ignore[arg-type]
        label=str(d.get("label", "")),
    )


def cascade_to_dict(result: CascadeResult) -> dict[str, Any]:
    """级联输出 → 可 JSON 化的字典。``rois`` 不存，它由车框推得。"""
    return {
        "cars": [_record_to_dict(c) for c in result.cars],
        "armors": [
            {
                "car_index": int(h.car_index),
                "record": _record_to_dict(h.record),
            }
            for h in result.armors
        ],
        "digits": [
            {"class_id": int(v.class_id), "confidence": float(v.confidence),
             "label": v.label}
            for v in result.digits
        ],
    }


def cascade_from_dict(d: Mapping[str, Any]) -> CascadeResult:
    """字典 → 级联输出。``rois`` 从车框重建。"""
    from jg_radar.detect.armor_detector import ArmorHit
    from jg_radar.detect.digit_classifier import DigitVote

    cars = [_record_from_dict(c) for c in d.get("cars", ())]
    armors = [
        ArmorHit(car_index=int(h["car_index"]), record=_record_from_dict(h["record"]))
        for h in d.get("armors", ())
    ]
    digits = [
        DigitVote(
            class_id=int(v["class_id"]),
            confidence=float(v["confidence"]),
            label=str(v.get("label", "")),
        )
        for v in d.get("digits", ())
    ]
    rois = (
        np.asarray([c.bbox for c in cars], dtype=np.float64).reshape(-1, 4)
        if cars
        else np.zeros((0, 4), dtype=np.float64)
    )
    return CascadeResult(cars=cars, rois=rois, armors=armors, digits=digits)


# ---------------------------------------------------------------------------
# 内存里的帧
# ---------------------------------------------------------------------------
class ListFrameSource:
    """帧已经全部在内存里。测试的主力。"""

    is_live = False
    """时间戳由调用方给，循环不该拿墙上时钟去比。"""

    def __init__(self, frames: Sequence[Frame]) -> None:
        self._frames = list(frames)
        self._i = 0

    def read(self) -> Frame | None:
        if self._i >= len(self._frames):
            return None
        frame = self._frames[self._i]
        self._i += 1
        return frame

    def close(self) -> None:
        pass

    @property
    def remaining(self) -> int:
        return len(self._frames) - self._i

    def __len__(self) -> int:
        return len(self._frames)


class SyntheticFrameSource:
    """按固定节奏产出空帧。

    给两条路径用：线程化事件循环需要"一直有帧"才能观察超龄行为；
    冒烟测试需要一条不依赖任何 fixture 的输入。
    """

    is_live = False
    """``captured_at`` 从 ``start_at`` 起按 ``dt`` 推，与墙上时钟无关。"""

    def __init__(
        self,
        *,
        count: int,
        dt: float = 0.05,
        start_at: float = 0.0,
        width: int = 320,
        height: int = 240,
        make_image: Callable[[int], np.ndarray] | None = None,
    ) -> None:
        if dt <= 0:
            raise ValueError("dt 必须为正")
        self.count = max(0, int(count))
        self.dt = float(dt)
        self.start_at = float(start_at)
        self.width = int(width)
        self.height = int(height)
        self._make_image = make_image
        self._i = 0

    def read(self) -> Frame | None:
        if self._i >= self.count:
            return None
        i = self._i
        self._i += 1
        image = (
            self._make_image(i)
            if self._make_image is not None
            else np.zeros((self.height, self.width, 3), dtype=np.uint8)
        )
        return Frame(seq=i, captured_at=self.start_at + i * self.dt, image=image)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 回放目录
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReplayCalibration:
    """回放目录自带的标定。用来现场构建查找表。"""

    width: int
    height: int
    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray
    t: np.ndarray
    plane_z: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "K": np.asarray(self.K, dtype=float).tolist(),
            "dist": np.asarray(self.dist, dtype=float).tolist(),
            "R": np.asarray(self.R, dtype=float).tolist(),
            "t": np.asarray(self.t, dtype=float).tolist(),
            "plane_z": float(self.plane_z),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ReplayCalibration":
        return cls(
            width=int(d["width"]),
            height=int(d["height"]),
            K=np.asarray(d["K"], dtype=np.float64).reshape(3, 3),
            dist=np.asarray(d["dist"], dtype=np.float64).reshape(-1),
            R=np.asarray(d["R"], dtype=np.float64).reshape(3, 3),
            t=np.asarray(d["t"], dtype=np.float64).reshape(3),
            plane_z=float(d.get("plane_z", 0.0)),
        )


class ReplayFrameSource:
    """读一个回放目录：``manifest.json`` + ``cascade.jsonl``。

    目录结构::

        replay_seq_001/
          manifest.json     相机、位姿、帧数、节奏
          cascade.jsonl     每行一帧的级联输出

    ``cascade.jsonl`` 缺失时退化成"按 manifest 造空帧"——用来测
    "整段没有任何目标"这种降级场景。
    """

    is_live = False
    """帧时间线是录制时的那条，不是现在。拿墙上时钟比它必然错得离谱。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        manifest_path = self.path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"回放目录缺少 manifest.json：{self.path}")

        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.calibration = ReplayCalibration.from_dict(raw["calibration"])
        self.frame_count = int(raw["frame_count"])
        self.dt = float(raw.get("dt", 0.05))
        self.start_at = float(raw.get("start_at", 0.0))
        self.name = str(raw.get("name", self.path.name))
        self.description = str(raw.get("description", ""))

        self._cascades = self._load_cascade()
        self._i = 0

    def _load_cascade(self) -> dict[int, CascadeResult]:
        path = self.path / "cascade.jsonl"
        if not path.is_file():
            return {}
        out: dict[int, CascadeResult] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[int(row["seq"])] = cascade_from_dict(row)
        return out

    def read(self) -> Frame | None:
        if self._i >= self.frame_count:
            return None
        i = self._i
        self._i += 1
        # 每一帧都带 cascade（哪怕里面是空的）——"画面里没有车"必须是
        # 一条**明确的**信息，而不是"没带检测、去跑模型吧"。
        cascade = self._cascades.get(i)
        if cascade is None:
            cascade = cascade_from_dict({})
        return Frame.with_cascade(
            seq=i,
            captured_at=self.start_at + i * self.dt,
            cascade=cascade,
        )

    def close(self) -> None:
        pass

    # -- 标定到查找表 -----------------------------------------------------
    def build_lookup(self, *, stride: int = 1):
        """用目录里的标定**现场构建**查找表。

        刻意不落盘成 ``.npz``：回放 fixture 要能跟着代码一起改，缓存文件
        会有"改了相机忘了重算表"的风险，而那个风险的症状是"坐标整体偏一点"
        ——最难发现的那一类。
        """
        from jg_radar.transform import CameraModel, Intrinsics, build_lookup_map

        cal = self.calibration
        camera = CameraModel(
            intrinsics=Intrinsics.from_matrix(cal.K, cal.dist),
            width=cal.width,
            height=cal.height,
        )
        return build_lookup_map(camera, cal.R, cal.t, cal.plane_z, stride=stride)

    def __repr__(self) -> str:  # pragma: no cover - 诊断用
        return f"ReplayFrameSource({self.path.name!r}, {self.frame_count} frames)"
