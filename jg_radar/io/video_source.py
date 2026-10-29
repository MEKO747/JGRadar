"""视频帧源：把一段**真实录像**喂给流水线。

## 为什么需要这个文件

项目里原本只有三种帧源，没有一种能吃真实图像：

- ``ReplayFrameSource`` 装的是**级联输出**（检测已经算好），用来测跟踪与仲裁；
- ``SyntheticFrameSource`` 造空帧；
- ``ListFrameSource`` 帧已经在内存里（测试用）。

于是"在真实图像上跑检测"这条路是断的——没有相机就跑不了。这个文件的
全部意义就是把那段补上：**不需要相机、不需要驱动、不需要现场**，一段
mp4 就能让检测→投影→跟踪→仲裁整条链吃上真实数据。

它同时也是"接真实相机"的预演：相机驱动将来要做的事，就是按
``Frame(seq, captured_at, image)`` 产帧并把 ``is_live`` 置 ``True``。
这里除了 ``is_live`` 和时间戳来源，其余一模一样。

## 时间戳：必须取**采集时刻**

有旁边的 ``*_timestamps.csv`` 就用它（EGA 的回放包里带，列为
``frame_index, wall_time_ns, wall_time_s, video_time_s, camera_stamp``），
取 ``camera_stamp`` 当 ``captured_at``。

这不是讲究，是正确性：``track/kalman.py`` 的 dt 直接由相邻两帧的
``captured_at`` 相减得到。用"读到这一帧的时刻"（30fps 录像在快机器上
可能只隔 3ms）会让滤波器以为机器人瞬移，χ² 门随即拒掉真实观测。

没有 CSV 时退回 ``start_at + i / fps``——那是"按标称帧率等距"的近似，
只在没有更好信息时使用。

``is_live = False``：帧时间线来自录制，跟开机多久无关。事件循环因此会用
``frame_clock``（``now = captured_at``），帧龄恒为 0、结果可复现——这正是
回放该有的语义。**不要**因为"这是真实数据"就把它标成 ``True``：那样
``now`` 会取墙上时钟，而录制时间是 2026 年，帧龄会算出几亿秒，
``StageB`` 当场抛 ``ClockMismatch``。

## 自描述的标定

和 ``ReplayFrameSource`` 一样，标定从**录像旁边的文件**里读，而不是从
配置里重复一遍。EGA 的回放包提供了两个来源，而它们**不一致**：

- ``calibration_snapshot.yaml`` 的 ``extrinsics.translation_vector``
- ``extrinsics.txt`` 的 ``translation_vector``（与快照的 y、z **符号相反**）

实测（用 4024×3036 的真实内参建表、看地面覆盖率）：
``extrinsics.txt`` 给出的光心在场地坐标系里是 ``(-83, 663, 388) cm``
——场地边缘外侧 0.8m、高 3.9m，视轴俯角 20.6°，**地面覆盖率 100%**；
快照那份算出来的光心 z 是 **负的 4.4m**（相机在地下），覆盖率 0。
EGA 自己的回放程序也优先读 ``extrinsics.txt``（其 replay_README：
"extrinsics.txt（可选；没有时由快照 R/t 临时生成）"）。

所以这里**优先 .txt**，并把这个取舍写在注释里——否则下一个人会以为是
bug 而"修"回去。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jg_radar.io.frame_source import Frame, ReplayCalibration

__all__ = ["VideoFrameSource", "load_ega_calibration"]

#: EGA 的场地坐标是**米**（field_length: 28.0），本项目是**厘米**。
#: 这个系数只作用于平移向量；旋转无量纲、内参与畸变也与单位无关。
EGA_WORLD_TO_CM = 100.0


def _read_extrinsics_txt(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """读 EGA 的 ``extrinsics.txt``：``rotation_matrix:`` 后 3 行 + ``translation_vector:`` 后 3 行。

    格式是裸文本（每行一个数），不是 YAML——所以这里手写一个最小的
    状态机，而不是引 yaml。解析失败返回 ``None``（调用方退回快照）。
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    rows: list[list[float]] = []
    vec: list[float] = []
    mode: str | None = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        low = line.lower().rstrip(":")
        if low.startswith("rotation_matrix"):
            mode = "R"
            continue
        if low.startswith("translation_vector"):
            mode = "t"
            continue
        if low.startswith("extrinsic_3x4"):
            mode = None  # 后面那份是拼好的，不重复取
            continue
        if mode is None:
            continue
        parts = line.replace(",", " ").split()
        if not parts:
            continue
        try:
            nums = [float(v) for v in parts]
        except ValueError:
            mode = None
            continue
        if mode == "R" and len(nums) == 3 and len(rows) < 3:
            rows.append(nums)
        elif mode == "t" and len(nums) == 1 and len(vec) < 3:
            vec.append(nums[0])
    if len(rows) == 3 and len(vec) == 3:
        return np.asarray(rows, dtype=np.float64), np.asarray(vec, dtype=np.float64)
    return None


def load_ega_calibration(replay_dir: str | Path) -> ReplayCalibration | None:
    """从一个 EGA 回放目录里读出标定，换算成本项目的单位。

    返回 ``None`` 表示"这个目录里没有可识别的 EGA 标定"——调用方应当
    退回配置里的相机参数，而不是猜。

    **这是数据适配，不是运行时逻辑**：它存在的唯一原因是"第一份真实标定
    只能从别人那里借"。将来自己标定之后，这段代码可以删掉。
    """
    import yaml

    d = Path(replay_dir)
    snap = d / "calibration_snapshot.yaml"
    if not snap.is_file():
        return None
    raw = yaml.safe_load(snap.read_text(encoding="utf-8")) or {}
    intr = raw.get("intrinsics") or {}
    K = np.asarray(intr.get("K"), dtype=np.float64).reshape(3, 3)
    dist = np.asarray(intr.get("dist_coeffs"), dtype=np.float64).reshape(-1)

    ext = raw.get("extrinsics") or {}
    R = np.asarray(ext.get("rotation_matrix"), dtype=np.float64).reshape(3, 3)
    t = np.asarray(ext.get("translation_vector"), dtype=np.float64).reshape(3)

    txt = _read_extrinsics_txt(d / "extrinsics.txt")
    if txt is not None:
        R_txt, t_txt = txt
        # 只有当两份旋转一致时才采信 .txt 的平移——否则说明这两个文件
        # 根本不是同一次标定，此时"优先谁"是个没有答案的问题，退回快照。
        if np.allclose(R_txt, R, atol=1e-6):
            t = t_txt

    return ReplayCalibration(
        width=int(intr.get("image_width", 0)),
        height=int(intr.get("image_height", 0)),
        K=K,
        dist=dist,
        R=R,
        t=t * EGA_WORLD_TO_CM,
        plane_z=0.0,
    )


def _read_timestamps_csv(path: Path) -> dict[int, float]:
    """读 ``frame_index, ..., camera_stamp`` 形式的 CSV，返回 ``{帧号: 采集时刻}``。"""
    import csv

    out: dict[int, float] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                idx = row.get("frame_index")
                stamp = row.get("camera_stamp")
                if idx is None or stamp is None:
                    continue
                out[int(idx)] = float(stamp)
    except (OSError, ValueError):
        return {}
    return out


@dataclass(frozen=True)
class _VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


class VideoFrameSource:
    """一段 mp4 → 一帧帧带 ``image`` 的 ``Frame``。"""

    is_live = False
    """帧时间线来自录制。标成 ``True`` 会让 now 取墙上时钟，帧龄算出几亿秒。"""

    def __init__(
        self,
        path: str | Path,
        *,
        timestamps: str | Path | None = None,
        start_at: float = 0.0,
        max_frames: int | None = None,
        calibration: ReplayCalibration | None = None,
    ) -> None:
        import cv2  # 局部 import：只有真的要读视频时才需要它

        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"找不到视频：{self.path}")

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(
                f"OpenCV 打不开 {self.path}。缺解码器时会走到这里——"
                f"先确认文件本身能用 ffprobe/播放器打开。"
            )
        self.info = _VideoInfo(
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        self.max_frames = int(max_frames) if max_frames else self.info.frame_count
        self.start_at = float(start_at)

        # 时间戳：显式给的优先，否则找同目录下的 *_timestamps.csv
        ts_path = Path(timestamps) if timestamps else None
        if ts_path is None:
            for cand in sorted(self.path.parent.glob("*timestamps*.csv")):
                ts_path = cand
                break
        self._stamps = _read_timestamps_csv(ts_path) if ts_path else {}
        self.timestamps_path = ts_path
        """实际用到的时间戳文件；``None`` 表示退回标称帧率。"""

        # 标定：显式给的优先，否则尝试从同目录的 EGA 快照里读
        self.calibration = (
            calibration
            if calibration is not None
            else load_ega_calibration(self.path.parent)
        )
        self._i = 0

    # -- 读取 -------------------------------------------------------------
    def read(self) -> Frame | None:
        if self._i >= self.max_frames:
            return None
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        i = self._i
        self._i += 1
        # 带 image 而不是 cascade：这一帧的检测结果要**真跑**出来。
        return Frame(seq=i, captured_at=self.captured_at(i), image=image, cascade=None)

    def captured_at(self, index: int) -> float:
        """第 ``index`` 帧的采集时刻。

        有逐帧时间戳就用它（真实抖动能被滤波器看到）；没有就按标称帧率等距排。
        """
        stamp = self._stamps.get(int(index))
        if stamp is not None:
            return float(stamp)
        return self.start_at + float(index) / max(self.info.fps, 1e-6)

    def close(self) -> None:
        try:
            self._cap.release()
        except Exception:  # pragma: no cover - release 失败不该影响收尾
            pass

    # -- 诊断 -------------------------------------------------------------
    @property
    def frame_count(self) -> int:
        return self.max_frames

    def __repr__(self) -> str:  # pragma: no cover - 诊断用
        src = self.timestamps_path.name if self.timestamps_path else "标称帧率"
        cal = "自带标定" if self.calibration is not None else "无标定"
        return (
            f"VideoFrameSource({self.path.name!r}, "
            f"{self.info.width}x{self.info.height} @{self.info.fps:.1f}fps, "
            f"{self.max_frames} 帧, 时间戳={src}, {cal})"
        )
