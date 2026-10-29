#!/usr/bin/env python3
"""生成回放 fixture。改动场景后重跑本文件即可复现。

    python3 tests/fixtures/_generate_replay.py

产出：
  replay_seq_001/manifest.json   相机、位姿、帧数、节奏、场景说明
  replay_seq_001/cascade.jsonl   每行一帧的**检测级联输出**
  replay_seq_001/truth.jsonl     每帧每台机器人的真实场地坐标

## 为什么装的是级联输出，不是世界坐标

存世界坐标的话，检测和投影两段都被跳过了，回放能测的只剩跟踪和仲裁。
存级联输出（车框、装甲板框、数字）则把投影链留在流水线里**真跑**——
只有"神经网络"那一步被替换掉。本项目手里没有权重，能真跑的就要真跑。

## 相机

**12MP + 12mm 那套的 1/4 降采样版**：内参按 0.25 缩放，畸变系数不变
（畸变系数作用在归一化坐标上，与分辨率无关，所以降采样时它**必须**原样保留）。
视野、光心相对位置完全一致，因此 `_generate.py` 里那张"可视梯形"的结论
在这里同样成立。用 1006×759 而不是 4024×3036 纯粹是为了让测试跑得快：
查找表从 1.2 亿个像素降到 76 万个。

## 场景（96 帧 / 4.8 秒 / 20fps）

| 目标 | 帧区间 | 用意 |
|---|---|---|
| `B3` | 全程可见 | 对照组：证明降级只发生在该降级的槽位上 |
| `B1` | 0–24 可见，之后消失 | 走完 `MEASURED→HOLD→BLIND→MISSING` 整条链 |
| `R2` | 30 起出现 | 中途新建轨迹；且阵营不同，测身份不串味 |

`B1` 在 t=101.25 失去观测，配置里 `lost_predict_s=0.8`、`ttl_s=3.0`，
于是 102.05 转 `BLIND_PREDICTION`、104.25 后转 `MISSING`——都在 96 帧之内。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jg_radar.detect.armor_detector import ArmorHit  # noqa: E402
from jg_radar.detect.digit_classifier import DigitVote  # noqa: E402
from jg_radar.detect.factory import CascadeResult  # noqa: E402
from jg_radar.detect.protocol import DetectionRecord  # noqa: E402
from jg_radar.io.frame_source import cascade_to_dict  # noqa: E402
from jg_radar.transform import CameraModel, Intrinsics, look_at  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "replay_seq_001"

# -- 相机：12MP + 12mm 的 1/4 降采样 --------------------------------------
SCALE = 0.25
WIDTH, HEIGHT = 1006, 759
FX12, CX12, CY12 = 3478.0, 2012.0, 1518.0
_K12_SCALED = [
    [FX12 * SCALE, 0.0, CX12 * SCALE],
    [0.0, FX12 * SCALE, CY12 * SCALE],
    [0.0, 0.0, 1.0],
]
K = _K12_SCALED
DIST = [-0.12, 0.03, 1e-4, -1e-4, 0.0]  # 与分辨率无关，原样保留
EYE = [1400.0, -500.0, 360.0]
TARGET = [1400.0, 750.0, 0.0]

# -- 场景节奏 -------------------------------------------------------------
FRAME_COUNT = 96
DT = 0.05
START_AT = 100.0
"""起始时间**故意不为 0**：任何"把 now 当成从 0 开始的帧计数"的代码
在这里都会露馅。"""

CAR_HEIGHT_CM = 60.0
"""车顶相对地面的高度，用来把地面接触点撑成一个有透视感的框。"""

# 类别表对齐 config/default.yaml 的 detect.car_classes
CLASS_OF = {"B1": 0, "B3": 2, "R2": 6}
DIGIT_CLASS_OF = {"1": 0, "2": 1, "3": 2}


def trajectory(name: str, seq: int) -> tuple[float, float] | None:
    """某台机器人在第 ``seq`` 帧的真实场地坐标。``None`` = 不在场上（或不可见）。

    路径必须落在相机真正看得见的那块梯形里。12mm 架在端线后 5m 时，
    近端只看得见中间一小段、远端才展开到全场——下面的取值都留了余量，
    并且生成时会逐帧断言"框确实在画面内"，写歪了会当场报错而不是
    悄悄产出一份没法用的 fixture。
    """
    t = seq * DT

    if name == "B3":
        # 全程在场上，沿 y 缓慢推进
        return (1500.0 + 60.0 * np.sin(t * 0.7), 450.0 + 260.0 * (seq / FRAME_COUNT))

    if name == "B1":
        if seq > 24:
            return None  # 被遮挡/驶出视野，总之看不见了
        return (880.0 + 9.0 * seq, 880.0)

    if name == "R2":
        if seq < 30:
            return None
        return (1750.0 - 2.0 * (seq - 30), 1020.0 + 3.0 * (seq - 30))

    raise KeyError(name)


def car_bbox(
    camera: CameraModel, R: np.ndarray, t: np.ndarray, pos: tuple[float, float]
) -> tuple[float, float, float, float]:
    """由地面接触点撑出一个车框。

    **底边中点就是地面接触点**：跟踪层取世界坐标用的正是底边中点，因此这里
    让它严格等于投影出的地面点，回放解出来的坐标就能和真值逐帧对齐——
    测试于是可以断言"误差 < N cm"而不只是"状态对了"。

    用"地面点 + 车顶点"两个世界点定出框高，透视缩放就是真的，不是拍脑袋
    给个常数像素高。
    """
    x, y = pos
    world = np.asarray([[x, y, 0.0], [x, y, CAR_HEIGHT_CM]], dtype=np.float64)
    px, valid = camera.project_world(world, R, t)
    if not valid.all():
        raise ValueError(f"({x:.1f}, {y:.1f}) 不在相机前方——路径写错了")
    if not camera.in_image(px, margin=2.0).all():
        raise ValueError(
            f"({x:.1f}, {y:.1f}) 投影到 {np.round(px, 1).tolist()}，落在画面外；"
            "请把轨迹移进可视梯形内"
        )

    u = float(px[0, 0])
    bottom, top = float(px[0, 1]), float(px[1, 1])
    height = bottom - top
    half_w = height * 0.575  # 车比高略宽
    return (
        round(u - half_w, 2),
        round(top, 2),
        round(u + half_w, 2),
        round(bottom, 2),
    )


def armor_boxes(bbox: tuple[float, float, float, float]):
    """在车框里摆两块装甲板（左右各一），返回像素框。

    真实级联里这一级是在**车 ROI 内**跑的，所以框必然落在车框内部——这里
    照这个约束造，下游任何"装甲板应在车框内"的隐含假设都不会被 fixture 破坏。
    """
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = (x2 - x1) * 0.26
    ah = h * 0.42
    ay1 = y1 + h * 0.30
    ay2 = ay1 + ah
    left = (round(x1 + (x2 - x1) * 0.06, 2), round(ay1, 2),
            round(x1 + (x2 - x1) * 0.06 + w, 2), round(ay2, 2))
    right = (round(x2 - (x2 - x1) * 0.06 - w, 2), round(ay1, 2),
             round(x2 - (x2 - x1) * 0.06, 2), round(ay2, 2))
    return [left, right]


def build_frame(
    seq: int, camera: CameraModel, R: np.ndarray, t: np.ndarray
) -> tuple[CascadeResult, dict[str, list[float]]]:
    cars: list[DetectionRecord] = []
    armors: list[ArmorHit] = []
    digits: list[DigitVote] = []
    truth: dict[str, list[float]] = {}

    for name in ("B1", "B3", "R2"):
        pos = trajectory(name, seq)
        if pos is None:
            continue
        bbox = car_bbox(camera, R, t, pos)
        idx = len(cars)
        cars.append(
            DetectionRecord(
                class_id=CLASS_OF[name],
                confidence=0.92,
                bbox=bbox,
                label=name,
            )
        )
        truth[name] = [round(float(pos[0]), 2), round(float(pos[1]), 2)]

        digit = name[1]  # B1 → "1"，R2 → "2"
        for abox in armor_boxes(bbox):
            armors.append(
                ArmorHit(
                    car_index=idx,
                    record=DetectionRecord(class_id=0, confidence=0.85, bbox=abox),
                )
            )
            # digits 与 armors **按下标一一对应**——这是 Cascade.run 的约定：
            # digit.classify(frame, [h.bbox for h in armors])。
            digits.append(
                DigitVote(
                    class_id=DIGIT_CLASS_OF[digit],
                    confidence=0.88,
                    label=digit,
                )
            )

    rois = (
        np.asarray([c.bbox for c in cars], dtype=np.float64).reshape(-1, 4)
        if cars
        else np.zeros((0, 4), dtype=np.float64)
    )
    return (
        CascadeResult(cars=cars, rois=rois, armors=armors, digits=digits),
        truth,
    )


def main() -> None:
    intr = Intrinsics.from_matrix(K, DIST)
    camera = CameraModel(intr, WIDTH, HEIGHT)
    R, t = look_at(EYE, TARGET)

    OUT.mkdir(parents=True, exist_ok=True)

    cascade_lines: list[str] = []
    truth_lines: list[str] = []
    for seq in range(FRAME_COUNT):
        result, truth = build_frame(seq, camera, R, t)
        doc = cascade_to_dict(result)
        doc["seq"] = seq
        cascade_lines.append(json.dumps(doc, ensure_ascii=False, separators=(",", ":")))
        truth_lines.append(
            json.dumps(
                {"seq": seq, "captured_at": START_AT + seq * DT, "robots": truth},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    (OUT / "cascade.jsonl").write_text(
        "\n".join(cascade_lines) + "\n", encoding="utf-8"
    )
    (OUT / "truth.jsonl").write_text("\n".join(truth_lines) + "\n", encoding="utf-8")

    manifest = {
        "name": "replay_seq_001",
        "description": (
            "合成回放：B3 全程可见；B1 在第 25 帧失去观测，用来走完 "
            "MEASURED→OCCLUSION_HOLD→BLIND_PREDICTION→MISSING；"
            "R2 在第 31 帧出现。检测级联的输出是造出来的，投影链是真跑的。"
        ),
        "frame_count": FRAME_COUNT,
        "dt": DT,
        "start_at": START_AT,
        "calibration": {
            "width": WIDTH,
            "height": HEIGHT,
            "K": K,
            "dist": DIST,
            "R": R.tolist(),
            "t": t.tolist(),
            "plane_z": 0.0,
        },
        "note": "由 tests/fixtures/_generate_replay.py 生成，请勿手改。",
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    n_vis = sum(1 for line in truth_lines if json.loads(line)["robots"])
    print(f"✓ {OUT.relative_to(HERE)}  {FRAME_COUNT} 帧 / {n_vis} 帧有目标")
    print(f"  cascade.jsonl {len(cascade_lines)} 行")
    print(f"  truth.jsonl   {len(truth_lines)} 行")


if __name__ == "__main__":
    main()
