"""帧源与回放 fixture 的契约。

回放 fixture 是 M6 之后**唯一**能整条流水线跑到底的输入（没有权重、没有相机）。
它要是悄悄坏掉，所有依赖它的测试会一起变成"什么都没发生"——每帧零检测、
槽位全 MISSING，而那看起来和"这一场确实没敌人"一模一样。所以这里把
fixture 的结构钉死。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from jg_radar.io.frame_source import (
    Frame,
    ListFrameSource,
    ReplayCalibration,
    ReplayFrameSource,
    SyntheticFrameSource,
    cascade_from_dict,
    cascade_to_dict,
    iterate,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "replay_seq_001"


# ----------------------------------------------------------------------
# 基本帧源
# ----------------------------------------------------------------------
def test_list_source_walks_off_the_end_and_stays_there():
    src = ListFrameSource([Frame(seq=i, captured_at=float(i)) for i in range(3)])
    assert len(src) == 3
    assert [f.seq for f in iterate(src)] == [0, 1, 2]
    # 到尽头返回 None，**不抛异常**：事件循环靠这个正常收尾
    assert src.read() is None
    assert src.read() is None


def test_synthetic_source_uses_a_fixed_cadence():
    src = SyntheticFrameSource(count=4, dt=0.05, start_at=10.0)
    frames = list(iterate(src))
    assert [f.captured_at for f in frames] == [10.0, 10.05, 10.10, 10.15]
    assert all(f.image is not None for f in frames)
    assert all(f.cascade is None for f in frames)


def test_synthetic_source_rejects_a_zero_period():
    with pytest.raises(ValueError):
        SyntheticFrameSource(count=1, dt=0.0)


def test_a_frame_without_detections_is_distinguishable_from_an_empty_one():
    """``cascade is None``（没带）和 ``cascade.cars == []``（确实没有）是两件事。

    前者要触发"去跑检测器"，后者是**一条观测结果**。混为一谈的话，
    "这一帧画面里真的没有车"会被当成"还没检测"，于是一帧真实的空检测
    被静默跳过——而那正是最需要被记录下来的降级场景。
    """
    assert not Frame(seq=0, captured_at=0.0).has_detections
    empty = Frame.with_cascade(0, 0.0, cascade_from_dict({}))
    assert empty.has_detections
    assert empty.cascade.cars == []


# ----------------------------------------------------------------------
# 级联输出的序列化
# ----------------------------------------------------------------------
def test_cascade_round_trips_through_json():
    original = cascade_from_dict(
        {
            "cars": [
                {"class_id": 2, "confidence": 0.9, "bbox": [10.0, 20.0, 40.0, 70.0],
                 "label": "B3"},
                {"class_id": 6, "confidence": 0.8, "bbox": [50.0, 21.0, 80.0, 71.0],
                 "label": "R2"},
            ],
            "armors": [
                {"car_index": 0, "record": {"class_id": 0, "confidence": 0.85,
                                            "bbox": [12.0, 30.0, 20.0, 50.0]}},
                {"car_index": 1, "record": {"class_id": 0, "confidence": 0.85,
                                            "bbox": [52.0, 30.0, 60.0, 50.0]}},
            ],
            "digits": [
                {"class_id": 2, "confidence": 0.88, "label": "3"},
                {"class_id": 1, "confidence": 0.88, "label": "2"},
            ],
        }
    )
    back = cascade_from_dict(json.loads(json.dumps(cascade_to_dict(original))))

    assert [c.label for c in back.cars] == ["B3", "R2"]
    assert [c.class_id for c in back.cars] == [2, 6]
    assert [h.car_index for h in back.armors] == [0, 1]
    assert [v.label for v in back.digits] == ["3", "2"]
    # rois 由车框重建
    assert back.rois.shape == (2, 4)
    assert np.allclose(back.rois[1], [50.0, 21.0, 80.0, 71.0])


def test_the_roi_matrix_is_rebuilt_not_stored():
    """``rois`` 是车框的冗余视图，序列化里**不存**——存了就会有不一致的可能。"""
    doc = cascade_to_dict(cascade_from_dict(
        {"cars": [{"class_id": 0, "confidence": 1.0, "bbox": [1.0, 2.0, 3.0, 4.0]}]}
    ))
    assert "rois" not in doc


def test_empty_cascade_has_an_empty_roi_matrix():
    result = cascade_from_dict({})
    assert result.cars == [] and result.armors == [] and result.digits == []
    assert result.rois.shape == (0, 4)
    assert result.empty


# ----------------------------------------------------------------------
# 回放目录：自描述
# ----------------------------------------------------------------------
def test_replay_manifest_is_self_describing():
    """测试不该在代码里重复一遍相机会数——那正是"改了 fixture 忘了改测试"的入口。"""
    src = ReplayFrameSource(FIXTURE)
    cal = src.calibration

    assert src.frame_count == 96
    assert src.dt == pytest.approx(0.05)
    # 起始时间不为 0：任何"把 now 当成帧计数"的代码在这里都会露馅
    assert src.start_at == pytest.approx(100.0)
    assert (cal.width, cal.height) == (1006, 759)
    assert cal.K.shape == (3, 3) and cal.R.shape == (3, 3) and cal.t.shape == (3,)
    assert cal.K[0, 0] == pytest.approx(3478.0 * 0.25)
    # 畸变系数作用在归一化坐标上，与分辨率无关 —— 降采样时必须原样保留
    assert cal.dist.tolist() == [-0.12, 0.03, 1e-4, -1e-4, 0.0]


def test_replay_frames_carry_detections_and_a_monotonic_clock():
    src = ReplayFrameSource(FIXTURE)
    frames = list(iterate(src))
    assert len(frames) == 96
    assert all(f.has_detections for f in frames), (
        "回放里的**每一帧**都必须带检测结果，包括空的那几帧"
    )
    times = [f.captured_at for f in frames]
    assert times == sorted(times)
    assert times[0] == pytest.approx(100.0)
    assert times[-1] == pytest.approx(100.0 + 95 * 0.05)


def test_replay_cascade_matches_the_truth_file():
    """fixture 里的检测框必须和真值坐标对得上——自洽性检查。

    这条挂了说明**生成器**坏了，而不是流水线坏了。分开来报，省得以后
    在一堆流水线断言里找"到底是哪一段错了"。
    """
    src = ReplayFrameSource(FIXTURE)
    lookup = src.build_lookup()
    truth = {
        json.loads(line)["seq"]: json.loads(line)["robots"]
        for line in (FIXTURE / "truth.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    seen = 0
    for frame in iterate(ReplayFrameSource(FIXTURE)):
        if not frame.cascade.cars:
            continue
        boxes = np.asarray([c.bbox for c in frame.cascade.cars], dtype=np.float64)
        world, ok = lookup.sample_bbox_ground(boxes)
        assert ok.all(), "车框底边中点必须落在查找表的有效区域内"
        assert set(truth[frame.seq]) == {c.label for c in frame.cascade.cars}
        for car, got in zip(frame.cascade.cars, world):
            want = np.asarray(truth[frame.seq][car.label], dtype=np.float64)
            assert np.hypot(*(got - want)) < 8.0, (
                f"第 {frame.seq} 帧 {car.label}：投影回 {np.round(got, 1)}，"
                f"真值 {want}；fixture 生成器与相机参数对不上了"
            )
            seen += 1
    assert seen > 150, "fixture 里的目标太少了，覆盖不到整条状态链"


def test_replay_covers_every_degradation_step():
    """场景必须**故意**包含：全程可见的、中途消失的、中途出现的。

    这三个是 M6 要验的全部内容。少了任何一个，测试都会"通过"但什么都没测到。
    """
    present: dict[str, int] = {}
    for frame in iterate(ReplayFrameSource(FIXTURE)):
        for car in frame.cascade.cars:
            present[car.label] = present.get(car.label, 0) + 1

    assert present["B3"] == 96, "B3 应当全程可见"
    assert present["B1"] == 25, "B1 应当观测 25 帧后消失"
    assert present["R2"] == 66, "R2 应当中途出现并留到结束"


def test_replay_digits_are_index_parallel_to_armors():
    """``digits[i]`` 对应 ``armors[i]``——这是 ``Cascade.run`` 的约定。

    fixture 要是把顺序弄乱了，数字就会**贴到别的车上**：B2 的数字投给 B1，
    身份投票于是持续投出一个错误结论，而且看起来完全合理。这种错没人能
    从输出上发现，只能在这里挡住。
    """
    checked = 0
    for frame in iterate(ReplayFrameSource(FIXTURE)):
        result = frame.cascade
        assert len(result.digits) == len(result.armors)
        for i, hit in enumerate(result.armors):
            car = result.cars[hit.car_index]
            assert result.digits[i].label == car.label[1], (
                f"第 {frame.seq} 帧：装甲板 {i} 属于 {car.label}，"
                f"但对应的数字读数是 {result.digits[i].label!r}"
            )
            # 装甲板框必须落在它那台车的框里 —— 真级联是在车 ROI 内跑的
            ax1, ay1, ax2, ay2 = hit.record.bbox
            cx1, cy1, cx2, cy2 = car.bbox
            assert cx1 <= ax1 and ax2 <= cx2 and cy1 <= ay1 and ay2 <= cy2
            checked += 1
    assert checked > 300


def test_replay_without_a_cascade_file_yields_explicitly_empty_frames(tmp_path):
    """只有 manifest 时退化成"整段没有任何目标"——测全 MISSING 的降级场景。"""
    src = ReplayFrameSource(FIXTURE)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "frame_count": 5,
                "dt": 0.05,
                "start_at": 0.0,
                "calibration": src.calibration.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    frames = list(iterate(ReplayFrameSource(tmp_path)))
    assert len(frames) == 5
    assert all(f.has_detections and f.cascade.empty for f in frames)


def test_replay_rejects_a_directory_without_a_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReplayFrameSource(tmp_path)


def test_replay_can_be_read_twice_independently():
    """两个独立的源互不干扰——测试之间不该因为共享游标而串味。"""
    a = list(iterate(ReplayFrameSource(FIXTURE)))
    b = list(iterate(ReplayFrameSource(FIXTURE)))
    assert len(a) == len(b) == 96
    assert [f.seq for f in a] == [f.seq for f in b]


def _lookup_from(cal):
    from jg_radar.transform import CameraModel, Intrinsics, build_lookup_map

    camera = CameraModel(
        intrinsics=Intrinsics.from_matrix(cal.K, cal.dist),
        width=cal.width,
        height=cal.height,
    )
    return build_lookup_map(camera, cal.R, cal.t, cal.plane_z)


def test_replay_calibration_survives_a_json_round_trip():
    """标定写进 manifest 再读回来，必须给出**逐位相同**的查找表。

    差一点点的话，症状是"世界坐标整体偏一点"——最难发现的那一类。
    """
    cal = ReplayFrameSource(FIXTURE).calibration
    assert np.array_equal(_lookup_from(cal).coords_cm,
                          _lookup_from(ReplayCalibration.from_dict(cal.to_dict())).coords_cm,
                          equal_nan=True)


def test_replay_lookup_signature_notices_a_moved_camera():
    """相机挪了 1cm，签名就必须变——否则会悄悄用着一张过期的表。"""
    cal = ReplayFrameSource(FIXTURE).calibration
    moved = dataclasses.replace(cal, t=cal.t + np.array([1.0, 0.0, 0.0]))
    assert _lookup_from(cal).signature != _lookup_from(moved).signature


def test_frame_equality_is_disabled_on_purpose():
    """``Frame`` 上有 ndarray，dataclass 自动生成的 ``__eq__`` 会退化成
    逐元素比较并返回数组，于是 ``assert a == b`` 直接抛
    "truth value of an array is ambiguous"——一个只在断言里炸、
    报错信息完全指不到原因的比较。所以 ``eq=False``，比较退化为身份比较。"""
    a = Frame(seq=0, captured_at=0.0, image=np.zeros((2, 2, 3), dtype=np.uint8))
    b = Frame(seq=0, captured_at=0.0, image=np.zeros((2, 2, 3), dtype=np.uint8))
    assert a is not b and a != b
    assert a == a
