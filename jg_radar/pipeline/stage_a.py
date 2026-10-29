"""Stage A：一帧 → 一组观测。**只做转换，不做判断。**

这里不做任何"这个检测靠不靠谱"的判断——那是跟踪层的双闸门该干的事，
它才有历史可以比较。Stage A 的职责窄得多：把检测级联的输出摊成
``TrackObservation``，把像素框投到地面上，然后把整包**冻结在采集时刻**。

## 世界坐标取框的哪个点

**底边中点**，不是框中心。

俯视相机下，穿过框中心的射线比穿过底边中点的更平，打到地面时要多走一段，
于是机器人的坐标被投到它**身后**去。这段偏差随距离增长，实测在
12MP+12mm 的架设下是 87–182cm——而 ``switch_jump_cm`` 才 150cm。
也就是说光是把锚点选错，远处目标的每一次换源都会被判成跳变，滞回门
被迫抬高到 ``reacquire_s``，"丝滑"就静默地没了。

细节见 ``LookupMap.sample_bbox_center`` 的警告与
``tests/test_transform_roundtrip.py::test_the_anchor_pixel_decides_the_position_by_metres``。

## 投不出来的观测照样要传下去

被遮挡、投影落到画面外、查表无效——这些情况下 ``world_cm`` 是 ``None``。
**不能因此把这条观测丢掉**：它在图像域仍然是有效的证据，还能参与关联、
还能攒身份票。丢掉的代价是一条本来可以维持的轨迹被拆散，然后重新建立，
期间身份要重新投票——而身份正是这个项目最慢的那个变量。

## 但"为什么没有坐标"要分开记

``valid[i] == False`` 有三种成因，而它们的含义完全不同：

- 射线**撞在墙或立柱上**（``mesh`` 模式下才有这个结论）——目标大概率还在，
  只是被挡住了。这是"暂时看不见"，策略上应该让轨迹靠预测撑住。
- 射线落到**场地之外**（画面边缘、天空）——那里本来就没有对应的场地点。
- 查表**在这个像素上没有值**。

把它们混成一个 ``None`` 就会丢掉"被挡住了"这条信息，于是"被立柱挡了一下"
和"目标开出了场地"在日志里长得一模一样，而这两件事该触发完全不同的反应。
所以 ``StageA`` 单独数一个 ``occluded``，进 ``diagnostics``。
"""

from __future__ import annotations

import numpy as np

from jg_radar.detect.factory import CascadeResult
from jg_radar.pipeline.mailbox import StageAPacket
from jg_radar.pipeline.timings import Timings
from jg_radar.track.types import TrackObservation
from jg_radar.types import default_slot_names

__all__ = ["StageA"]


class StageA:
    """帧 → ``StageAPacket``。无状态，可重入。"""

    def __init__(
        self,
        lookup,
        *,
        cascade=None,
        timings: Timings | None = None,
        slot_names=None,
    ) -> None:
        self.lookup = lookup
        self.cascade = cascade
        self.slot_names = tuple(slot_names) if slot_names is not None else default_slot_names()
        """帧上没带检测结果时用它自己跑。回放路径带结果，所以这里是 ``None``。"""
        self.timings = timings or Timings()
        self.unlabeled = 0
        """没有整车类别名的观测数。

        类别表配错（比如权重是 EGA 的 0–14 而 YAML 里写的是 shark 的槽位名）
        会让**每一条**观测都拿不到类别，于是身份永远定不下来、雷达站一路
        上报"没有敌人"。这个计数涨起来就是在说这件事。
        """
        self.occluded = 0
        """底边中点那条射线**先撞上了遮挡面**的观测数（墙、立柱）。

        只在 ``mesh`` 模式的表上才可能非零；平面模式的表里 ``occluded``
        按构造全 False，所以这个计数恒为 0 不是"没被挡过"，是"这张表根本
        没有遮挡这个概念"。
        """
        self.frames = 0
        # 表的 ``occluded`` 永不为 None（``LookupMap.__post_init__`` 兜底铺成全
        # False），所以这里可以放心扫一遍。放在构造时扫一次，而不是每帧扫：
        # 12MP 的布尔面扫一次是毫秒级，每帧扫就是白白多花。
        self._counts_occlusion = bool(np.count_nonzero(lookup.occluded))

    def run(self, frame) -> StageAPacket:
        """把一帧转成一个冻结的包。"""
        with self.timings.measure("stage_a"):
            cascade = self._cascade_for(frame)
            observations = self._observations(cascade)
        self.frames += 1
        # 包里只有 ``frame.captured_at`` 这一个时刻，包内所有观测都属于它。
        return StageAPacket(
            frame_seq=int(frame.seq),
            captured_at=float(frame.captured_at),
            observations=observations,
        )

    # -- 内部 -------------------------------------------------------------
    def _cascade_for(self, frame) -> CascadeResult:
        if frame.cascade is not None:
            return frame.cascade
        if self.cascade is None:
            # 两条路都没有 = 我们没有任何依据说"这一帧没有目标"。
            # 静默返回空会伪装成一次正常的空检测，而整个项目最不能容忍的
            # 就是"预测/沉默被当成测量"。宁可炸。
            raise ValueError(
                "这一帧既没带检测结果，Stage A 也没有级联可用；"
                "回放帧应当带 cascade，实时帧应当带 image 并给 Stage A 一个 cascade"
            )
        if frame.image is None:
            raise ValueError("这一帧没有图像，跑不了检测级联")
        with self.timings.measure("detect"):
            return self.cascade.run(frame.image)

    def _observations(self, cascade: CascadeResult) -> tuple[TrackObservation, ...]:
        if not cascade.cars:
            return ()

        boxes = np.asarray([c.bbox for c in cascade.cars], dtype=np.float64).reshape(-1, 4)
        with self.timings.measure("project"):
            world, valid = self.lookup.sample_bbox_ground(boxes)
            # 同一个像素的第二个问题："拿不到坐标，是因为被挡住了吗？"
            # 纯平面模式的表里这个答案是恒 False，那就不必问。
            blocked = (
                self.lookup.occluded_bbox_ground(boxes)
                if self._counts_occlusion
                else np.zeros(len(boxes), dtype=bool)
            )
        self.occluded += int(np.count_nonzero(blocked))

        out: list[TrackObservation] = []
        for i, car in enumerate(cascade.cars):
            identity = self._identity_for(cascade, i)
            if not identity:
                self.unlabeled += 1
            position = (float(world[i, 0]), float(world[i, 1])) if valid[i] else None
            out.append(
                TrackObservation(
                    bbox=tuple(float(v) for v in car.bbox),
                    world_cm=position,
                    confidence=float(car.confidence),
                    car_class=identity,
                    digit=self._digit_for(cascade, i),
                    car_index=i,
                )
            )
        return tuple(out)

    def _identity_for(self, cascade: CascadeResult, car_index: int) -> str:
        """完整车辆身份优先；通用车模型则由同车装甲颜色和数字组合。"""
        label = cascade.cars[car_index].label
        if label in self.slot_names:
            return label
        candidates = []
        for index, hit in enumerate(cascade.armors):
            if hit.car_index != car_index:
                continue
            armor_label = hit.record.label
            if armor_label in self.slot_names:
                candidates.append((float(hit.confidence), armor_label))
                continue
            side = {"red": "R", "blue": "B"}.get(armor_label.lower())
            if side is None or index >= len(cascade.digits):
                continue
            vote = cascade.digits[index]
            if not vote.known:
                continue
            role = "7" if vote.label.upper() == "S" else vote.label
            name = side + role
            if name in self.slot_names:
                candidates.append((min(float(hit.confidence), float(vote.confidence)), name))
        return max(candidates, key=lambda item: item[0])[1] if candidates else ""

    @staticmethod
    def _digit_for(cascade: CascadeResult, car_index: int) -> str:
        """这台车上置信度最高的那个数字读数。读不到就返回空串。

        ``digits[j]`` 与 ``armors[j]`` **按下标一一对应**——这是 ``Cascade.run``
        的约定（``digit.classify(frame, [h.bbox for h in armors])``）。
        按 ``car_index`` 过滤出属于这台车的装甲板，再回头取对应位置的数字。

        数字只用来**印证**身份，指不出槽位（"1" 是 R1 还是 B1？阵营不在数字
        里）。所以读不到不是问题，读错才是——见 ``track.identity``。
        """
        best_label = ""
        best_conf = -1.0
        for j, hit in enumerate(cascade.armors):
            if hit.car_index != car_index or j >= len(cascade.digits):
                continue
            vote = cascade.digits[j]
            if not vote.known:
                continue
            if float(vote.confidence) > best_conf:
                best_conf = float(vote.confidence)
                best_label = "7" if vote.label.upper() == "S" else vote.label
        return best_label

    def diagnostics(self) -> dict[str, float]:
        diag = {
            "stage_a_frames": float(self.frames),
            "stage_a_unlabeled": float(self.unlabeled),
            "stage_a_occluded": float(self.occluded),
            **self.timings.snapshot(),
        }
        if self.cascade is not None:
            for name in ("car", "armor", "digit"):
                detector = getattr(self.cascade, name, None)
                diag[f"{name}_inference_errors"] = float(getattr(detector, "inference_errors", 0))
        return diag
