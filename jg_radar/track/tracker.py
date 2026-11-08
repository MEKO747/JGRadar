"""``MultiTracker`` —— 把滤波器、关联、状态机、身份投票拼成一条链路。

计划里 ``track/`` 没有列这个文件，但少了它 M5/M6 没有东西可调。它只做编排，
每条规则都住在各自的模块里：这里能读到的应该全是"先做什么后做什么"。

## 一帧的顺序（不能换）

1. **外推**：所有轨迹推到 ``now``。**必须在关联之前**——门限量的是"观测 vs
   预测"，不先预测就退化成"观测 vs 上一帧"，目标一动门限就失效。
2. **关联**：门 → 代价 → 匈牙利。
3. **更新**：配上的喂观测，没配上的记 miss。**先全部更新再统一判状态**：
   边更新边判会让先处理的轨迹用上后处理的轨迹还没发生的效果。
4. **新建**：没配上的检测开新轨迹。
5. **状态 + 输出**：状态机定状态，身份票箱定槽位名。
6. **回收**：本帧之后删掉超过 ttl 的。

## 三条"不上报"的理由

跟踪器不是"有轨迹就报"。下面三种情况它都**不产生输出**：

1. **身份没定**（``label`` 为空，或票权不够集中）。位置报得再准，不知道是谁
   就没有意义——报错人比不报更糟。方案 §十 的"身份只需建立一次"正是这个意思：
   前一两秒的身份是在攒，不是在猜。
2. **轨迹还没确认**（连续命中不足 ``min_hits``）。一帧孤立假检测会开出一条
   候选轨迹，如果它也上报，就会在按槽位名索引的输出字典里把**已经跟了十几帧
   的那条顶掉**——字典是谁后写谁赢，静默地拿假的换真的。候选轨迹一 miss 就死，
   活不过一帧。
3. **观测投不到场地上**（``world_cm is None``）。见 ``_filter_observations``。

三条都是同一个取向：**宁可少报，不可错报。**
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Sequence

import numpy as np

from jg_radar.track.association import (
    Associations,
    associate,
    build_cost_matrix,
    gate_mask,
    raw_jump_matrix,
    same_label_matrix,
)
from jg_radar.track.blind import Waypoint, choose_waypoint, parse_waypoints
from jg_radar.track.identity import IdentityVote, armor_votes, corroborated_votes
from jg_radar.track.state_machine import StateThresholds, resolve_state
from jg_radar.track.types import Track, TrackObservation, TrackStatus
from jg_radar.types import RobotSlotOutput, RobotState, default_slot_names, missing_slot

__all__ = ["MultiTracker"]

_NAN = float("nan")


@dataclass
class _SlotMemory:
    owner_id: int
    trusted: RobotSlotOutput
    challenger_id: int | None = None
    challenger_since: float = 0.0
    challenger_position: tuple[int, int] | None = None
    challenger_evidence_count: int = 0
    challenger_evidence_at: float = float('-inf')
    withdrawn: bool = False


class MultiTracker:
    """多目标跟踪器。输入一帧的观测，输出各槽位的状态。"""

    def __init__(
        self,
        config,
        *,
        slot_names: Sequence[str] | None = None,
        waypoints: Iterable[Sequence] | None = None,
    ) -> None:
        self.cfg = config
        self.slot_names = tuple(slot_names) if slot_names else default_slot_names()
        self.waypoints: tuple[Waypoint, ...] = parse_waypoints(
            waypoints if waypoints is not None else getattr(config, "blind_waypoints", ())
        )
        self.thresholds = StateThresholds.from_config(config)

        self._tracks: list[Track] = []
        self._next_id = 1
        self._dropped_no_world = 0
        self._owners: dict[str, _SlotMemory] = {}
        self._conflicts = 0
        self.last_diagnostics: dict[str, float] = {}

    # -- 只读视图 ---------------------------------------------------------
    @property
    def tracks(self) -> tuple[Track, ...]:
        return tuple(self._tracks)

    def __len__(self) -> int:
        return len(self._tracks)

    # -- 主入口 -----------------------------------------------------------
    def step(
        self, observations: Sequence[TrackObservation], *, now: float
    ) -> dict[str, RobotSlotOutput]:
        """推进一帧。返回 ``{槽位名: 输出}``，**只含身份已定的轨迹**。"""
        obs = self._filter_observations(observations)
        for track in self._tracks:
            track.predict_to(now)

        matches = self._associate(obs)

        matched_tracks = {i for i, _ in matches.pairs}
        matched_dets = {j for _, j in matches.pairs}

        for i, j in matches.pairs:
            self._apply_observation(self._tracks[i], obs[j], now)

        for i, track in enumerate(self._tracks):
            if i not in matched_tracks:
                track.miss()

        for j, o in enumerate(obs):
            if j not in matched_dets:
                self._spawn(o, now)

        for track in self._tracks:
            self._refresh_identity(track, now)

        self._conflicts = 0
        outputs = self._publish(now)
        self._reap(now)

        self.last_diagnostics = {
            "tracks": float(len(self._tracks)),
            "matched": float(len(matches.pairs)),
            "observations": float(len(obs)),
            "reported": float(len(outputs)),
            "dropped_no_world": float(self._dropped_no_world),
            "identity_conflicts": float(self._conflicts),
        }
        return outputs

    def _filter_observations(
        self, observations: Sequence[TrackObservation]
    ) -> list[TrackObservation]:
        """丢掉没能投到场地上的观测。

        **跟踪器只跟踪能定位的目标。** 一个投不到场地上的检测给不出坐标，
        对雷达没有任何用处；更要命的是它会**参与关联**——一个真实轨迹本可以
        配上那个检测，但如果它被这个没有坐标的检测抢走了（代价更低），
        这条轨迹就会拿到一个假的观测。

        "确实看见了但一时投不出来"（目标压在画面边缘、查表无效）在这里表现为
        该帧没有观测 ⇒ 轨迹走遮挡保持。这正是想要的降级，不是缺陷。
        """
        kept = [o for o in observations if o.world_cm is not None]
        self._dropped_no_world = len(observations) - len(kept)
        return kept

    # -- 关联 -------------------------------------------------------------
    def _associate(self, obs: Sequence[TrackObservation]) -> Associations:
        tracks = self._tracks
        if not tracks or not obs:
            return Associations((), tuple(range(len(tracks))), tuple(range(len(obs))))

        cfg = self.cfg
        track_boxes = np.array(
            [t.predicted_bbox for t in tracks], dtype=np.float64
        ).reshape(-1, 4)
        det_boxes = np.array([o.bbox for o in obs], dtype=np.float64).reshape(-1, 4)

        # 跳动门用的是**原始观测中心**（last_center），不是预测框中心。
        # 用预测框等于让滤波器给自己开脱：它已经被带偏时，预测框跟着偏，
        # 跳变看起来就"不大"了。
        last_centers = np.array(
            [t.last_center if t.last_center is not None else (_NAN, _NAN) for t in tracks],
            dtype=np.float64,
        ).reshape(-1, 2)
        det_centers = np.array([o.center for o in obs], dtype=np.float64).reshape(-1, 2)
        raw = raw_jump_matrix(last_centers, det_centers)

        chi2 = np.full((len(tracks), len(obs)), np.nan, dtype=np.float64)
        for i, t in enumerate(tracks):
            if t.box is None:
                continue
            for j, o in enumerate(obs):
                chi2[i, j] = t.box.mahalanobis_center(o.bbox)

        same = same_label_matrix(
            [t.label for t in tracks], [o.car_class for o in obs]
        )

        allowed = gate_mask(
            raw_jump=raw,
            raw_jump_max=float(cfg.raw_jump_px),
            chi2=chi2,
            chi2_max=float(cfg.gate_chi2),
        )

        # 框向上扩张时中心会跟着人移动，底边中点却可能仍贴着车辆。
        # 只有旧框与新框高度包含、底边接近且原始中心跳变仍在限内，才补回配对；
        # 最终仍交给全局匹配和代价门，不把“同槽位”当成强制关联。
        for i, track in enumerate(tracks):
            if track.last_bbox is None:
                continue
            ax1, ay1, ax2, ay2 = track.last_bbox
            area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
            for j, detection in enumerate(obs):
                bx1, by1, bx2, by2 = detection.bbox
                area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
                overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
                bottom = math.hypot((ax1 + ax2 - bx1 - bx2) / 2.0, ay2 - by2)
                tolerance = max(8.0, min(30.0, 0.15 * min(ax2 - ax1, bx2 - bx1)))
                if (min(area_a, area_b) > 0 and overlap / min(area_a, area_b) >= 0.7
                        and bottom <= tolerance and raw[i, j] <= float(cfg.raw_jump_px)
                        and (same is None or same[i, j])):
                    allowed[i, j] = True

        track_world = self._world_array(tracks)
        det_world = self._world_array(obs)
        cost = build_cost_matrix(
            track_boxes,
            det_boxes,
            allowed=allowed,
            w_iou=0.6,
            w_center=1.0,
            track_world=track_world,
            det_world=det_world,
            w_world=0.0 if track_world is None else 0.5,
            center_scale_px=float(getattr(cfg, "assoc_center_scale_px", 120.0)),
            world_scale_cm=float(getattr(cfg, "assoc_world_scale_cm", 300.0)),
        )
        # 单帧身份可错：保留空间关联，在同样接近的候选间优先相同身份。
        # 硬门禁会阻止新证据进入原轨迹，并产生一个新身份的幽灵副本。
        if same is not None:
            cost += 0.2 * (~same)
        return associate(cost, max_cost=float(cfg.max_cost))

    @staticmethod
    def _world_array(items) -> np.ndarray | None:
        """取世界坐标；只要有一个取不到就返回 ``None``（整项不加权）。

        逐对填 NaN 也行，但"有投影的轨迹和没投影的检测混在一帧"本来就少，
        而**整项关掉**比"部分开启"更容易解释代价矩阵里到底有什么。
        """
        out = []
        for it in items:
            if isinstance(it, Track):
                if it.world is None:
                    return None
                out.append(it.world.position_cm)
            else:
                if it.world_cm is None:
                    return None
                out.append(it.world_cm)
        if not out:
            return None
        return np.array(out, dtype=np.float64).reshape(-1, 2)

    # -- 增删改 -----------------------------------------------------------
    def _apply_observation(
        self, track: Track, obs: TrackObservation, now: float
    ) -> None:
        if track.hits >= int(getattr(self.cfg, "min_hits", 2)) and (
            track.state is RobotState.BLIND_PREDICTION
            or track.age(now) > self.thresholds.lost_predict_s
        ):
            track.reacquiring = True
        confirmed = (
            not track.reacquiring
            or track.consecutive_hits + 1 >= self.thresholds.reacquire_hits
        )
        track.observe(
            obs,
            now=now,
            box_noise=float(getattr(self.cfg, "box_measurement_noise_px2", 9.0)),
            world_noise=float(self.cfg.measurement_noise_cm2),
            box_process=float(getattr(self.cfg, "box_process_noise_px2", 4.0)),
            world_process=float(self.cfg.process_noise_cm2),
            min_hits=int(getattr(self.cfg, "min_hits", 2)),
            confirm_observation=confirmed,
        )
        if confirmed:
            track.reacquiring = False
        if obs.direct_identity or not obs.armor_evidence:
            votes = corroborated_votes(obs.car_class, obs.digit)
        else:
            votes = armor_votes(obs.armor_evidence)
        track.identity.decay_to(now)
        track.identity.add_votes(votes, now=now)
        for vote in votes:
            if vote.label not in self.slot_names:
                continue
            track.identity_last_seen[vote.label] = float(now)
            old = track.identity_quality.get(vote.label, vote.weight)
            track.identity_quality[vote.label] = 0.7 * old + 0.3 * min(vote.weight, 1.0)
        self._refresh_identity(track, now)

    def _spawn(self, obs: TrackObservation, now: float) -> Track:
        track = Track(
            track_id=self._next_id,
            created_at=now,
            identity=IdentityVote(
                half_life_s=float(getattr(self.cfg, "identity_half_life_s", 2.0)),
                switch_margin=float(getattr(self.cfg, "identity_switch_margin", 0.15)),
            ),
        )
        self._next_id += 1
        self._tracks.append(track)
        self._apply_observation(track, obs, now)
        return track

    def _refresh_identity(self, track: Track, now: float) -> None:
        """按票箱刷新轨迹的标签。

        票箱空时标签置空——**身份不明的轨迹不该带着上一个标签继续跑**。
        """
        track.identity.decay_to(now)
        best = track.identity.best
        if best is None:
            track.label = ""
            return
        label, _weight = best
        share = track.identity.share(label)
        age = now - track.identity_last_seen.get(label, float("-inf"))
        track.label = label if (share >= float(getattr(self.cfg, "min_identity_share", 0.5))
                                and age <= float(self.cfg.identity_evidence_ttl_s)) else ""

    def _reap(self, now: float) -> None:
        """回收轨迹。

        两条回收路径：从未确认过且已经 miss 的（``DEAD``，活不过一帧），
        以及超过 ``ttl_s`` 的。

        **越过 ttl 的那一帧仍然会发一次 MISSING**（``_publish`` 在 ``_reap``
        之前跑），此后该槽位不再由跟踪器产出。发布层负责把没有轨迹的槽位
        一律填成 MISSING——所以"这条轨迹没了"和"这个槽位没有信息"在对外
        语义上是一致的。
        """
        alive: list[Track] = []
        for track in self._tracks:
            if track.status is TrackStatus.DEAD:
                continue
            if track.age(now) > float(self.cfg.ttl_s):
                track.status = TrackStatus.DEAD
                continue
            alive.append(track)
        self._tracks = alive

    # -- 输出 -------------------------------------------------------------
    def _publish(self, now: float) -> dict[str, RobotSlotOutput]:
        out: dict[str, RobotSlotOutput] = {}
        claims: dict[str, list[Track]] = {}
        for track in self._tracks:
            if track.status not in (TrackStatus.CONFIRMED, TrackStatus.LOST):
                continue
            name = track.label
            if name not in self.slot_names:
                continue
            claims.setdefault(name, []).append(track)

        # 已过期的身份必须明确撤回一次，不能让缓冲继续发布上一次的测量。
        for name in list(self._owners):
            if name not in claims:
                memory = self._owners[name]
                if not memory.withdrawn:
                    out[name] = missing_slot(name, now)
                    memory.withdrawn = True

        for name, candidates in claims.items():
            memory = self._owners.get(name)
            if memory is not None:
                memory.withdrawn = False
            # 一个轨迹已经被更正为其他身份时，立即撤销旧槽位，不能保留旧标签。
            owner = next((t for t in self._tracks if memory and t.track_id == memory.owner_id), None)
            if owner is not None and owner.label and owner.label != name:
                self._owners.pop(name, None)
                memory = None
                owner = None
            def rank(t):
                quality = t.identity_quality.get(name, 0.0) * t.identity.share(name)
                return (t.matched, quality, t.hits, -t.track_id)
            candidates.sort(key=rank, reverse=True)
            chosen = candidates[0]
            quality = lambda t: t.identity_quality.get(name, 0.0) * t.identity.share(name)
            live = [t for t in candidates if t.matched]
            margin = float(self.cfg.identity_conflict_margin)
            ambiguous = (len(live) > 1 and quality(live[0]) - quality(live[1]) < margin
                         and self._track_distance(live[0], live[1]) > float(self.cfg.identity_switch_jump_cm))
            if memory is None:
                if ambiguous:
                    self._conflicts += 1
                    continue
                slot = self._slot(chosen, name, now)
                if slot.state is RobotState.MEASURED:
                    self._owners[name] = _SlotMemory(chosen.track_id, slot)
                out[name] = slot
                continue

            slot = self._slot(chosen, name, now)
            age = max(0.0, now - memory.trusted.observed_at)
            limit = float(self.cfg.identity_switch_jump_cm) + float(self.cfg.identity_max_speed_cm_s) * age
            jump = (math.dist(slot.position_cm, memory.trusted.position_cm)
                    if slot.position_cm is not None and memory.trusted.position_cm is not None else 0.0)
            different = chosen.track_id != memory.owner_id
            needs_check = ambiguous or (slot.state is RobotState.MEASURED and jump > limit)
            if different and owner is not None and owner.matched and chosen.matched:
                needs_check = True
            if needs_check:
                self._conflicts += 1
                # 竞争者必须持续匹配、空间自洽、身份质量明显优于仍可见的现任。
                can_challenge = chosen.matched and not ambiguous and (
                    owner is None or not owner.matched or chosen is owner or
                    quality(chosen) >= quality(owner) + margin)
                if can_challenge and slot.position_cm is not None:
                    if (memory.challenger_id != chosen.track_id or memory.challenger_position is None
                            or math.dist(memory.challenger_position, slot.position_cm) > (
                                float(self.cfg.identity_switch_jump_cm)
                                + float(self.cfg.identity_max_speed_cm_s) * max(0.0, now - memory.challenger_since))):
                        memory.challenger_id = chosen.track_id
                        memory.challenger_since = now
                        memory.challenger_position = slot.position_cm
                        memory.challenger_evidence_count = 0
                        memory.challenger_evidence_at = float('-inf')
                    evidence_at = chosen.identity_last_seen.get(name, float('-inf'))
                    if evidence_at == now and evidence_at > memory.challenger_evidence_at:
                        memory.challenger_evidence_count += 1
                        memory.challenger_evidence_at = evidence_at
                    if (now - memory.challenger_since >= float(self.cfg.identity_challenge_s)
                            and memory.challenger_evidence_count >= int(self.cfg.reacquire_hits)):
                        memory.owner_id = chosen.track_id
                        memory.trusted = slot
                        memory.challenger_id = None
                        out[name] = slot
                        continue
                else:
                    memory.challenger_id = None
                held = self._hold(memory.trusted, now)
                if held is not None:
                    out[name] = held
                continue
            memory.challenger_id = None
            memory.owner_id = chosen.track_id
            if slot.state is RobotState.MEASURED:
                memory.trusted = slot
            out[name] = slot
        return out

    @staticmethod
    def _track_distance(a: Track, b: Track) -> float:
        if a.world is None or b.world is None:
            return float("inf")
        return math.dist(a.world.position_cm, b.world.position_cm)

    def _slot(self, track: Track, name: str, now: float) -> RobotSlotOutput:
        state = self._resolve(track, now)
        track.state = state
        slot = track.to_slot_output(name=name, state=state, now=now,
                                    confidence_half_life_s=float(self.cfg.confidence_half_life_s))
        slot.validate()
        return slot

    def _hold(self, trusted: RobotSlotOutput, now: float) -> RobotSlotOutput | None:
        age = max(0.0, now - trusted.observed_at)
        if age > float(self.cfg.lost_predict_s):
            return None
        std = math.hypot(trusted.position_std_cm or 0.0,
                         float(self.cfg.identity_max_speed_cm_s) * age / 3.0)
        result = replace(trusted, state=RobotState.OCCLUSION_HOLD, is_predicted=True,
                         confidence=trusted.confidence * 0.5 ** (age / max(self.cfg.confidence_half_life_s, 1e-6)),
                         position_std_cm=std, prediction_offset_cm=0.0)
        result.validate()
        return result

    @staticmethod
    def _outranks(candidate: Track, incumbent: Track) -> bool:
        """同槽位冲突时谁赢。

        先看**这一帧有没有在被观测**，再看累计命中数，最后看 id。

        第一项不能省，而且必须排在命中数前面。命中数是**累计**的，幽灵天然
        占便宜：一条跟了很久、后来滤波器跑偏而丢掉关联的轨迹，命中数远大于
        刚重建的那条。只比命中数的话，幽灵会永远把真正在被喂观测的那条压在
        下面——对外持续报 ``OCCLUSION_HOLD``/``BLIND_PREDICTION``，而目标其实
        每一帧都看得见。这是本项目最要挡的那类失效的反面：不是"把预测当成了
        测量"，而是"把测量当成了丢失"。一条正在被观测的轨迹比一条断了粮的
        轨迹更可信，这不需要任何阈值来判。

        两项都打平再比 id（更早建立的赢）。**确定性很重要**——不能依赖
        ``self._tracks`` 的顺序，那会随关联结果的顺序变化，回放两次就可能
        给出不同的人。
        """
        if candidate.matched != incumbent.matched:
            return candidate.matched
        if candidate.hits != incumbent.hits:
            return candidate.hits > incumbent.hits
        return candidate.track_id < incumbent.track_id

    def _resolve(self, track: Track, now: float) -> RobotState:
        matched = track.matched
        if not matched or track.reacquiring:
            self._update_waypoint(track, now)
        if track.reacquiring:
            if track.age(now) > self.thresholds.ttl_s or track.waypoint is None:
                return RobotState.MISSING
            return RobotState.BLIND_PREDICTION
        return resolve_state(
            matched=matched,
            age_s=track.age(now),
            previous=track.state,
            consecutive_hits=track.consecutive_hits,
            has_filter=track.world is not None,
            has_waypoint=track.waypoint is not None,
            thresholds=self.thresholds,
        )

    def _update_waypoint(self, track: Track, now: float) -> None:
        """丢失期间维护盲区航点。

        只有已经有世界坐标的轨迹才谈得上"往哪儿跑"——从没投到场地上的轨迹
        没有位置也没有速度，给它选航点是没有依据的。
        """
        if track.world is None:
            track.waypoint = None
            return
        wp, changed = choose_waypoint(
            self.waypoints,
            position=track.world.position_cm,
            velocity=track.world.velocity_cm_s,
            previous=track.waypoint,
            last_change_at=track.waypoint_since,
            now=now,
            min_dwell_s=float(self.cfg.blind_min_dwell_s),
            decay_cm=float(self.cfg.blind_distance_decay_cm),
            speed_cos_weight=float(self.cfg.blind_speed_cos_weight),
            revisit_penalty=float(self.cfg.blind_revisit_penalty),
        )
        if changed:
            track.waypoint_since = float(now)
        track.waypoint = wp

    def reset(self) -> None:
        """清空全部轨迹。

        **轨迹 id 不重置**：回放两段序列时 id 撞车会让人以为是同一条轨迹。
        """
        self._tracks = []
        self._owners.clear()
        self.last_diagnostics = {}
