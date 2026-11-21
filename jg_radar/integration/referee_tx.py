"""0x0305 雷达位置发送端：把融合结果打成 48 字节定长帧按节拍发出去。

协议原文（2027 高校系列赛通信协议 V1.0.0，表 1-36）：

- 数据段 **48 字节**，24 个 ``uint16``，单位 cm，``雷达→服务器→己方所有选手端``
- 发送频率**上限 5 Hz**
- 「当 x、y 均为 0 时，视为**未发送**此机器人坐标」；坐标超边界由选手端显示在边缘
- 2027 每方只有 **4 台有效位**：重装(1)、3 号步兵(3)、4 号步兵(4)、哨兵(7)，
  工程(2) 和空中(6) 是保留位

2027 的字节偏移（对方在前、己方在后，其余保留位写 0）：

| 偏移 | 内容 |
| --- | --- |
| 0–3 | 对方重装 x、y |
| 4–7 | 保留位 |
| 8–11 / 12–15 | 对方 3 号 / 4 号步兵 x、y |
| 16–19 | 保留位 |
| 20–23 | 对方哨兵 x、y |
| 24–27 | 己方重装 x、y |
| 28–31 | 保留位 |
| 32–35 / 36–39 | 己方 3 号 / 4 号步兵 x、y |
| 40–43 | 保留位 |
| 44–47 | 己方哨兵 x、y |

**这个模块只管载荷和节拍**，帧头/CRC 和串口在 `rmuc.referee_frame` 里——线格式
属于接收包那边（CRC 表本来就在那儿），融合侧不该再抄一份。

``stale_policy`` 决定「有坐标但已经过期」怎么发：

- ``drop``（默认）：写 ``(0,0)``，等于这一拍不报这台机器人
- ``send_stale``：照发旧值，但记进 ``sent_stale`` 计数，供论文和现场排查对账

发旧值大概率被判成 ``d≥1.6m`` 的错误并扣 P，所以默认不发；要证明哪种更好得
用回放跑整条 P(t) 曲线，不能靠直觉。
"""

from __future__ import annotations

from collections import deque
import struct
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping

from jg_radar.types import FIELD_MAX_X_CM, FIELD_MAX_Y_CM, FrameOutput

#: 协议规定的发送频率上限。代码里硬夹到它，配置写大了也只按它发。
MAX_RATE_HZ = 5.0

#: 频率自报要跑多久才稳定。分母是「首帧到现在」，跑 2 秒时 10 帧/2s 会跳出
#: 5.0 以上那种吓人的数——那不是真的超发。低于这个时长，自报只当粗看。
RATE_SETTLE_S = 30.0

#: 任意 1 秒窗口内**最多**允许发出的帧数。这不是「平均频率」，是硬闸门：
#: 节拍器只排程不拦截，主循环晚到一拍时（`due` 已过但只过了不到一个周期）
#: 会出现两帧只隔几十毫秒，一秒钟就能挤进 6 帧。发之前必须查一次窗口。
MAX_FRAMES_PER_WINDOW = 5

#: 默认发送频率。**必须比验收判据（≥4.7Hz）留余量**：设定值和判据都取 4.7 时
#: 是零余量，启动预热吃掉半拍、或者 1 秒窗口被硬闸门挡掉一拍，实测就掉到 4.68
#: 当场判不过——而那并不是链路坏了。4.8 Hz 相对 5 Hz 上限仍有 0.2 的余量，
#: 窗口密度靠 `MAX_FRAMES_PER_WINDOW` 的硬闸门兜住，不会超发。
DEFAULT_RATE_HZ = 4.8

#: 数据段长度（字节）。
PAYLOAD_LEN = 48

STALE_POLICY_DROP = "drop"
STALE_POLICY_SEND_STALE = "send_stale"
STALE_POLICIES = (STALE_POLICY_DROP, STALE_POLICY_SEND_STALE)

#: 2027 有效角色：重装 1、3 号步兵、4 号步兵、哨兵 7。
RADAR_ROLES: tuple[str, ...] = ("1", "3", "4", "7")

#: 对方四台的 x、y 起始偏移。
ENEMY_OFFSETS: dict[str, int] = {"1": 0, "3": 8, "4": 12, "7": 20}

#: 己方四台的 x、y 起始偏移。
OWN_OFFSETS: dict[str, int] = {"1": 24, "3": 32, "4": 36, "7": 44}

#: 对方在 4~7、16~19，己方在 28~31、40~43 各留 4 字节保留位，必须写 0。
RESERVED_RANGES: tuple[tuple[int, int], ...] = ((4, 8), (16, 20), (28, 32), (40, 44))

SIDE_TEAM_NAMES = {"R": "red", "B": "blue"}


def other_side(own_team: str) -> str:
    """己方阵营 → 对方阵营。"""
    side = own_team.upper()[:1]
    if side not in ("R", "B"):
        raise ValueError(f"未知阵营 {own_team!r}，应为 R/B 或 red/blue")
    return "B" if side == "R" else "R"


def build_radar_payload(
    slots: Mapping[str, tuple[float, float] | None],
    *,
    own_team: str = "R",
) -> bytes:
    """把 ``{槽位名: (x_cm, y_cm) | None}`` 打成 48 字节 0x0305 数据段。

    槽位名是融合层的 ``R1`` / ``B7`` 这种整名。没有坐标的机器人写 ``(0,0)``，
    协议里那等于「未发送此机器人坐标」，不是「它在原点」。
    """
    side = own_team.upper()[:1]
    enemy = other_side(side)
    payload = bytearray(PAYLOAD_LEN)
    for role, offset in ENEMY_OFFSETS.items():
        _write_point(payload, offset, slots.get(f"{enemy}{role}"))
    for role, offset in OWN_OFFSETS.items():
        _write_point(payload, offset, slots.get(f"{side}{role}"))
    return bytes(payload)


def _write_point(buffer: bytearray, offset: int, point: tuple[float, float] | None) -> None:
    """写一个 (x, y)。没有坐标就保持 (0,0)。"""
    if point is None:
        return
    x, y = point
    # 坐标超边界时选手端会显示在边缘；这里只防脏值把 uint16 撑爆。
    x = max(0, min(0xFFFF, int(round(float(x)))))
    y = max(0, min(0xFFFF, int(round(float(y)))))
    struct.pack_into("<HH", buffer, offset, x, y)


def parse_radar_payload(payload: bytes, *, own_team: str = "R") -> dict[str, tuple[int, int]]:
    """把数据段解回 ``{槽位名: (x, y)}``，测试和排查用。"""
    if len(payload) != PAYLOAD_LEN:
        raise ValueError(f"0x0305 数据段必须是 {PAYLOAD_LEN} 字节，收到 {len(payload)}")
    side = own_team.upper()[:1]
    enemy = other_side(side)
    result: dict[str, tuple[int, int]] = {}
    for offsets, prefix in ((ENEMY_OFFSETS, enemy), (OWN_OFFSETS, side)):
        for role, offset in offsets.items():
            x, y = struct.unpack_from("<HH", payload, offset)
            result[f"{prefix}{role}"] = (x, y)
    return result


@dataclass
class TxStats:
    """发送统计。现场排查和论文都用它。

    **三态分开记**，不许拿一个数混过去：

    - ``frames_composed``：组出了 48 字节数据段（还没碰串口）
    - ``frames_sent``：**串口写成功**——只有这个数能说「发出去了」
    - ``frames_failed``：组了帧但写失败（掉线/权限/没接设备）

    之前 ``frames_sent`` 在 ``tick()`` 里无条件 +1，写失败的帧也被算进去，
    验收器只读日志时间戳，于是「一根线都没通」能判成 uplink 达标。
    """

    #: 组帧次数（含写失败的那些）。
    frames_composed: int = 0
    #: **写成功**的帧数。协议的 5Hz 说的是这个数的窗口密度。
    frames_sent: int = 0
    #: 组了帧但串口没写成功。
    frames_failed: int = 0
    #: 因为 1 秒窗口已满而主动丢掉的拍（宁可丢一拍，也不超协议上限）。
    throttled: int = 0
    #: 每台机器人因为「没有坐标」或「坐标已过期」而写成 (0,0) 的次数。
    dropped: dict[str, int] = field(default_factory=dict)
    #: ``send_stale`` 策略下真正把过期坐标发出去的次数。
    sent_stale: dict[str, int] = field(default_factory=dict)
    #: 最近一帧里最老的那条坐标的年龄（秒）；没有坐标时为 ``None``。
    last_max_age_s: float | None = None
    #: 任意一个 ``window_s`` 秒窗口里出现过的最多帧数。现场直接看有没有贴线。
    max_frames_in_window: int = 0
    first_tick_at: float | None = None
    first_sent_at: float | None = None
    last_sent_at: float | None = None

    def effective_rate_hz(self, now: float | None = None) -> float:
        """实际发送频率 = 已发帧数 / 运行了多久。

        分母用**墙钟**而不是「首帧到末帧的间隔」：节拍器是相位对齐的，
        末帧之后还欠一个周期，用间隔算会系统性偏低约 1/时长，10 秒窗口下
        看起来像 4.79 Hz，其实一直在按 4.8 Hz 发。

        分子是 ``frames_sent``（写成功），不是组帧数。
        """
        if self.first_tick_at is None or self.frames_sent == 0:
            return 0.0
        end = float(now) if now is not None else (self.last_sent_at or self.first_tick_at)
        span = end - self.first_tick_at
        if span <= 0:
            return 0.0
        return self.frames_sent / span

    def as_dict(self, now: float | None = None) -> dict:
        attempted = self.frames_composed
        end = float(now) if now is not None else (
            self.last_sent_at or self.first_tick_at or 0.0)
        elapsed = 0.0
        if self.first_tick_at is not None:
            elapsed = max(0.0, end - self.first_tick_at)
        return {
            "frames_composed": self.frames_composed,
            "frames_sent": self.frames_sent,
            "frames_failed": self.frames_failed,
            "throttled": self.throttled,
            "write_ok_ratio": (
                None if attempted == 0 else round(self.frames_sent / attempted, 4)
            ),
            "effective_rate_hz": round(self.effective_rate_hz(now), 3),
            # 短窗口下的频率只当粗看：2 秒跑 10 帧会算出 5.0+，现场看着像超发
            "rate_window_s": round(elapsed, 3),
            "rate_settled": elapsed >= RATE_SETTLE_S,
            "max_frames_1s": self.max_frames_in_window,
            "dropped": dict(self.dropped),
            "sent_stale": dict(self.sent_stale),
            "last_max_age_s": (
                None if self.last_max_age_s is None else round(self.last_max_age_s, 3)
            ),
        }


class RobotPositionSender:
    """按固定节拍把融合结果发成 0x0305。

    ``send`` 收到的是 **48 字节数据段**；帧头和 CRC 由调用方（
    ``rmuc.referee_frame``）补，这个类不碰线格式。

    **``send`` 必须回答「写成功了没有」**：只有明确返回 ``True`` 才算发出去。
    返回 ``None`` / ``False`` / 抛异常一律按失败记（``frames_failed``），
    并且**不加** ``frames_sent``——「组了帧」不等于「发出去了」，这条区分
    就是验收器会不会假通过的分界线。

    节拍用「下一次应发的时刻」推进而不是 ``now - last_sent >= period``：
    后者在循环抖动下会系统性变慢，实测频率会低到 4.5 Hz 那种不上不下的数。

    节拍之上还有一道**硬闸门**：发之前查最近 1 秒已发帧数，到
    ``MAX_FRAMES_PER_WINDOW`` 就不发（记 ``throttled``）。节拍器只排程，
    主循环晚到一拍时会出现两帧只隔几十毫秒的挤兑，一秒内能挤出 6 帧。
    """

    def __init__(
        self,
        send: Callable[[bytes], object],
        *,
        own_team: str = "R",
        rate_hz: float = DEFAULT_RATE_HZ,
        stale_policy: str = STALE_POLICY_DROP,
        stale_after_s: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if stale_policy not in STALE_POLICIES:
            raise ValueError(f"未知 stale_policy {stale_policy!r}；可选 {list(STALE_POLICIES)}")
        if stale_after_s <= 0:
            raise ValueError("stale_after_s 必须为正")
        self._send = send
        self.own_team = own_team.upper()[:1]
        self.stale_policy = stale_policy
        self.stale_after_s = float(stale_after_s)
        self.clock = clock
        self.requested_rate_hz = float(rate_hz)
        self.rate_hz = max(0.0, min(MAX_RATE_HZ, float(rate_hz)))
        #: 配置超过协议上限被夹下来的记录，供启动日志提示一次。
        self.clamped = self.rate_hz != self.requested_rate_hz
        self._period = 1.0 / self.rate_hz if self.rate_hz > 0 else float("inf")
        self._next_due: float | None = None
        self._positions: dict[str, tuple[float, float] | None] = {}
        self._observed_at: dict[str, float] = {}
        self.stats = TxStats()
        self.last_payload: bytes | None = None
        #: 最近一次尝试的写结果。日志必须带上它，否则只有组帧时间戳可以糊弄。
        self.last_write_ok = False
        self.last_write_error: str | None = None
        #: 最近 1 秒**写成功**的发送时刻，用来数窗口密度并做硬限。
        self._recent_sends: deque[float] = deque()

    # -- 输入 -------------------------------------------------------------
    def update(self, frame: FrameOutput) -> None:
        """用一帧融合结果刷新「当前该报什么」。

        这里存的是**观测时刻**而不是当场算出的年龄：发布停了以后坐标要自己
        继续变老，否则发送端会拿一份永远 0.3 秒前的快照一直发下去。
        """
        positions: dict[str, tuple[float, float] | None] = {}
        observed: dict[str, float] = {}
        for name, robot in frame.robots.items():
            positions[name] = (
                None if robot.position_cm is None else (float(robot.position_cm[0]), float(robot.position_cm[1]))
            )
            observed[name] = float(robot.observed_at)
        self._positions = positions
        self._observed_at = observed

    # -- 节拍 -------------------------------------------------------------
    def tick(self, now: float | None = None) -> bool:
        """到点就组帧并尝试发出。

        返回值是「这一拍**组了帧**」——写失败也算一次尝试，调用方据此落一行
        带 ``write_ok`` 的日志。真正有没有发出去看 :attr:`last_write_ok`
        和 ``stats.frames_sent``。
        """
        now = self.clock() if now is None else float(now)
        if self.stats.first_tick_at is None:
            self.stats.first_tick_at = now
        if self._period == float("inf"):
            return False
        if self._next_due is None:
            self._next_due = now
        if now < self._next_due:
            return False

        self._expire_window(now)
        # 硬闸门：宁可丢这一拍，也不突破协议的 5Hz 上限。
        if len(self._recent_sends) >= MAX_FRAMES_PER_WINDOW:
            self.stats.throttled += 1
            # **窗口一松就发**，不是白等一个周期。等一个周期的话，主循环卡多久
            # 空洞就多长（卡 0.25s + 一个周期 = 0.46s，贴着 0.5s 判据）；
            # 按「最老那帧满 1 秒」重排，闸门造成的空洞硬性 ≤ 1.0 - 窗口跨度，
            # 4.8Hz 下约 0.38s，与循环卡顿解耦。
            self._next_due = self._recent_sends[0] + 1.0
            return False

        # 落后超过一个周期（主循环卡了）就重新对表，不补发一串。
        if now - self._next_due >= self._period:
            self._next_due = now

        payload, max_age = self._compose(now)
        self.stats.frames_composed += 1
        written, error = self._dispatch(payload)
        self.last_payload = payload
        self.last_write_ok = written
        self.last_write_error = error
        self.stats.last_max_age_s = max_age
        if written:
            self.stats.frames_sent += 1
            self._recent_sends.append(now)
            self.stats.max_frames_in_window = max(
                self.stats.max_frames_in_window, len(self._recent_sends)
            )
            if self.stats.first_sent_at is None:
                self.stats.first_sent_at = now
            self.stats.last_sent_at = now
        else:
            # 没写出去就不占窗口额度，也不改 last_sent_at——统计要能对得上线。
            self.stats.frames_failed += 1
        self._next_due += self._period
        return True

    def _expire_window(self, now: float) -> None:
        """丢掉窗口外的记录。用 ``>= 1.0``：正好 1.0 秒的那帧算下一个窗口。

        与 ``tools/stage3_accept.py`` 里「任意 1 秒窗口」的算法保持一致，
        否则统计口径和验收口径会差一帧。
        """
        while self._recent_sends and now - self._recent_sends[0] >= 1.0:
            self._recent_sends.popleft()

    def _dispatch(self, payload: bytes) -> tuple[bool, str | None]:
        """真正写出去，并回答「写成功了没有」。

        只有回调**明确返回** ``True`` 才算写成功。``None`` 是「没回报」，
        不是「成功」——旧代码就是在这里把写失败当成了发送。
        """
        try:
            result = self._send(payload)
        except Exception as exc:  # noqa: BLE001 —— 掉线/权限/拔线都不能拖垮主循环
            return False, f"{type(exc).__name__}: {exc}"
        if result is True:
            return True, None
        # 回调自己记了原因（拔线/超时）就带上来，比一句"失败了"有用得多
        detail = getattr(self._send, "last_error", None)
        if detail:
            return False, str(detail)
        if result is None:
            return False, "发送回调没有返回成功标记（无法证明写进串口）"
        return False, f"发送回调报告失败：{result!r}"

    # -- 组帧 -------------------------------------------------------------
    def _compose(self, now: float) -> tuple[bytes, float | None]:
        payload = bytearray(PAYLOAD_LEN)
        max_age: float | None = None
        side = self.own_team
        enemy = other_side(side)
        for offsets, prefix in ((ENEMY_OFFSETS, enemy), (OWN_OFFSETS, side)):
            for role, offset in offsets.items():
                name = f"{prefix}{role}"
                point = self._positions.get(name)
                age = self._age_of(name, now)
                if age is not None:
                    max_age = age if max_age is None else max(max_age, age)

                if point is None or age is None:
                    self._bump(self.stats.dropped, name)
                    continue
                if age > self.stale_after_s:
                    if self.stale_policy == STALE_POLICY_DROP:
                        self._bump(self.stats.dropped, name)
                        continue
                    self._bump(self.stats.sent_stale, name)
                _write_point(payload, offset, point)
        return bytes(payload), max_age

    def _age_of(self, name: str, now: float) -> float | None:
        stamp = self._observed_at.get(name)
        if stamp is None:
            return None
        return max(0.0, now - stamp)

    @staticmethod
    def _bump(counter: dict[str, int], name: str) -> None:
        counter[name] = counter.get(name, 0) + 1

    # -- 状态 -------------------------------------------------------------
    def snapshot(self) -> dict:
        data = self.stats.as_dict(now=self.clock())
        data.update({
            "own_team": self.own_team,
            "requested_rate_hz": self.requested_rate_hz,
            "rate_hz": self.rate_hz,
            "stale_policy": self.stale_policy,
            "stale_after_s": self.stale_after_s,
            "last_write_ok": self.last_write_ok,
            "last_write_error": self.last_write_error,
        })
        return data


def slots_within_field(slots: Mapping[str, tuple[float, float] | None]) -> dict[str, tuple[float, float] | None]:
    """把越界坐标当成无效，避免把脏值发到选手端。"""
    cleaned: dict[str, tuple[float, float] | None] = {}
    for name, point in slots.items():
        if point is None:
            cleaned[name] = None
            continue
        x, y = point
        if 0 <= x <= FIELD_MAX_X_CM and 0 <= y <= FIELD_MAX_Y_CM:
            cleaned[name] = (float(x), float(y))
        else:
            cleaned[name] = None
    return cleaned
