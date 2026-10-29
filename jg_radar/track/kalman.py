"""卡尔曼滤波：**图像域管关联，世界域管输出**。

两个滤波器分工不同，不能合并成一个：

``BoxKalman``（像素，``[cx,cy,w,h]`` 各带速度）
    给关联用。关联必须在**图像域**做——遮挡期间车在画面里的位移和它在场地上的
    位移不是一回事：近处 1 米可能跨半个画面，远处 1 米只有几个像素。拿世界域
    距离做门限，远处的目标会被门限整个吃掉，近处的又全都放进来。

``WorldKalman``（cm，``[x,y,vx,vy]``）
    给输出用。裁判系统要的是场地坐标。像素域的噪声投影到场地后是**各向异性**的
    （同样的 1px，在近处是几厘米，在远处是几十厘米），在像素域平滑再投影等于
    把远处目标的噪声放大之后才去滤，白丢精度。

两者都用**恒速模型**（CV）。20fps 下加速度不占主导，而多一个加速度状态会让
协方差在遮挡期间发散得更快——盲区外推本来就是全项目最不可信的一段。

更新一律用 **Joseph 形式**：``P = (I−KH)P(I−KH)ᵀ + KRKᵀ``。教科书的
``P = (I−KH)P`` 在增益接近 1 时会因为两个大数相减而丢掉正定性，协方差一旦
出现负特征值，后面的马氏距离就是负数，门限全乱。
"""

from __future__ import annotations

import numpy as np

__all__ = ["BoxKalman", "WorldKalman", "clamp_dt", "DT_MIN", "DT_MAX"]

DT_MIN = 1.0 / 120.0
"""dt 下界。"""

DT_MAX = 0.5
"""dt 上界。"""

_EPS = 1e-9


def clamp_dt(dt: float) -> float:
    """把 dt 钳到 ``[DT_MIN, DT_MAX]`` 秒。

    仅保留为旧调用方的步长建议工具。predict() 不使用它缩短或补长物理时间：
    predict() 对完整 dt 分段积分，0 秒不推进，负值或非有限值直接拒绝。
    """
    if not np.isfinite(dt):
        return DT_MIN
    return float(min(max(float(dt), DT_MIN), DT_MAX))


def _cv_process_noise(dt: float, q: float, axes: int) -> np.ndarray:
    """分段常加速度白噪声模型（discrete white-noise acceleration）。

    单轴的块是 ``q · [[dt⁴/4, dt³/2], [dt³/2, dt²]]``。交叉项不能省：只给
    位置加噪声、不管速度，等于假设速度绝对准确，滤波器会过度相信外推。
    """
    q = max(float(q), 0.0)
    block = q * np.array(
        [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]], dtype=np.float64
    )
    Q = np.zeros((2 * axes, 2 * axes), dtype=np.float64)
    for i in range(axes):
        Q[i, i] = block[0, 0]
        Q[i, axes + i] = block[0, 1]
        Q[axes + i, i] = block[1, 0]
        Q[axes + i, axes + i] = block[1, 1]
    return Q


class _KalmanCV:
    """恒速模型的公共部分：预测、Joseph 更新、马氏距离。

    子类只给 ``axes`` 和初值。``H`` 是取前 ``axes`` 维的选择矩阵，所以用切片
    就够了，不物化矩阵。
    """

    def __init__(self, axes: int, *, process_noise: float, measurement_noise: float,
                 initial_velocity_var: float) -> None:
        if measurement_noise <= 0:
            raise ValueError("measurement_noise 必须为正，否则 S 会奇异")
        self.axes = int(axes)
        self.dim_x = 2 * self.axes
        self.q = float(process_noise)
        self.r = float(measurement_noise)
        self.x = np.zeros(self.dim_x, dtype=np.float64)
        self.P = np.zeros((self.dim_x, self.dim_x), dtype=np.float64)

    # -- 构造后的初始化 ------------------------------------------------
    def _init_state(self, measured: np.ndarray, velocity_var: float) -> None:
        self.x[: self.axes] = measured
        self.x[self.axes:] = 0.0
        diag = np.concatenate([
            np.full(self.axes, self.r, dtype=np.float64),
            np.full(self.axes, max(float(velocity_var), _EPS), dtype=np.float64),
        ])
        self.P = np.diag(diag)

    # -- 预测 / 更新 ----------------------------------------------------
    def predict(self, dt: float) -> None:
        """按 ``dt`` 秒外推一步。``dt`` 会被钳制。"""
        elapsed = float(dt)
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("预测时间间隔必须是有限的非负数")
        # 限制单步长度，完整积分真实经过的时间；短间隔也不向上补时。
        remaining = elapsed
        while remaining > 0.0:
            h = min(remaining, DT_MAX)
            F = np.eye(self.dim_x, dtype=np.float64)
            F[: self.axes, self.axes:] = np.eye(self.axes) * h
            self.x = F @ self.x
            self.P = F @ self.P @ F.T + _cv_process_noise(h, self.q, self.axes)
            remaining = max(0.0, remaining - h)
        self._symmetrize()

    def update(self, measured: np.ndarray) -> None:
        """用一次观测更新。``measured`` 是长度 ``axes`` 的向量。"""
        z = np.asarray(measured, dtype=np.float64).reshape(-1)
        if z.shape[0] != self.axes:
            raise ValueError(f"观测维度应为 {self.axes}，得到 {z.shape[0]}")

        H = np.zeros((self.axes, self.dim_x), dtype=np.float64)
        H[:, : self.axes] = np.eye(self.axes)

        y = z - H @ self.x
        S = H @ self.P @ H.T
        S = S + np.eye(self.axes) * self.r
        K = np.linalg.solve(S, (self.P @ H.T).T).T

        self.x = self.x + K @ y
        I_KH = np.eye(self.dim_x) - K @ H
        R = np.eye(self.axes) * self.r
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self._symmetrize()

    # -- 查询 -----------------------------------------------------------
    def mahalanobis_sq(self, measured: np.ndarray) -> float:
        """观测相对**当前**分布的马氏距离平方。

        调用方应先 ``predict(dt)`` 再问——否则量的是"观测 vs 上一帧状态"，
        目标一动残差就偏大，门限形同虚设。

        返回 ``inf`` 表示无法计算（协方差非正定）。**不吞掉**这种情况：
        门限把它拒掉是对的，但不能假装它是个正常的大数。
        """
        z = np.asarray(measured, dtype=np.float64).reshape(-1)
        if z.shape[0] != self.axes:
            raise ValueError(f"观测维度应为 {self.axes}，得到 {z.shape[0]}")

        H = np.zeros((self.axes, self.dim_x), dtype=np.float64)
        H[:, : self.axes] = np.eye(self.axes)
        y = z - H @ self.x
        S = H @ self.P @ H.T + np.eye(self.axes) * self.r
        try:
            sol = np.linalg.solve(S, y)
        except np.linalg.LinAlgError:  # pragma: no cover - 需要病态到极点
            return float("inf")
        d2 = float(y @ sol)
        return d2 if np.isfinite(d2) and d2 >= 0 else float("inf")

    @property
    def state(self) -> np.ndarray:
        """完整状态向量的副本。"""
        return self.x.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self.P.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[self.axes:].copy()

    def _symmetrize(self) -> None:
        self.P = 0.5 * (self.P + self.P.T)


class BoxKalman(_KalmanCV):
    """图像域框滤波：``[cx, cy, w, h]`` + 各自的速度。

    速度状态**有用**：它让外推出来的框跟着目标走，而不是原地不动。遮挡 0.3 秒
    时一个 100px/s 的目标已经挪了 30px，不动的框和它的 IoU 会掉到 0.5 以下，
    一恢复检测就可能配错。

    **状态是 ``(cx, cy, w, h)`` 而不是 ``(x1, y1, x2, y2)``**，而检测给的是后者。
    转换只在这一个类的边界上做（``update_bbox`` / ``mahalanobis_bbox`` / ``bbox``），
    外面看不到。中心 + 尺寸的参数化是有意的：位置和尺寸的过程噪声量纲不同，
    用角点参数化会把两者耦合进同一个协方差块，调参时互相牵扯。
    """

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        *,
        process_noise: float = 4.0,
        measurement_noise: float = 9.0,
        initial_velocity_var: float = 1.0e4,
    ) -> None:
        super().__init__(
            4,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            initial_velocity_var=initial_velocity_var,
        )
        self._init_state(self.to_state(bbox), initial_velocity_var)

    # -- 角点 ↔ 中心尺寸 --------------------------------------------------
    @staticmethod
    def to_state(bbox: tuple[float, float, float, float]) -> np.ndarray:
        """``(x1,y1,x2,y2)`` → ``(cx,cy,w,h)``。"""
        x1, y1, x2, y2 = (float(v) for v in bbox)
        return np.array(
            [(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], dtype=np.float64
        )

    def update_bbox(self, bbox: tuple[float, float, float, float]) -> None:
        self.update(self.to_state(bbox))

    def mahalanobis_bbox(self, bbox: tuple[float, float, float, float]) -> float:
        return self.mahalanobis_sq(self.to_state(bbox))

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """当前框 ``x1,y1,x2,y2``（左上/右下，与原图同约定）。

        宽高**钳到 ≥1px**。外推久了 ``w`` 可能算出负数，那会让 IoU 变成负数、
        代价矩阵失去 [0,1] 的值域，匈牙利就会拿一堆无意义的负数去配。这里钳住，
        同时把"这个框已经不可信"交给状态机去说，而不是让几何悄悄崩掉。
        """
        cx, cy, w, h = self.x[:4]
        w = max(float(w), 1.0)
        h = max(float(h), 1.0)
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    @property
    def center(self) -> tuple[float, float]:
        return (float(self.x[0]), float(self.x[1]))

    @property
    def size(self) -> tuple[float, float]:
        return (float(self.x[2]), float(self.x[3]))


class WorldKalman(_KalmanCV):
    """场地坐标滤波：``[x, y, vx, vy]``，单位 cm。"""

    def __init__(
        self,
        position_cm: tuple[float, float],
        *,
        process_noise: float = 900.0,
        measurement_noise: float = 25.0,
        initial_velocity_var: float = 1.0e5,
    ) -> None:
        super().__init__(
            2,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            initial_velocity_var=initial_velocity_var,
        )
        self._init_state(np.asarray(position_cm, dtype=np.float64), initial_velocity_var)

    @property
    def position_cm(self) -> tuple[float, float]:
        return (float(self.x[0]), float(self.x[1]))

    @property
    def velocity_cm_s(self) -> tuple[float, float]:
        """速度（cm/s）。盲区航点打分要用。"""
        return (float(self.x[2]), float(self.x[3]))

    @property
    def position_std_cm(self) -> float:
        """位置标准差（cm），取两轴方差均值的平方根。

        这是 ``RobotSlotOutput.position_std_cm`` 的来源。它必须**随遮挡时长增长**：
        下游就是靠它判断"这个坐标还能不能信"，一个恒定的标准差等于没有信息。
        """
        var = 0.5 * (self.P[0, 0] + self.P[1, 1])
        return float(np.sqrt(max(var, 0.0)))
