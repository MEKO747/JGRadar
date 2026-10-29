"""配置模式 —— 全部用 dataclass 定义，未知键一律报错。

为什么不用裸 dict：
  TCR 的 ``params.yaml`` 里有一处重复的 ``camera:`` 键，YAML 规范下后者静默覆盖
  前者，调试时极难发现。本模块要求每个键都对应一个 dataclass 字段，
  多一个键就抛 ``ConfigError``；``loader`` 另外禁止重复键。
"""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field as dc_field,
    fields as dc_fields,
    is_dataclass,
)
from typing import Any, Mapping, get_type_hints

from jg_radar.types import SCHEMA_VERSION


class ConfigError(ValueError):
    """配置格式或取值错误。"""


# --------------------------------------------------------------------------
# 通用构建器
# --------------------------------------------------------------------------
def _coerce(tp: Any, value: Any, path: str) -> Any:
    """按注解类型转换 YAML 载入的裸值。"""
    if is_dataclass(tp):
        return _build(tp, value, path)

    origin = getattr(tp, "__origin__", None)

    if origin is tuple:
        args = tp.__args__
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: 期望列表，得到 {type(value).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:  # tuple[T, ...]
            return tuple(_coerce(args[0], v, f"{path}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"{path}: 期望 {len(args)} 个元素，得到 {len(value)}")
        return tuple(_coerce(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, value)))

    if origin is list:
        (arg,) = tp.__args__ or (Any,)
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: 期望列表")
        return [_coerce(arg, v, f"{path}[{i}]") for i, v in enumerate(value)]

    if origin is dict or origin is Mapping:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: 期望映射")
        return dict(value)

    # Optional[X] → Union[X, None]
    if origin is not None and type(None) in getattr(tp, "__args__", ()):
        if value is None:
            return None
        inner = [a for a in tp.__args__ if a is not type(None)][0]
        return _coerce(inner, value, path)

    if tp in (int, float) and isinstance(value, bool):
        raise ConfigError(f"{path}: 布尔值不能当数值用")
    if tp is float and isinstance(value, (int, float)):
        return float(value)
    if tp is int and isinstance(value, (int, float)):
        return int(value)
    if tp is bool and not isinstance(value, bool):
        raise ConfigError(f"{path}: 期望布尔值")
    if tp is str and not isinstance(value, str):
        raise ConfigError(f"{path}: 期望字符串")
    return value


def _build(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: 期望映射，得到 {type(data).__name__}")

    hints = get_type_hints(cls)
    known = {f.name for f in dc_fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"{path}: 未知配置项 {unknown}；本组可用键 = {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for f in dc_fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce(hints[f.name], data[f.name], f"{path}.{f.name}")
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


# --------------------------------------------------------------------------
# 各组配置
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CameraConfig:
    """相机内参与采集参数。相机未选型前保持占位值。"""

    name: str = "unselected"
    width: int = 4024
    height: int = 3036
    fps: float = 20.0
    K: tuple[tuple[float, float, float], ...] = (
        (3478.0, 0.0, 2012.0),
        (0.0, 3478.0, 1518.0),
        (0.0, 0.0, 1.0),
    )
    dist: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    """OpenCV 顺序 k1 k2 p1 p2 k3。"""
    pixel_format: str = "BayerRG8"


@dataclass(frozen=True)
class FieldConfig:
    """场地网格与坐标系。"""

    mesh_path: str = ""
    """PLY/STL/OBJ 三角网格路径；空则退化为纯地面平面模式。"""
    mesh_to_field: tuple[tuple[float, float, float, float], ...] = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    """4×4 刚体变换，CAD/网格坐标 → 场地坐标（cm）。

    **必须是 4 行。** 这里原来写的是 3×4，而这个值一路喂给
    ``RawMesh.transformed``，那里只做 ``reshape(4, 4)``——少一行就在 reshape
    处抛"cannot reshape array of size 12 into shape (4,4)"，跟"配置里少写了
    一行"这个真实原因完全对不上。``pipeline.build.mesh_to_field_matrix`` 现在
    两种都收，但默认值本身不该是错的。
    """
    ground_plane_z: float = 0.0
    """地面平面高度（场地坐标 cm）。"""
    keypoints_path: str = ""
    unit_scale: float = 1.0
    """网格单位 → cm 的缩放系数。"""
    size_x_cm: int = 2800
    size_y_cm: int = 1500
    roles: tuple[str, ...] = ("1", "2", "3", "4", "6", "7")
    sides: tuple[str, ...] = ("R", "B")


@dataclass(frozen=True)
class TransformConfig:
    """相机外参与投影链门限。"""

    R: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    """世界（场地 cm）→ 相机 的旋转。"""
    t: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """世界 → 相机的平移，单位 cm。"""
    projection_mode: str = "ground_plane"
    """``ground_plane`` | ``mesh``。"""
    lookup_map_path: str = ""
    """预计算像素→世界查找表（.npz）。运行时主路径。"""
    reproj_rms_max_px: float = 3.0
    reproj_max_px: float = 8.0
    roundtrip_max_cm: float = 2.0
    reproj_rms_px: float | None = None
    """标定实测残差，写入配置后作为变换噪声下限。"""


@dataclass(frozen=True)
class DetectConfig:
    """检测层。后端可插拔。"""

    backend: str = "stub"
    """``stub`` | ``empty`` | ``onnx``。``blob`` 仅供测试与基准线，不经配置构造。"""
    car_model: str = ""
    armor_model: str = ""
    digit_model: str = ""
    digit_backend: str = ""
    car_img_size: int = 1280
    armor_img_size: int = 192
    digit_img_size: int = 48
    car_conf: float = 0.35
    armor_conf: float = 0.4
    digit_conf: float = 0.5
    iou: float = 0.45
    max_det: int = 32
    roi_batch: int = 16
    min_box_px: float = 4.0

    # ---- 类别表：换一份权重只改这几行，不动任何代码 ----
    #
    # 键是**后端输出的 class_id**（YAML 里写成 0 还是 "0" 都行），值是语义标签。
    # 表里没有的 id 会被整级丢弃——网络输出的整数只有配上训练时的类别表才有
    # 意义，表外的 id 是噪声，而不是"另一个类"。
    #
    # 所以这张表同时干两件事：**翻译**与**过滤**。留空表示这一级不过滤。
    #
    # 三家的约定互不相同（EGA 用 0–14 的颜色编码，shark 直接给 B1–B5/R1–R5，
    # TCR 是 15 类），照抄任何一家都是在赌自己手里是哪份权重。
    #
    # ⚠️ 这三张表是**封闭集合**：叠加配置时写出来就是**全集**，不与上一层
    # 逐键合并（见 ``loader`` 模块文档）。因为它们是白名单——少写一个键不是
    # "这一项保持默认"，而是"这一项**仍然被接受**"。要追加写 ``car_classes+``，
    # 要清空写 ``{}``。
    car_classes: dict[int, str] = dc_field(default_factory=dict)
    armor_classes: dict[int, str] = dc_field(default_factory=dict)
    digit_classes: dict[int, str] = dc_field(default_factory=dict)


@dataclass(frozen=True)
class TrackConfig:
    """跟踪层：双闸门阈值必须来自实测。"""

    raw_jump_px: float = 120.0
    """原始观测跳变门（像素）。"""
    process_noise_cm2: float = 900.0
    """位置过程噪声方差（cm²）。"""
    measurement_noise_cm2: float = 25.0
    min_hits: int = 2
    """新轨迹要连续命中几次才确认并上报。防止单帧假检测抢走已确认轨迹的槽位。"""
    lost_predict_s: float = 0.8
    """超过此时长从 OCCLUSION_HOLD 转入 BLIND_PREDICTION。"""
    ttl_s: float = 3.0
    """超过此时长彻底 MISSING。"""
    reacquire_hits: int = 2
    """从 BLIND_PREDICTION 恢复所需连续命中数。单帧假检测不该拽回一条盲区轨迹。"""
    max_cost: float = 1.2
    gate_chi2: float = 9.49
    """χ² 95%, 4 自由度。"""
    box_process_noise_px2: float = 4.0
    box_measurement_noise_px2: float = 9.0
    """图像域滤波器的过程/测量噪声（px²）。与上面两个 cm² 的**量纲不同**，
    不能互相套用——远处 1px 是几十厘米，近处 1px 只有几厘米。"""
    assoc_center_scale_px: float = 120.0
    assoc_world_scale_cm: float = 300.0
    """代价里距离项的衰减尺度。"""
    identity_half_life_s: float = 2.0
    """身份票权的半衰期。太长则身份锁死，太短则每帧都在重新认人。"""
    identity_switch_margin: float = 0.15
    """挑战者票权要超出在职者这么多比例才允许改标签。防止读数的反复横跳
    变成对外身份的反复横跳。"""
    confidence_half_life_s: float = 1.0
    """丢失期间置信度的半衰期。"""
    min_identity_share: float = 0.5
    """最高票占比低于它就认为身份未定，不上报。"""
    blind_speed_cos_weight: float = 1.0
    blind_distance_decay_cm: float = 400.0
    blind_min_dwell_s: float = 0.5
    blind_revisit_penalty: float = 0.3
    blind_waypoints: tuple[tuple[str, float, float], ...] = ()
    """盲区候选落点 ``(名字, x_cm, y_cm)``。为空则长时丢失直接判 MISSING——
    没有候选就先说不知道。"""


@dataclass(frozen=True)
class FuseConfig:
    """融合仲裁。阶梯顺序即优先级，从高到低。"""

    priority: tuple[str, ...] = (
        "radio_fresh",
        "vision_measured",
        "radio_stale",
        "vision_occlusion_hold",
        "vision_blind",
    )
    radio_fresh_s: float = 0.2
    radio_stale_s: float = 10.0
    vision_watchdog_s: float = 0.5
    dwell_s: float = 0.3
    """挑战者必须连续满足条件的时长。"""
    switch_jump_cm: float = 150.0
    """切换时允许的最大位置跳变。"""
    reacquire_s: float = 0.8
    """跳变过大时，重新夺取所需的连续一致样本时长。"""

    # ---- 第二条轴：位置误差上界（见 fuse/arbiter.py 模块文档）----
    max_robot_speed_cm_s: float = 500.0
    """机器人最大速度（cm/s）。用来把"信息年龄"换算成"最多偏了多少"。

    500cm/s = 5m/s，覆盖 RMUC 地面机器人的实际上限。这个值**宁可取大**：
    取小了会把还堪用的坐标误判成过期，症状是明明有数据却频繁报 MISSING。
    """
    max_position_error_cm: float = 500.0
    """位置误差上界超过它，这条坐标就不作数（cm）。

    与上一项相除就是**实测坐标的保质期**：500 / 500 = **1.0 秒**。
    选 1 秒是因为它落在两个已知约束之间——无线电的 ``radio_stale_s``(10s)
    远大于它（9 秒前的报文必然被否），而视觉的看门狗 ``vision_watchdog_s``
    (0.5s) 远小于它（视觉喂进来的实测坐标不会因为这一条被否）。

    注意它**不是**精度门限，是**可用性**门限：问的是"这条坐标现在还值不值得
    报"，不是"它够不够准"。像 ``radio_stale_s`` 一样必须由实测推导，
    但它的实测方法就是量机器人速度，不需要专门的噪声标定设备。
    """


@dataclass(frozen=True)
class PipelineConfig:
    max_commit_age_s: float = 0.15
    """提交时若帧龄超过此值，丢弃而不是发旧数据。"""
    mailbox_size: int = 2
    calibration_epoch: int = 0


@dataclass(frozen=True)
class IOConfig:
    frame_source: str = "synthetic"
    """``synthetic`` | ``replay`` | ``video`` | ``camera``。"""
    replay_path: str = ""
    """回放**目录**（含 manifest.json）：检测结果已经算好，投影链真跑。"""
    video_path: str = ""
    """视频**文件**（mp4 等）：帧带 ``image``，每帧的检测**真跑**。
    标定从视频同目录的 ``calibration_snapshot.yaml`` / ``extrinsics.txt``
    自动读取，见 ``io/video_source.py``。"""
    video_timestamps: str = ""
    """逐帧采集时刻的 CSV。留空则自动找同目录下的 ``*timestamps*.csv``；
    再找不到就按标称帧率等距排——那会让滤波器丢掉真实抖动。"""
    sink_path: str = ""
    camera_index: int = 0
    """相机枚举序号，从 0 开始；camera_serial 非空时优先按序列号选。"""
    camera_serial: str = ""
    """可选设备序列号，避免多相机时枚举顺序变化。"""
    camera_width: int = 0
    """可选采集宽度；0 表示保持相机默认值。"""
    camera_height: int = 0
    """可选采集高度；0 表示保持相机默认值。"""


@dataclass(frozen=True)
class DiagnosticsConfig:
    ema_alpha: float = 0.1
    jsonl_path: str = ""
    overlay_every_n_frames: int = 30


@dataclass(frozen=True)
class RadarConfig:
    """根配置。"""

    schema_version: str = SCHEMA_VERSION
    camera: CameraConfig = dc_field(default_factory=CameraConfig)
    field: FieldConfig = dc_field(default_factory=FieldConfig)
    transform: TransformConfig = dc_field(default_factory=TransformConfig)
    detect: DetectConfig = dc_field(default_factory=DetectConfig)
    track: TrackConfig = dc_field(default_factory=TrackConfig)
    fuse: FuseConfig = dc_field(default_factory=FuseConfig)
    pipeline: PipelineConfig = dc_field(default_factory=PipelineConfig)
    io: IOConfig = dc_field(default_factory=IOConfig)
    diagnostics: DiagnosticsConfig = dc_field(default_factory=DiagnosticsConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RadarConfig":
        return _build(cls, data, "config")

    def slot_names(self) -> tuple[str, ...]:
        """按配置产出槽位名，顺序固定。"""
        return tuple(
            f"{side}{role}" for side in self.field.sides for role in self.field.roles
        )
