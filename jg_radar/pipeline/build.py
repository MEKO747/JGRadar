"""按配置装配整条流水线。**换相机、换模型、换权重都只改 YAML。**

装配顺序里有一个**必须做对**的地方：查找表从哪来。

- 配置里给了 ``transform.lookup_map_path`` → 直接加载预计算好的表（运行时主路径）
- 没给 → 现场用 ``camera`` + ``transform`` 两个段算一张

现场算的那条路是给测试和回放用的。**两者不能混**：拿配置里的相机去算一张表、
却拿另一套标定去投影，症状是"坐标整体偏一点"——没有报错、没有崩溃，
只是每个位置都差个几十厘米。所以回放路径必须把**回放目录自己的标定**
传进来（见 ``ReplayFrameSource.build_lookup``），别让它去读 ``config/camera.yaml``。

## 加载预计算表时**要验签名**

``LookupMap.signature`` 里揉进了分辨率、内参、畸变、外参、地面高度、投影模式
和网格内容。以前这个签名算出来了却没人比对——"改了标定忘了重算表"于是照样
能跑，症状是坐标整体偏几十厘米，没有异常、没有告警。现在加载时拿配置现算
一遍签名，对不上就拒绝启动。

## 投影模式

``transform.projection_mode`` 决定这张表怎么算：

- ``ground_plane``：与水平面求交，不需要网格
- ``mesh``：与 ``field.mesh_path`` 指的场地网格求交，高地/坡道上的目标才落得对；
  射线打在墙或立柱上时**标记为遮挡**而不是给一个地面坐标

模式名写错、或者选了 ``mesh`` 却没有网格，都在这里**直接报错**。静默退回平面
模式的话，得到的是一张看起来完全正常、实际把高地目标投到几米外的表。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jg_radar.io.frame_source import FrameSource, ReplayFrameSource, SyntheticFrameSource
from jg_radar.pipeline.event_loop import EventLoop, ThreadedEventLoop
from jg_radar.pipeline.stage_a import StageA
from jg_radar.pipeline.stage_b import StageB
from jg_radar.pipeline.timings import Timings
from jg_radar.track import MultiTracker

__all__ = [
    "build_loop",
    "build_lookup",
    "build_source",
    "build_stage_a",
    "build_stage_b",
    "load_field_mesh",
    "lookup_geometry",
    "mesh_to_field_matrix",
]


def mesh_to_field_matrix(rows) -> np.ndarray:
    """``field.mesh_to_field`` → 4×4。

    配置里可能是 4×4，也可能是 3×4（``FieldConfig`` 的默认值就是 3×4 的
    单位阵加一行平移）。两种都收下：``RawMesh.transformed`` 只会
    ``reshape(4,4)``，喂给它 12 个数会直接抛一个看不出原因的异常。
    """
    m = np.asarray(rows, dtype=np.float64)
    if m.shape == (3, 4):
        m = np.vstack([m, np.array([[0.0, 0.0, 0.0, 1.0]])])
    return m.reshape(4, 4)


def load_field_mesh(config):
    """按 ``field`` 段载入场地网格。``mesh_path`` 为空返回 ``None``。

    ``.npz`` 走 ``FieldMesh.load_npz``（``save_npz`` 存下来的，省去每次重解析
    PLY）；其余交给 ``load_mesh`` 按扩展名分派。
    """
    from jg_radar.field import FieldMesh, load_mesh

    path = str(getattr(config.field, "mesh_path", "") or "")
    if not path:
        return None
    if str(path).lower().endswith(".npz"):
        return FieldMesh.load_npz(path)
    return FieldMesh.from_raw(
        load_mesh(path),
        mesh_to_field=mesh_to_field_matrix(getattr(config.field, "mesh_to_field", ())),
        unit_scale=float(getattr(config.field, "unit_scale", 1.0)),
        ground_z=float(getattr(config.field, "ground_plane_z", 0.0)),
    )


def lookup_geometry(config, calibration):
    """算映射表要的那几个量。回放标定优先——它是**自描述**的。"""
    from jg_radar.transform import CameraModel, Intrinsics

    if calibration is not None:
        width, height = calibration.width, calibration.height
        K, dist = calibration.K, calibration.dist
        R, t = calibration.R, calibration.t
        plane_z = float(calibration.plane_z)
    else:
        camera_cfg = config.camera
        width, height = camera_cfg.width, camera_cfg.height
        K, dist = camera_cfg.K, camera_cfg.dist
        R, t = config.transform.R, config.transform.t
        plane_z = float(getattr(config.field, "ground_plane_z", 0.0))
    camera = CameraModel(
        intrinsics=Intrinsics.from_matrix(K, dist), width=width, height=height
    )
    return camera, np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64), plane_z


def build_lookup(config, *, calibration=None, stride: int = 1, mode=None, mesh=None):
    """查找表。优先用配置里指的那张（**验过签名**），否则现场算。"""
    from jg_radar.transform import build_lookup_map, compute_signature
    from jg_radar.transform.lookup_map import MODE_GROUND_PLANE, MODE_MESH, LookupMap

    if mode is None:
        mode = str(getattr(config.transform, "projection_mode", MODE_GROUND_PLANE))
    mode = str(mode).strip().lower()
    if mode not in (MODE_GROUND_PLANE, MODE_MESH):
        raise ValueError(
            f"transform.projection_mode={mode!r} 不认识；"
            f"可用 = ['{MODE_GROUND_PLANE}', '{MODE_MESH}']"
        )

    camera, R, t, plane_z = lookup_geometry(config, calibration)

    if mesh is None and mode == MODE_MESH and calibration is None:
        # 回放目录自带标定、也自带它那一套几何；网格是给实拍部署用的，
        # 所以只在走配置那条路上找网格。
        mesh = load_field_mesh(config)

    path = str(getattr(config.transform, "lookup_map_path", "") or "")
    if path:
        if calibration is not None:
            raise ValueError(
                "同时给了 lookup_map_path 和一份回放标定。两者只能取一个："
                "预计算表是按某一套标定算的，配上另一套投影坐标必然整体偏移。"
            )
        return _load_verified(path, camera, R, t, plane_z, mesh, mode, compute_signature,
                              LookupMap)

    return build_lookup_map(
        camera, R, t, plane_z, mode=mode, mesh=mesh, stride=stride
    )


def _load_verified(path, camera, R, t, plane_z, mesh, mode, compute_signature, LookupMap):
    """加载预计算表，并**拿配置现算一遍签名**对账。

    签名对不上意味着这张表是另一套标定/另一个网格算出来的。它可以被正常
    加载、正常查表、正常输出坐标，只是每个坐标都偏——正是最难发现的那一类。
    """
    from jg_radar.transform.lookup_map import effective_mesh_hash

    table = LookupMap.load(path)

    # 先比模式。它也折在签名里，所以签名那一步**能**发现这件事——但报出来
    # 是一个哈希对不上，而真正的原因（"这张表是平面算的，配置要网格"）是
    # 一句话就能说清的。直接比一下，把最有信息量的那条错误放在最前面。
    if table.mode != mode:
        raise ValueError(
            f"查找表 {path} 的投影模式是 {table.mode!r}，配置要的是 {mode!r}。"
            "两者的表结构不同：平面模式的表里不会有遮挡信息，"
            "而网格模式的高地坐标平面模式根本算不出来。请重跑 "
            "tools/build_lookup_map.py，或把 projection_mode 改回表里的那个。"
        )

    expected = compute_signature(
        camera.intrinsics, R, t, camera.width, camera.height, plane_z,
        mesh_hash=effective_mesh_hash(mode, mesh), mode=mode,
    )
    if not table.matches(expected):
        raise ValueError(
            f"查找表 {path} 的签名与当前配置对不上：\n"
            f"  表里记的  {table.signature}\n"
            f"  配置算的  {expected}\n"
            "含义是这张表是用**另一套**内参/外参/分辨率/地面高度/网格算出来的。"
            "继续用它不会报错，只会让每个坐标都偏几十厘米。"
            "请重跑 tools/build_lookup_map.py，或把 lookup_map_path 置空改为现场算。"
        )
    return table


def build_source(config, *, frame_source: FrameSource | None = None) -> FrameSource:
    """按 ``io`` 段造帧源。已经给了就原样用。"""
    if frame_source is not None:
        return frame_source

    kind = str(getattr(config.io, "frame_source", "synthetic")).strip().lower()
    if kind == "replay":
        path = str(config.io.replay_path or "")
        if not path:
            raise ValueError("io.frame_source=replay 但 io.replay_path 是空的")
        return ReplayFrameSource(path)
    if kind == "video":
        # 真实录像。与 replay 的关键区别：帧带 image 而不是 cascade，
        # 所以**检测是真跑的**——这是"没有相机也能吃到真实数据"的唯一入口。
        from jg_radar.io.video_source import VideoFrameSource

        path = str(getattr(config.io, "video_path", "") or "")
        if not path:
            raise ValueError("io.frame_source=video 但 io.video_path 是空的")
        stamps = str(getattr(config.io, "video_timestamps", "") or "")
        return VideoFrameSource(path, timestamps=stamps or None)
    if kind == "synthetic":
        return SyntheticFrameSource(
            count=int(getattr(config.io, "synthetic_frames", 32)),
            dt=1.0 / max(1e-6, float(config.camera.fps)),
            start_at=0.0,
        )
    if kind == "camera":
        from jg_radar.io.hikrobot_camera import HikRobotCameraSource

        return HikRobotCameraSource(
            camera_index=int(getattr(config.io, "camera_index", 0)),
            serial_number=str(getattr(config.io, "camera_serial", "") or ""),
            width=int(getattr(config.io, "camera_width", 0)),
            height=int(getattr(config.io, "camera_height", 0)),
        )
    raise ValueError(
        f"未知的 io.frame_source={kind!r}；可用：replay / synthetic / video / camera"
    )


def build_stage_a(config, lookup, *, cascade=None, timings: Timings | None = None) -> StageA:
    return StageA(lookup, cascade=cascade, timings=timings, slot_names=config.slot_names())


def build_stage_b(config, *, timings: Timings | None = None, publisher=None) -> StageB:
    if publisher is None:
        from jg_radar.fuse.publish import PublishLayer
        publisher = PublishLayer(config.fuse, slot_names=config.slot_names())
    return StageB(
        tracker=MultiTracker(config.track, slot_names=config.slot_names()),
        pipeline=config.pipeline,
        fuse=config.fuse,
        publisher=publisher,
        timings=timings,
    )


def build_loop(
    config,
    *,
    frame_source: FrameSource | None = None,
    lookup=None,
    cascade=None,
    sink=None,
    threaded: bool = False,
    clock=None,
):
    """一把装配完整条流水线，返回可以 ``run()`` 的循环。"""
    source = build_source(config, frame_source=frame_source)

    # 回放目录/视频自带标定 —— 用它，别去读配置里的相机，否则投影链和
    # fixture 用的就是两套参数了。
    calibration = getattr(source, "calibration", None)
    if lookup is None:
        lookup = build_lookup(config, calibration=calibration)

    # 帧上没带检测结果时（video / 相机路径）Stage A 得自己跑级联。
    # 不给它一个级联的话，Stage A 会抛"这一帧既没带检测结果，也没有级联可用"
    # ——那是**对的**行为（沉默不能伪装成"没有目标"），但装配层不该把它留给
    # 运行期：配置里明明写着 detect 段。
    if cascade is None:
        from jg_radar.detect.factory import build_cascade

        cascade = build_cascade(config.detect)
    # **load() 之后必须查就绪**，光调 load() 不够：``OnnxBackend.load`` 是
    # 不抛的——没装 onnxruntime、权重路径写错，它只把原因记进
    # ``backend.reason`` 就返回；随后 ``infer`` 抛"未就绪"，而
    # ``BaseDetector._safe_infer`` 把它吞成"这一帧没有目标"。
    # 于是"模型压根没加载"的表现是：**雷达站一路上报没有敌人、全程零异常**。
    # 这正是本项目反复要挡的那一类失效，所以在这里把它变成启动期错误。
    #
    # 判据分两级：整车级**必须**就绪（它是入口，没它后两级连 ROI 都没有）；
    # 装甲/数字级只在**配置里给了权重路径**时才要求就绪——数字级留空是合法
    # 配置（见 config/ega_replay.yaml），它会走"占位票"那条路。
    if hasattr(cascade, "load"):
        cascade.load()
    car = getattr(cascade, "car", None)
    if car is not None and not getattr(car, "is_ready", True):
        raise RuntimeError(
            "整车检测级未就绪，流水线拒绝启动："
            f"{getattr(getattr(car, 'backend', None), 'reason', '') or '未知原因'}。"
            "继续跑下去不会报错，只会对外一路报“没有敌人”。"
        )
    for _stage, _path in (
        ("armor", getattr(config.detect, "armor_model", "")),
        ("digit", getattr(config.detect, "digit_model", "")),
    ):
        _det = getattr(cascade, _stage, None)
        if _det is not None and _path and not getattr(_det, "is_ready", True):
            raise RuntimeError(
                f"{_stage} 级配置了权重 {_path!r} 但未就绪："
                f"{getattr(getattr(_det, 'backend', None), 'reason', '') or '未知原因'}。"
                "要么把 onnxruntime 装上，要么把该项置空（留空是合法的降级）。"
            )

    timings = Timings(alpha=float(getattr(config.diagnostics, "ema_alpha", 0.1)))
    stage_a = build_stage_a(config, lookup, cascade=cascade, timings=timings)
    stage_b = build_stage_b(config, timings=timings)

    # 实时源必须让仲裁/发布能在视觉阻塞时继续推进；回放仍可逐帧同步执行。
    threaded = bool(threaded or getattr(source, "is_live", False))
    loop_cls = ThreadedEventLoop if threaded else EventLoop
    kwargs = {"source": source, "stage_a": stage_a, "stage_b": stage_b, "sink": sink}
    if threaded:
        from jg_radar.pipeline.mailbox import Mailbox
        kwargs["mailbox"] = Mailbox(capacity=int(config.pipeline.mailbox_size))
    if clock is not None:
        kwargs["clock"] = clock
    return loop_cls(**kwargs)


def default_replay_dir(root: str | Path | None = None) -> Path:
    """仓库里那份回放 fixture 的路径。测试与 ``replay_runner`` 共用。"""
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    return base / "tests" / "fixtures" / "replay_seq_001"
