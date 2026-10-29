#!/usr/bin/env python3
"""生成像素 → 场地坐标查找表，落成 ``.npz``。**运行时主路径的上游。**

    # 先看代价，不真跑
    python3 tools/build_lookup_map.py --config config/default.yaml \
        --camera config/camera.yaml --field config/field.yaml --dry-run

    # 真跑，落到配置里那个 lookup_map_path
    python3 tools/build_lookup_map.py --config config/default.yaml \
        --camera config/camera.yaml --field config/field.yaml

在此之前 ``LookupMap.save`` 存在却**没有任何调用方**——也就是说这张表
在代码里根本产不出来，只能靠现场算。现场算在平面模式下没问题，在网格模式
下每启动一次都要重跑几十分钟。这个工具补的就是那一步。

## 为什么先报代价

``mesh`` 模式下要算的是 **(射线 × 三角形)** 对：12MP 全场网格是
1200 万 × 几万，几十分钟到几小时起步。没有预估的话，`--dry-run` 缺失的
后果是"按了回车然后不确定它是在跑还是卡死了"。

预估不是拿一个拍脑袋的常数乘出来的，而是**在这台机器上、对这张网格**
先跑一小撮射线实测吞吐再线性外推。常数会随 numpy 版本、CPU、网格形状
（退化三角形、缓存局部性）漂移，实测不会。

线性外推**只会偏乐观**：它忽略了分块的固定开销和内存带宽随规模下降。
所以预估偏大不是问题，偏小才是——工具会在跑之前把这句话说出来。

## 签名

算完顺手把表读回来、拿配置现算一遍签名对账。`LookupMap.matches` 以前
没有调用方，于是"表和配置对不上"这件事没有任何一处能发现。这里让它
在**产出这一端**就断掉，而不是等到运行时。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jg_radar.config import load_config  # noqa: E402
from jg_radar.pipeline.build import (  # noqa: E402
    load_field_mesh,
    lookup_geometry,
)
from jg_radar.transform import build_lookup_map, compute_signature  # noqa: E402
from jg_radar.transform.lookup_map import (  # noqa: E402
    MODE_GROUND_PLANE,
    MODE_MESH,
    LookupMap,
    effective_mesh_hash,
)

#: 预估时探针射线的条数。够多才有代表性，够少才不至于让人等预估本身。
PROBE_RAYS = 1024

#: 预估里每次查询的固定开销（微秒）。查表、装箱、掩码都是按像素线性走的，
#: 跟射线数无关；单独拎出来是因为在网格模式下它会被求交彻底淹没，而在
#: 平面模式下它就是全部。
_PER_PIXEL_US = 1.5


def human(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 90.0:
        return f"{seconds:.1f}s"
    if seconds < 5400.0:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


def probe_throughput(camera, R, t, mesh, *, max_block_mb: float) -> tuple[float, str]:
    """实测一小撮射线的求交吞吐，返回 ``(微秒/射线, 说明文字)``。

    探针用**全图均匀取样**的像素而不是随便几条：射线的方向分布会影响
    三角形命中率，而命中率会影响 ``argmin`` 那一行的开销。均匀取样至少
    和真实调用是同一个分布。
    """
    from jg_radar.transform.raycast import intersect_mesh

    n = min(PROBE_RAYS, camera.width * camera.height)
    step = max(1, int(np.sqrt(camera.width * camera.height / max(1, n))))
    us = np.arange(0, camera.width, step)
    vs = np.arange(0, camera.height, step)
    gu, gv = np.meshgrid(us, vs)
    pixels = np.stack([gu.ravel(), gv.ravel()], axis=1).astype(np.float64)
    pixels = pixels[:PROBE_RAYS]
    if not len(pixels):  # pragma: no cover - 分辨率为零的相机不存在
        return 0.0, "探针为空"

    origins, dirs = camera.rays_world(pixels, R, t)
    # 先跑一遍把 numpy 的首次调用开销、页错误、BLAS 线程池都预热掉，
    # 否则测到的是启动成本而不是吞吐。
    intersect_mesh(origins, dirs, mesh.vertices, mesh.faces,
                   max_block_mb=max_block_mb)
    t0 = time.perf_counter()
    intersect_mesh(origins, dirs, mesh.vertices, mesh.faces,
                   max_block_mb=max_block_mb)
    dt = time.perf_counter() - t0
    return dt / len(pixels) * 1e6, f"{len(pixels)} 条射线实测"


def estimate(camera, mesh, *, mode: str, stride: int, max_block_mb: float,
             us_per_ray: float) -> dict[str, float]:
    """从实测吞吐外推全表代价。"""
    n_pixels = int(np.ceil(camera.height / stride)) * int(np.ceil(camera.width / stride))
    n_faces = int(len(mesh.faces)) if mesh is not None else 0
    ray_seconds = us_per_ray * n_pixels * 1e-6 if mode == MODE_MESH else 0.0
    fixed = _PER_PIXEL_US * n_pixels * 1e-6
    # 表本身的内存：coords 是 float64 的 (H,W,2)，另加两个布尔面。
    table_mb = (camera.height * camera.width * (2 * 8 + 2)) / 1e6
    return {
        "n_pixels": float(n_pixels),
        "n_faces": float(n_faces),
        "pairs": float(n_pixels) * float(n_faces),
        "ray_seconds": ray_seconds,
        "fixed_seconds": fixed,
        "total_seconds": ray_seconds + fixed,
        "table_mb": table_mb,
        "us_per_ray": us_per_ray,
    }


def report_estimate(est: dict[str, float], *, mode: str, stride: int,
                    max_block_mb: float) -> None:
    print("代价预估")
    print(f"  模式              {mode}")
    print(f"  stride            {stride}（表中每 {stride}×{stride} 个像素取一个）")
    print(f"  待算像素          {int(est['n_pixels']):,}")
    if mode == MODE_MESH:
        print(f"  网格面数          {int(est['n_faces']):,}")
        print(f"  射线×三角形       {est['pairs']:.3e} 对"
              f"（分块预算 {max_block_mb:.0f}MB）")
        print(f"  求交实测吞吐      {est['us_per_ray']:.2f} µs/射线")
    print(f"  构建耗时          ≈ {human(est['total_seconds'])}")
    print(f"  表常驻内存        ≈ {est['table_mb']:.0f}MB（落盘 float32 减半）")
    if mode == MODE_MESH:
        print("  外推是**线性**的、忽略了分块固定开销；实际只会更慢，不会更快。")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="生成像素→场地坐标查找表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", action="append", default=[],
                    help="配置 YAML，可给多次（后面的覆盖前面的）")
    ap.add_argument("--camera", default="", help="额外叠加相机 YAML")
    ap.add_argument("--field", default="", help="额外叠加场地 YAML")
    ap.add_argument("--mode", default="", choices=["", MODE_GROUND_PLANE, MODE_MESH],
                    help="覆盖 transform.projection_mode")
    ap.add_argument("--mesh", default="", help="覆盖 field.mesh_path")
    ap.add_argument("--out", default="",
                    help="输出路径；默认为配置里的 transform.lookup_map_path")
    ap.add_argument("--stride", type=int, default=1,
                    help="抽样步长，>1 出粗表（预览与快速验证用）")
    ap.add_argument("--max-block-mb", type=float, default=192.0,
                    help="网格求交的中间数组内存预算")
    ap.add_argument("--dry-run", action="store_true", help="只报代价，不真算")
    args = ap.parse_args(argv)

    paths = list(args.config)
    for extra in (args.camera, args.field):
        if extra:
            paths.append(extra)
    if not paths:
        ap.error("至少要给一个 --config")
    cfg = load_config(*paths)

    mode = (args.mode or cfg.transform.projection_mode).strip().lower()
    if mode not in (MODE_GROUND_PLANE, MODE_MESH):
        print(f"未知的投影模式 {mode!r}；可用 = ['{MODE_GROUND_PLANE}', '{MODE_MESH}']",
              file=sys.stderr)
        return 2

    if args.mesh:
        # 覆盖配置里的网格路径。直接改一份 dataclass(frozen) 不方便，
        # 用一个只带 field 的浅壳；load_field_mesh 只读 field 段。
        cfg = _with_mesh_path(cfg, args.mesh)

    if args.stride < 1:
        print("--stride 必须 ≥ 1", file=sys.stderr)
        return 2

    camera, R, t, plane_z = lookup_geometry(cfg, None)
    print(f"相机 {camera.width}×{camera.height}，地面 z={plane_z:.1f}cm，模式 {mode}")

    mesh = load_field_mesh(cfg) if mode == MODE_MESH else None
    if mode == MODE_MESH:
        if mesh is None:
            print("projection_mode=mesh 但 field.mesh_path 是空的。网格模式必须有网格："
                  "没有网格就只能退回平面模式，而那会把高地目标静默地投到它身后。",
                  file=sys.stderr)
            return 2
        print(f"  {mesh.summary()}")

    us_per_ray = 0.0
    if mode == MODE_MESH:
        us_per_ray, note = probe_throughput(camera, R, t, mesh,
                                            max_block_mb=args.max_block_mb)
        print(f"  吞吐探针：{note}")

    est = estimate(camera, mesh, mode=mode, stride=args.stride,
                   max_block_mb=args.max_block_mb, us_per_ray=us_per_ray)
    report_estimate(est, mode=mode, stride=args.stride,
                    max_block_mb=args.max_block_mb)

    out = args.out or str(getattr(cfg.transform, "lookup_map_path", "") or "")
    if args.dry_run:
        print("--dry-run：到此为止。")
        return 0
    if not out:
        print("没有给出 --out，配置里的 transform.lookup_map_path 也是空的。"
              "要么指定一个输出路径，要么先把配置里的那一项填上。", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    table = build_lookup_map(camera, R, t, plane_z, mode=mode, mesh=mesh,
                             stride=args.stride, max_block_mb=args.max_block_mb)
    dt = time.perf_counter() - t0

    print("\n构建完成")
    print(f"  实际耗时          {human(dt)}"
          f"（预估 {human(est['total_seconds'])}）")
    print(f"  有效覆盖          {table.coverage:.2%}")
    if mode == MODE_MESH:
        # 这个数才是网格模式相对平面模式**唯一多出来的信息**：
        # 画面里有百分之多少是被墙/立柱挡住的。
        print(f"  遮挡占比          {table.occluded_coverage:.2%}")
    else:
        print("  遮挡占比          —（平面模式没有遮挡的概念，全 False）")
    print(f"  签名              {table.signature}")

    path = table.save(out)
    print(f"  已写入            {path}（{path.stat().st_size / 1e6:.1f}MB）")

    # 读回来对账。产出端就把"签名对不上"断掉，而不是留到运行时。
    back = LookupMap.load(path)
    digest = effective_mesh_hash(mode, mesh)
    expected = compute_signature(camera.intrinsics, R, t, camera.width, camera.height,
                                 plane_z, mesh_hash=digest, mode=mode)
    if not back.matches(expected) or back.mode != mode:
        print("落盘的表读回来签名对不上——写盘/读盘这一环有问题。", file=sys.stderr)
        return 1
    print("  回读校验          通过（签名与模式一致）")
    return 0


def _with_mesh_path(cfg, path: str):
    """给配置换一个网格路径。``field`` 之外的部分原样保留。"""
    import dataclasses

    field = dataclasses.replace(cfg.field, mesh_path=str(path))
    return dataclasses.replace(cfg, field=field)


if __name__ == "__main__":
    raise SystemExit(main())
