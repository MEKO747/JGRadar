"""YAML 载入与多文件合并。

两处刻意从严，都是被现有开源方案的真实 bug 逼出来的：

1. **重复键报错**。TCR 的 ``params.yaml`` 里 ``camera:`` 出现两次，后者静默
   覆盖前者，只有逐行读文件才看得出来。这里直接抛异常。
2. **未知键报错**。见 ``schema._build``。

合并语义：后者覆盖前者；两边都是映射时递归合并。所以
``default.yaml`` + ``camera.yaml`` + ``field.yaml`` 的叠加是增量式的。

## 封闭集合：逐键叠加**不总是**对的

增量式叠加对"配置组"是对的——``fuse`` 段里只想改 ``dwell_s``，不该把
``radio_fresh_s`` 一并抹掉。但对**封闭集合**（closed set）就错了：那种字段
表达的是"**全集的枚举**"，写在覆盖方就意味着"这就是全集"。

``detect`` 的三张类别表就是封闭集合。它们同时是**白名单**（见
``schema.DetectConfig``：表外的 class_id 会被整级丢弃），所以少写一个键不是
"这一项保持默认"，而是"**这一项仍然被接受**"。

真实的坑长这样：

    default.yaml    detect.car_classes: {0: B1, 1: B2, ..., 9: R5}
    ega_replay.yaml detect.car_classes: {0: car}

覆盖方想说的是"这份权重只有一类，叫 car"。但逐键叠加的结果是
``{0: car, 1: B2, ..., 9: R5}``——**另外九个键原封不动地留着**。而
``B2``/``B3``/``B4``/``R1``…恰好都是合法槽位名，于是模型只要输出
class_id=1，一票权重 1.0 的"敌方 B2"就会被报出去。空映射 ``{}`` 更隐蔽：
它是递归合并的 no-op，等于什么都没写。

所以这类路径**整块替换**，包括空映射（``{}`` = 显式清空 = 该级不过滤，
与 ``default.yaml`` 一直承诺的语义一致）。

真要在上一层之上追加，把键写成 ``car_classes+``——它是"追加"，且只对
封闭集合有意义（普通映射本来就是递归合并，不需要 ``+``）。同一个映射里
``k`` 和 ``k+`` 同时出现没有确定的先后，直接报错。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Mapping, MutableMapping, Sequence

import yaml

from jg_radar.config.schema import ConfigError, RadarConfig

#: 追加后缀：``car_classes+`` 表示"在基配置那张表之上追加"，而不是整块替换。
APPEND_SUFFIX = "+"

#: **封闭集合**的路径（相对配置根，点号分隔）。写在覆盖方 = 全集，整块替换。
#: 加新路径前先问一句：这个字段"少写一个键"的语义，是"保持默认"还是
#: "仍然接受"？只有后者才属于这里。
CLOSED_SET_PATHS = frozenset(
    {
        "detect.car_classes",
        "detect.armor_classes",
        "detect.digit_classes",
    }
)


def _reject_append_conflicts(override: Mapping, path: str) -> None:
    """同一个映射里既写 ``k`` 又写 ``k+``：没有确定的先后，报错而不是猜。"""
    stems = {
        k[: -len(APPEND_SUFFIX)]
        for k in override
        if isinstance(k, str) and k.endswith(APPEND_SUFFIX)
    }
    both = sorted(stems & set(override))
    if both:
        names = "、".join(repr(k) for k in both)
        raise ConfigError(
            f"{path}: {names} 与同名的 {APPEND_SUFFIX!r} 后缀键同时出现。"
            f"前者是整块替换，后者是在上一层之上追加，一起写没有先后可言，二选一。"
        )


class _StrictLoader(yaml.SafeLoader):
    """拒绝重复键的 SafeLoader。"""


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False):
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    seen_at: dict[Any, int] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigError(
                f"{loader.name}:{key_node.start_mark.line + 1} 重复的键 {key!r}"
                f"（首次出现于第 {seen_at[key] + 1} 行）"
            )
        seen_at[key] = key_node.start_mark.line
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """载入单个 YAML 文件，拒绝重复键。"""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"配置文件不存在：{p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.load(fh, Loader=_StrictLoader)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: 顶层必须是映射，得到 {type(data).__name__}")
    return data


def deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
    *,
    closed: Collection[str] = CLOSED_SET_PATHS,
    _path: str = "",
) -> dict[str, Any]:
    """递归合并；映射递归、标量/列表覆盖。

    ``closed`` 里的路径（相对配置根）例外，**整块替换**，见模块文档。
    ``_path`` 是递归内部用的当前路径前缀，外部调用不要传。

    注意 ``+`` 只对**映射**的封闭集合有意义。列表/元组（``field.roles``、
    ``fuse.priority``、``track.blind_waypoints``）本来就是覆盖语义，写 ``+``
    不会变成"往后追加"——真要追加得改这里，别指望那个后缀。
    """
    _reject_append_conflicts(override, _path or "<根>")

    out: MutableMapping[str, Any] = dict(base)
    for raw_key, value in override.items():
        append = isinstance(raw_key, str) and raw_key.endswith(APPEND_SUFFIX)
        key = raw_key[: -len(APPEND_SUFFIX)] if append else raw_key
        path = f"{_path}.{key}" if _path else str(key)

        if not isinstance(value, Mapping):
            # 标量/列表：覆盖。
            out[key] = value
            continue

        base_value = out.get(key)
        if not isinstance(base_value, Mapping):
            # 上一层没有这一节（或它不是映射）：从空映射起递归。
            # **必须还是走递归**，不能直接 `out[key] = value`——否则
            # `car_classes+` 这种标记键会因为没人剥后缀而原样留在结果里，
            # 到了 schema 那儿就是一个"未知配置项"。
            base_value = {}
        elif path in closed and not append:
            # 封闭集合且没带 `+`：**整块替换**。空映射也走这里——"写 {}
            # 表示清空"能成立就靠这一步；递归合并会把 {} 当成无事发生。
            base_value = {}
        out[key] = deep_merge(base_value, value, closed=closed, _path=path)
    return dict(out)


def load_config(*paths: str | Path) -> RadarConfig:
    """按顺序载入并合并多个 YAML，返回校验过的根配置。"""
    if not paths:
        raise ConfigError("至少要给一个配置文件")
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, load_yaml(path))
    return RadarConfig.from_dict(merged)


def load_config_with_paths(paths: Sequence[str | Path]) -> tuple[RadarConfig, tuple[Path, ...]]:
    """返回配置与归一化后的路径，便于在日志里记录"这批结果用了哪些配置"。"""
    return load_config(*paths), tuple(Path(p).resolve() for p in paths)
