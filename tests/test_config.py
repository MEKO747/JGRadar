"""配置层测试。

重点是两条"从严"规则，它们分别对应现有开源方案里真实踩过的坑：
  - 重复键（TCR params.yaml 的 camera: 出现两次）
  - 未知键（拼错的键名静默失效）
"""

from __future__ import annotations

import pytest
import yaml

from jg_radar.config import ConfigError, RadarConfig, deep_merge, load_config, load_yaml
from jg_radar.types import SCHEMA_VERSION


def test_load_project_configs(config_dir):
    cfg = load_config(
        config_dir / "default.yaml",
        config_dir / "camera.yaml",
        config_dir / "field.yaml",
    )
    assert isinstance(cfg, RadarConfig)
    assert cfg.schema_version == SCHEMA_VERSION
    assert cfg.detect.backend == "stub"
    assert cfg.camera.width == 4024
    assert cfg.field.size_x_cm == 2800
    assert len(cfg.transform.R) == 3
    assert len(cfg.transform.R[0]) == 3
    assert isinstance(cfg.transform.t[0], float)


def test_slot_names_from_config(config_dir):
    cfg = load_config(config_dir / "default.yaml", config_dir / "field.yaml")
    names = cfg.slot_names()
    assert names == ("R1", "R2", "R3", "R4", "R6", "R7", "B1", "B2", "B3", "B4", "B6", "B7")


def test_defaults_are_constructible():
    cfg = RadarConfig.from_dict({})
    assert cfg.detect.backend == "stub"
    assert cfg.fuse.priority[0] == "radio_fresh"
    assert cfg.track.gate_chi2 == pytest.approx(9.49)


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="未知配置项"):
        RadarConfig.from_dict({"track": {"raw_jump": 10.0}})  # 少了 _px


def test_unknown_top_level_key_rejected():
    with pytest.raises(ConfigError, match="未知配置项"):
        RadarConfig.from_dict({"cameraa": {}})


def test_unknown_nested_key_rejected():
    with pytest.raises(ConfigError, match=r"config\.transform"):
        RadarConfig.from_dict({"transform": {"projection_modee": "mesh"}})


def test_duplicate_key_rejected(tmp_path):
    p = tmp_path / "dup.yaml"
    p.write_text(
        "camera:\n  width: 100\ncamera:\n  width: 200\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="重复的键"):
        load_yaml(p)


def test_duplicate_key_that_yaml_would_silently_drop(tmp_path):
    """证明这不是杞人忧天：原生 SafeLoader 会静默取后者。"""
    text = "camera:\n  width: 100\ncamera:\n  width: 200\n"
    assert yaml.safe_load(text) == {"camera": {"width": 200}}
    p = tmp_path / "dup.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_yaml(p)


def test_type_coercion():
    cfg = RadarConfig.from_dict(
        {
            "camera": {"width": 1920, "fps": 30, "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            "field": {"roles": ["1", "3"], "sides": ["R"]},
        }
    )
    assert cfg.camera.width == 1920
    assert cfg.camera.fps == pytest.approx(30.0)
    assert isinstance(cfg.camera.K[0][0], float)
    assert cfg.field.roles == ("1", "3")
    assert cfg.slot_names() == ("R1", "R3")


def test_wrong_type_rejected():
    with pytest.raises(ConfigError, match="期望字符串"):
        RadarConfig.from_dict({"detect": {"backend": 3}})
    with pytest.raises(ConfigError, match="期望列表"):
        RadarConfig.from_dict({"field": {"roles": "1"}})
    with pytest.raises(ConfigError, match="布尔值不能当数值用"):
        RadarConfig.from_dict({"pipeline": {"max_commit_age_s": True}})


def test_tuple_arity_checked():
    with pytest.raises(ConfigError, match="期望 3 个元素"):
        RadarConfig.from_dict({"transform": {"t": [0.0, 0.0]}})


def test_none_allowed_for_optional():
    cfg = RadarConfig.from_dict({"transform": {"reproj_rms_px": None}})
    assert cfg.transform.reproj_rms_px is None
    cfg2 = RadarConfig.from_dict({"transform": {"reproj_rms_px": 2.5}})
    assert cfg2.transform.reproj_rms_px == pytest.approx(2.5)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="不存在"):
        load_yaml(tmp_path / "nope.yaml")


def test_deep_merge():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    over = {"b": {"d": 9, "e": 10}, "f": 11}
    assert deep_merge(base, over) == {"a": 1, "b": {"c": 2, "d": 9, "e": 10}, "f": 11}
    # 原字典不被修改
    assert base == {"a": 1, "b": {"c": 2, "d": 3}}


def test_merge_order_later_wins(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("detect:\n  backend: trt\n  iou: 0.5\ntrack:\n  ttl_s: 1.0\n", encoding="utf-8")
    b.write_text("detect:\n  backend: onnx\n", encoding="utf-8")
    cfg = load_config(a, b)
    assert cfg.detect.backend == "onnx"   # 后者覆盖
    assert cfg.detect.iou == pytest.approx(0.5)  # 未被覆盖的保留
    assert cfg.track.ttl_s == pytest.approx(1.0)


def test_empty_file_is_empty_mapping(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("# 只有注释\n", encoding="utf-8")
    assert load_yaml(p) == {}


# ---------------------------------------------------------------------------
# 封闭集合（detect 的三张类别表）
# ---------------------------------------------------------------------------
# 这三张表同时是**白名单**（表外的 class_id 会被整级丢弃），所以叠加配置时
# "少写一个键"不是"这一项保持默认"，而是"这一项**仍然被接受**"。
# 真实踩过的坑：default.yaml 那张示例表 1→B2 … 全是合法槽位名，于是权重输出
# class_id=1 就变成一票权重 1.0 的"敌方 B2"报出去。详见 ega_replay.yaml【发现 1】。


def test_closed_set_replaces_instead_of_merging():
    """写一张表 = 全集，不是"在上一层之上打补丁"。"""
    base = {"detect": {"car_classes": {0: "B1", 1: "B2", 2: "B3"}}}
    over = {"detect": {"car_classes": {0: "car"}}}
    assert deep_merge(base, over)["detect"]["car_classes"] == {0: "car"}


def test_closed_set_empty_mapping_clears():
    """写 ``{}`` = 显式清空（= 该级不过滤）。

    这条曾经是 no-op：递归合并里空映射"无事发生"，于是上一层那张表照样生效。
    """
    base = {"detect": {"car_classes": {0: "B1", 1: "B2"}}}
    got = deep_merge(base, {"detect": {"car_classes": {}}})
    assert got["detect"]["car_classes"] == {}


def test_closed_set_covers_all_three_class_tables():
    """三张表一视同仁——只修 car_classes 的话，换到装甲级照样能出错身份。"""
    base = {"detect": {"car_classes": {1: "B2"}, "armor_classes": {1: "B2"}, "digit_classes": {1: "2"}}}
    got = deep_merge(base, {"detect": {k: {} for k in base["detect"]}})["detect"]
    assert got == {"car_classes": {}, "armor_classes": {}, "digit_classes": {}}


def test_append_suffix_merges_into_base():
    """真要打补丁得明说：``car_classes+``。"""
    base = {"detect": {"car_classes": {0: "B1", 1: "B2"}}}
    over = {"detect": {"car_classes+": {2: "B3"}}}
    got = deep_merge(base, over)["detect"]
    assert got["car_classes"] == {0: "B1", 1: "B2", 2: "B3"}
    # `+` 是给合并器看的标记，不是配置项，不该留在结果里
    assert "car_classes+" not in got


def test_append_suffix_creates_the_key_when_absent():
    assert deep_merge({}, {"detect": {"car_classes+": {0: "car"}}})["detect"] == {
        "car_classes": {0: "car"}
    }


def test_append_and_plain_in_the_same_mapping_is_rejected():
    """两种写法一起给没有确定的先后，报错而不是猜。"""
    over = {"detect": {"car_classes": {0: "X"}, "car_classes+": {1: "Y"}}}
    with pytest.raises(ConfigError, match="后缀键"):
        deep_merge({"detect": {}}, over)


def test_ordinary_mappings_still_merge():
    """封闭集合是**例外**而不是新常态：普通配置组必须继续逐键叠加，
    否则 `config/camera.yaml` 那种"只改几个键"的裁剪文件会全废。"""
    base = {"detect": {"backend": "stub", "iou": 0.45, "car_classes": {0: "B1"}}}
    got = deep_merge(base, {"detect": {"iou": 0.7}})["detect"]
    assert got["backend"] == "stub"  # 没提到的保留
    assert got["iou"] == pytest.approx(0.7)
    assert got["car_classes"] == {0: "B1"}


def test_closed_set_applies_through_load_config(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("detect:\n  car_classes:\n    0: B1\n    1: B2\n", encoding="utf-8")
    b.write_text("detect:\n  car_classes:\n    0: car\n", encoding="utf-8")
    assert load_config(a, b).detect.car_classes == {0: "car"}


def test_shipped_ega_overlay_keeps_only_its_own_class(config_dir):
    """随包发布的 ega_replay.yaml：那张表里只能有它自己写的那个类。

    EGA 的 best.onnx 只有一类（names=={0:'item'}），所以 1–9 号键全是
    default.yaml 漏过来的。数字对不上权重，是"自信地报错人"的入口。
    """
    ega = config_dir / "ega_replay.yaml"
    if not ega.is_file():
        pytest.skip("ega_replay.yaml 不在这个环境里")
    cfg = load_config(config_dir / "default.yaml", ega)
    assert cfg.detect.car_classes == {0: "item"}
    assert 1 not in cfg.detect.car_classes


def test_a_leaked_class_id_can_no_longer_name_an_enemy(config_dir):
    """锁**后果**：合并后的 EGA 配置里，class_id≥1 不再是任何槽位。

    修之前它是 'B2'——白名单放行 → 标签 'B2' → 身份投票满票 →
    跟踪器 `label in slot_names` 放行 → 作为敌方 B2 报出去。
    """
    from jg_radar.detect.labels import ClassMap

    ega = config_dir / "ega_replay.yaml"
    if not ega.is_file():
        pytest.skip("ega_replay.yaml 不在这个环境里")
    cfg = load_config(config_dir / "default.yaml", ega)
    cmap = ClassMap.from_spec(cfg.detect.car_classes or None)
    slots = set(cfg.slot_names())
    leaked = {
        cid: cmap.label(cid)
        for cid in range(1, 16)
        if cmap.has(cid) and cmap.label(cid) in slots
    }
    assert not leaked, f"这些 class_id 能直接变成敌方身份：{leaked}"


def test_every_closed_path_actually_exists():
    """守卫的守卫：``CLOSED_SET_PATHS`` 是一串**字符串**路径。

    它失配的方式是**静默**的——有人把 ``car_classes`` 改个名，规则就不再
    命中，而症状（身份自信地报错人）要等到换权重才出现。所以这里把每条
    路径都对着 dataclass 走一遍，走不通就红。
    """
    from dataclasses import fields, is_dataclass
    from typing import get_type_hints

    from jg_radar.config import CLOSED_SET_PATHS

    assert CLOSED_SET_PATHS, "封闭集合不该是空的——空了就等于这条规则没生效"

    for path in sorted(CLOSED_SET_PATHS):
        current = RadarConfig
        for part in path.split("."):
            assert is_dataclass(current), f"{path}: {part} 之前就不是 dataclass 了"
            names = {f.name for f in fields(current)}
            assert part in names, (
                f"{path}: {current.__name__} 里没有 {part!r}；"
                f"字段改名了就要同步 loader.CLOSED_SET_PATHS，"
                f"否则这条规则静默失效。现有字段 = {sorted(names)}"
            )
            current = get_type_hints(current)[part]
        # 走到底必须还是个映射（dict[...]），否则封闭集合无从谈起
        assert "dict" in str(current).lower(), f"{path} 的最终类型不是映射：{current}"


def test_config_is_frozen(config_dir):
    cfg = load_config(config_dir / "default.yaml")
    with pytest.raises(Exception):
        cfg.detect.backend = "trt"  # type: ignore[misc]
