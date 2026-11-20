"""配置文件与代码默认值的一致性。

这条测试的存在理由是一起真实事故：`schema.py` 把 `dwell_s` 从 0.3 改成 0.2，
但 `config/default.yaml` 里还写着 0.3——**yaml 覆盖代码默认值**，于是改动在
真实运行里从来没生效过，只有走 dataclass 默认值的单元测试看到的是 0.2。

所以：改任何默认参数，必须同时改 `config/default.yaml`，并且在这里被逐字段
比对。以后再漂移，这条测试先红，而不是等实跑数据对不上才发现。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from jg_radar.config.loader import load_config
from jg_radar.config.schema import FuseConfig

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


@pytest.fixture(scope="module")
def default_config():
    return load_config(DEFAULT_YAML)


def test_dwell_is_two_hundred_ms(default_config):
    """admin 指出的那一条：加载 yaml 后必须是 0.2，不是 0.3。"""
    assert default_config.fuse.dwell_s == pytest.approx(0.2)


def test_stale_mark_is_half_a_second(default_config):
    assert default_config.fuse.stale_mark_s == pytest.approx(0.5)


def test_yaml_fuse_section_does_not_drift_from_schema(default_config):
    """``fuse`` 整段的逐字段比对。

    刻意做**全量**比对而不是只盯几个数：漏掉的字段正是下次会漂移的那个。
    真有需要偏离代码默认值的参数，改 schema 默认值或把它从这份比对里显式
    排除，都比让它悄悄不一致强。
    """
    defaults = FuseConfig()
    drifted = [
        (field.name, getattr(default_config.fuse, field.name), getattr(defaults, field.name))
        for field in dataclasses.fields(FuseConfig)
        if getattr(default_config.fuse, field.name) != getattr(defaults, field.name)
    ]
    assert drifted == [], f"config/default.yaml 与 schema 默认值不一致：{drifted}"
