"""配置：dataclass 模式 + 严格 YAML 载入。"""

from jg_radar.config.loader import (
    APPEND_SUFFIX,
    CLOSED_SET_PATHS,
    deep_merge,
    load_config,
    load_yaml,
)
from jg_radar.config.schema import (
    CameraConfig,
    ConfigError,
    DetectConfig,
    DiagnosticsConfig,
    FieldConfig,
    FuseConfig,
    IOConfig,
    PipelineConfig,
    RadarConfig,
    TrackConfig,
    TransformConfig,
)

__all__ = [
    "APPEND_SUFFIX",
    "CLOSED_SET_PATHS",
    "CameraConfig",
    "ConfigError",
    "DetectConfig",
    "DiagnosticsConfig",
    "FieldConfig",
    "FuseConfig",
    "IOConfig",
    "PipelineConfig",
    "RadarConfig",
    "TrackConfig",
    "TransformConfig",
    "deep_merge",
    "load_config",
    "load_yaml",
]
