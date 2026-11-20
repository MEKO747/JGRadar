"""测试公共配置。

不依赖 `pip install -e .`：直接把仓库根加进 sys.path，让测试在任何环境下都能跑。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RADIO_PACKAGE = ROOT / "radio" / "radar_parse_em_wave"
# 接收包（``rmuc``）在 radio/ 下，没有 pip install 进环境；直接加进 sys.path，
# 免得测试能不能导入取决于"先 import 了哪个 tools 模块"这种副作用。
for _path in (ROOT, RADIO_PACKAGE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONFIG_DIR = ROOT / "config"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    return FIXTURES


@pytest.fixture(scope="session")
def config_dir() -> Path:
    return CONFIG_DIR
