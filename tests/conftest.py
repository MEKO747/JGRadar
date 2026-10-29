"""测试公共配置。

不依赖 `pip install -e .`：直接把仓库根加进 sys.path，让测试在任何环境下都能跑。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONFIG_DIR = ROOT / "config"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    return FIXTURES


@pytest.fixture(scope="session")
def config_dir() -> Path:
    return CONFIG_DIR
