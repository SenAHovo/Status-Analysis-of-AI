from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root() -> Path:
    """Repository root containing pyproject.toml and config/."""
    return PROJECT_ROOT
