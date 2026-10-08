"""Business default configuration loaded from ``config/defaults.yaml``.

Only non-sensitive defaults live here. Credentials, absolute paths and
permissions are managed by settings/.env and the program, never in defaults.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field

from ai_status_report.schemas.common import SCHEMA_VERSION, ProjectModel
from ai_status_report.schemas.research import (
    DEFAULT_LENGTH_RANGE,
    DEFAULT_READER_STYLE,
    Geoscope,
    ReaderStyle,
)

DEFAULT_FILE_NAME = "config/defaults.yaml"

READER_EMPHASIS: dict[ReaderStyle, str] = {
    ReaderStyle.GENERAL: "技术与应用的均衡概览，不假设专业背景",
    ReaderStyle.STUDENT: "概念、机制、技术路径与入门案例，适量解释术语",
    ReaderStyle.ACADEMIC: "最新进展、研究问题与趋势，区分预印本、实验结果与成熟结论",
    ReaderStyle.BUSINESS: "工程进展、落地条件、成本收益依据与实施风险",
}

GEO_LABELS: dict[Geoscope, str] = {
    Geoscope.GLOBAL_WITH_CHINA: "全球并兼顾中国",
    Geoscope.GLOBAL: "全球",
    Geoscope.CHINA: "中国",
}

DEFAULT_STRUCTURE = ["背景", "现状", "趋势", "建议"]


class DefaultsFile(ProjectModel):
    """Validated content of config/defaults.yaml."""

    schema_version: str = SCHEMA_VERSION
    timezone: str = Field(default="Asia/Shanghai", max_length=64)
    language: str = Field(default="zh", pattern=r"^[a-z]{2,3}$")
    geography: Geoscope = Geoscope.GLOBAL_WITH_CHINA
    audience: ReaderStyle = DEFAULT_READER_STYLE
    length_min_words: int = Field(default=DEFAULT_LENGTH_RANGE[0], ge=200)
    length_max_words: int = Field(default=DEFAULT_LENGTH_RANGE[1], ge=200)
    report_structure: list[str] = Field(default_factory=lambda: list(DEFAULT_STRUCTURE))
    output_format: str = Field(default="pdf", max_length=16)
    period_default_months: int = Field(default=12, ge=1, le=120)
    default_relative_expression: str = Field(default="最近十二个月", max_length=80)

    @property
    def region_label(self) -> str:
        return GEO_LABELS[self.geography]

    @property
    def reader_emphasis(self) -> str:
        return READER_EMPHASIS[self.audience]


def load_defaults_file(path: Path) -> DefaultsFile:
    """Load and validate the defaults YAML; never echoes file content."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError("defaults config unavailable") from exc
    except yaml.YAMLError as exc:
        raise ValueError("defaults config is not valid yaml") from exc
    if not isinstance(raw, dict):
        raise TypeError("defaults config root must be a mapping")
    try:
        return DefaultsFile.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError; message holds no secrets
        raise ValueError(f"defaults config invalid: {exc}") from exc


def defaults_for_root(root: Path) -> DefaultsFile:
    """Load defaults for a project root, falling back to built-in values."""
    path = root / DEFAULT_FILE_NAME
    if not path.is_file():
        return DefaultsFile()
    return load_defaults_file(path)
