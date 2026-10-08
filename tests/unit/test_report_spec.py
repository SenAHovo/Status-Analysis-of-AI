import pytest
from pydantic import ValidationError

from ai_status_report.schemas.report_spec import REPORT_SECTION_ORDER, default_report_spec


def test_default_report_spec_has_fixed_four_chapters_and_dependencies():
    spec = default_report_spec()

    assert tuple(item.section_id for item in spec.sections) == REPORT_SECTION_ORDER
    assert [item.title for item in spec.sections] == ["背景", "现状", "趋势", "建议"]
    assert spec.section("recommendations").depends_on == ["current-status", "trends"]


def test_report_spec_rejects_reordered_or_missing_top_level_chapters():
    spec = default_report_spec().model_dump()
    spec["sections"] = spec["sections"][:-1]

    with pytest.raises(ValidationError):
        type(default_report_spec()).model_validate(spec)
