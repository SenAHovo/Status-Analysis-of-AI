"""Briefing: intent routing, calendar-safe windows, defaults and normalization."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from ai_status_report.briefing.clock import FixedClock, SystemClock
from ai_status_report.briefing.defaults import DefaultsFile, defaults_for_root, load_defaults_file
from ai_status_report.briefing.intents import Intent, classify
from ai_status_report.briefing.normalize import BriefPlan, extract_topic, normalize_brief
from ai_status_report.briefing.timewindows import resolve_period, shift_months
from ai_status_report.schemas import (
    OTHER_OPTION,
    OriginKind,
    PreferenceField,
    PreferenceSelection,
    ReaderStyle,
    WritingStyle,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")


def fixed_clock() -> FixedClock:
    return FixedClock("Asia/Shanghai", datetime(2026, 9, 8, 10, 30, tzinfo=SHANGHAI))


def default_brief(text: str = "分析人工智能现状") -> BriefPlan:
    return normalize_brief(text, fixed_clock(), DefaultsFile())


# --- intents ---

def test_report_request_routes_to_report():
    assert classify("分析人工智能现状").kind is Intent.REPORT_REQUEST


def test_out_of_scope_action_is_unsupported():
    assert classify("请帮我订一张去北京的机票").kind is Intent.UNSUPPORTED
    assert classify("今天天气怎么样").kind is Intent.UNSUPPORTED


def test_small_talk_without_subject_is_conversation():
    assert classify("你好").kind is Intent.CONVERSATION
    assert classify("继续").kind is Intent.CONVERSATION


def test_mixed_greeting_with_report_action_stays_report():
    # Greeting words must not short-circuit a genuine report request.
    assert classify("你好，请分析一下人工智能发展现状").kind is Intent.REPORT_REQUEST


def test_mixed_domain_and_out_of_scope_word_stays_report():
    # "天气" here is the research subject, not an out-of-scope request.
    assert classify("请分析人工智能在天气预报中的应用现状").kind is Intent.REPORT_REQUEST


# --- clock ---

def test_fixed_clock_is_deterministic_and_zoned():
    clock = fixed_clock()
    assert clock.today() == date(2026, 9, 8)
    assert clock.now().tzinfo is SHANGHAI


def test_fixed_clock_rejects_naive_moment():
    naive = datetime.fromisoformat("2026-09-08T10:00:00")
    with pytest.raises(ValueError):
        FixedClock("Asia/Shanghai", naive)


def test_system_clock_returns_aware_time():
    moment = SystemClock("Asia/Shanghai").now()
    assert moment.tzinfo is not None


# --- calendar-safe windows ---

def test_shift_months_leap_day_2024():
    assert shift_months(date(2024, 2, 29), -36) == date(2021, 2, 28)
    assert shift_months(date(2023, 1, 31), 1) == date(2023, 2, 28)


def test_resolve_default_window_is_twelve_months():
    result = resolve_period("分析人工智能现状", date(2026, 9, 8))
    assert result.valid
    assert result.start_date == date(2025, 9, 8)
    assert result.end_date == date(2026, 9, 8)
    assert result.explicit is False


def test_resolve_near_three_years_uses_calendar_arithmetic():
    result = resolve_period("请分析近三年的人工智能现状", date(2026, 9, 8))
    assert result.valid
    assert result.start_date == date(2023, 9, 8)
    assert result.end_date == date(2026, 9, 8)
    assert result.explicit is True


def test_resolve_explicit_cutoff_fixes_end():
    result = resolve_period("截至2025年12月31日的人工智能现状", date(2026, 9, 8))
    assert result.valid
    assert result.end_date == date(2025, 12, 31)
    assert result.explicit is True


def test_resolve_conflicting_cutoffs_is_invalid():
    result = resolve_period(
        "截至2025年6月1日或截至2025年12月31日", date(2026, 9, 8), default_months=12
    )
    assert not result.valid
    assert "multiple_cutoff_dates" in result.conflicts


def test_resolve_window_before_cutoff_is_invalid():
    # Start bound later than the explicit cutoff leaves an empty reversed window.
    result = resolve_period("自2025年至今，截至2024年12月31日", date(2026, 9, 8))
    assert not result.valid
    assert "window_precedes_cutoff" in result.conflicts


# --- defaults file ---

def test_defaults_file_builtin_and_file_round_trip(tmp_path):
    built = DefaultsFile()
    assert built.period_default_months == 12
    assert built.audience.value == "general"

    path = tmp_path / "defaults.yaml"
    path.write_text(
        "schema_version: '1'\ntimezone: Asia/Shanghai\nlanguage: zh\n"
        "geography: china\naudience: academic\n"
        "length_min_words: 3000\nlength_max_words: 5000\n"
        "report_structure: [背景, 现状, 趋势, 建议]\n"
        "output_format: pdf\nperiod_default_months: 24\n"
        "default_relative_expression: 最近两年\n",
        encoding="utf-8",
    )
    loaded = load_defaults_file(path)
    assert loaded.geography.value == "china"
    assert loaded.audience.value == "academic"
    assert loaded.period_default_months == 24
    assert loaded.region_label == "中国"


def test_defaults_file_rejects_bad_yaml_and_non_mapping(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(": not : [valid", encoding="utf-8")
    with pytest.raises(ValueError):
        load_defaults_file(bad_yaml)

    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("just a string\n", encoding="utf-8")
    with pytest.raises(TypeError):
        load_defaults_file(scalar)


def test_defaults_for_root_falls_back_when_missing(tmp_path):
    defaults = defaults_for_root(tmp_path)
    assert defaults == DefaultsFile()


def test_defaults_for_root_reads_project_config(project_root):
    defaults = defaults_for_root(project_root)
    assert defaults.period_default_months == 12
    assert defaults.output_format == "pdf"


# --- normalize ---

def test_short_request_builds_complete_brief():
    plan = default_brief()
    brief = plan.brief
    assert brief.run_id.startswith("run-")
    assert brief.topic
    assert brief.period.start_date < brief.period.end_date
    assert brief.created_at.tzinfo is not None
    assert plan.notes

    by_field = {item.field: item for item in brief.defaults_applied}
    for key in ("region", "audience", "language", "length", "period", "structure", "output"):
        assert key in by_field, key
    assert by_field["region"].origin is OriginKind.DEFAULT
    assert by_field["period"].origin is OriginKind.DEFAULT
    assert by_field["audience"].origin is OriginKind.DEFAULT


def test_user_region_audience_and_period_are_user_origin():
    plan = normalize_brief(
        "请为中国管理者分析中国人工智能现状，重点关注最近十二个月，篇幅5000字",
        fixed_clock(),
        DefaultsFile(),
    )
    brief = plan.brief
    assert brief.geography.value == "china"
    assert brief.audience.value == "business"
    assert brief.length_target.min_words == 5000

    by_field = {item.field: item for item in brief.defaults_applied}
    assert by_field["region"].origin is OriginKind.USER
    assert by_field["audience"].origin is OriginKind.USER
    assert by_field["period"].origin is OriginKind.USER
    assert by_field["length"].origin is OriginKind.USER


def test_non_report_request_is_rejected():
    with pytest.raises(ValueError, match="not a report request"):
        normalize_brief("你好，帮我订机票", fixed_clock(), DefaultsFile())


def test_extract_topic_best_effort():
    assert extract_topic("请分析一下人工智能发展现状") == "人工智能"
    assert extract_topic("帮我整理一份中国大模型市场报告") == "中国大模型市场"
    assert extract_topic("人工智能") == "人工智能"


def test_brief_display_lines_cover_resolved_fields():
    lines = default_brief().display_lines()
    assert any(line.startswith("主题：") for line in lines)
    assert any("（default）" in line for line in lines)


def test_dialogue_preferences_flow_into_research_brief():
    plan = normalize_brief(
        "分析人工智能现状",
        fixed_clock(),
        DefaultsFile(),
        preferences=[
            PreferenceSelection(field=PreferenceField.WRITING_STYLE, value=WritingStyle.ANALYTICAL.value),
            PreferenceSelection(field=PreferenceField.USAGE_SCENARIO, value=ReaderStyle.STUDENT.value),
        ],
        run_id="run-dialogue-001",
    )

    brief = plan.brief
    assert brief.run_id == "run-dialogue-001"
    assert brief.writing_style is WritingStyle.ANALYTICAL
    assert brief.audience is ReaderStyle.STUDENT
    origins = {item.field: item.origin for item in brief.defaults_applied}
    assert origins["writing_style"] is OriginKind.USER
    assert origins["audience"] is OriginKind.USER
    assert origins["length"] is OriginKind.DEFAULT


def test_dialogue_other_preferences_are_preserved_or_safely_defaulted():
    plan = normalize_brief(
        "分析人工智能现状",
        fixed_clock(),
        DefaultsFile(),
        preferences=[
            PreferenceSelection(
                field=PreferenceField.WRITING_STYLE,
                value=OTHER_OPTION,
                custom_text="课堂案例式说明",
            ),
            PreferenceSelection(
                field=PreferenceField.USAGE_SCENARIO,
                value=OTHER_OPTION,
                custom_text="面向非技术管理者的内部培训",
            ),
        ],
        run_id="run-dialogue-002",
    )

    brief = plan.brief
    assert brief.custom_writing_style == "课堂案例式说明"
    assert brief.custom_usage_scenario == "面向非技术管理者的内部培训"
    assert brief.length_target.label == "4000-6000字"
