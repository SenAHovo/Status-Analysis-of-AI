"""Normalize a short request into a complete, traceable ResearchBrief.

Defaults are resolved field by field and every resolved value records its
origin (user / default). The rule layer is intentionally small and
documented; it keeps the acceptance behaviour testable while model-assisted
routing is wired on top of these boundaries.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC

from ai_status_report.briefing.clock import Clock
from ai_status_report.briefing.defaults import GEO_LABELS, DefaultsFile
from ai_status_report.briefing.intents import classify
from ai_status_report.briefing.timewindows import resolve_period
from ai_status_report.schemas import (
    DEFAULT_WRITING_STYLE,
    OTHER_OPTION,
    PREFERENCE_OPTION_LABELS,
    DefaultedField,
    Geoscope,
    LengthTarget,
    OriginKind,
    PeriodSpec,
    PreferenceField,
    PreferenceSelection,
    ReaderStyle,
    ResearchBrief,
    WritingStyle,
)
from ai_status_report.schemas.common import new_id

_POLITE = re.compile(r"^\s*(?:请|麻烦|帮我|帮我一下|帮忙|我想|我想请|麻烦请|想要|需要|可以帮我)\s*")

_ACTION_ORDER = (
    "生成一份", "写一份", "做一份", "出一份", "整理一份", "调查一下", "调研一下",
    "分析一下", "研究一下", "分析", "研究", "调研", "调查", "综述", "梳理",
    "评估", "盘点", "追踪", "概览", "总结", "整理",
)
_TAIL_NOUNS = re.compile(r"(?:的)?(?:发展现状|现状与趋势|现状|进展|趋势|发展|报告|综述|白皮书|分析|研究)$")

_AUDIENCE_RULES: tuple[tuple[tuple[str, ...], ReaderStyle], ...] = (
    (("学生", "面向学生", "给学生的", "给大学生", "初学者", "新手"), ReaderStyle.STUDENT),
    (("学术", "科研", "研究者", "研究人员", "论文", "老师", "教授"), ReaderStyle.ACADEMIC),
    (("管理", "高管", "管理层", "企业负责人", "公司领导", "老板", "决策者", "董事会"), ReaderStyle.BUSINESS),
)

_LENGTH_RANGE = re.compile(r"(\d{3,5})\s*[-~至]\s*(\d{3,5})\s*(?:个)?(?:中|汉)?字")
_LENGTH_SINGLE = re.compile(r"约?\s*(\d{3,5})\s*(?:个)?(?:中|汉)?字")

_CN_YEAR_SPAN = re.compile(r"(?:自|从)\d{4}\s*年(?:起|开始|至今)|(?:\d{4}\s*年\s*(?:以来|至今))")


def _detect_geoscope(text: str) -> Geoscope | None:
    china = any(tok in text for tok in ("中国", "国内", "本土"))
    global_ = any(tok in text for tok in ("全球", "世界", "国际", "海外", "国外"))
    if china and global_:
        return Geoscope.GLOBAL_WITH_CHINA
    if china:
        return Geoscope.CHINA
    if global_:
        return Geoscope.GLOBAL
    return None


def _detect_audience(text: str) -> ReaderStyle | None:
    for markers, style in _AUDIENCE_RULES:
        if any(marker in text for marker in markers):
            return style
    return None


def _detect_length(text: str) -> tuple[int, int] | None:
    match = _LENGTH_RANGE.search(text)
    if match:
        low, high = int(match.group(1)), int(match.group(2))
        return (min(low, high), max(low, high))
    match = _LENGTH_SINGLE.search(text)
    if match:
        value = int(match.group(1))
        return (value, value)
    return None


def extract_topic(text: str) -> str:
    """Best-effort subject extraction; deterministic and unit tested."""
    topic = _POLITE.sub("", text)
    earliest: tuple[int, str] | None = None
    for verb in _ACTION_ORDER:
        index = topic.find(verb)
        if index != -1 and (earliest is None or index < earliest[0]):
            earliest = (index, verb)
    if earliest is not None:
        index, verb = earliest
        topic = topic[index + len(verb):]
    topic = re.sub(r"^[一份篇下个的]+", "", topic)
    # Keep the first clause of a longer sentence.
    topic = re.split(r"[，。；,.!?？\n]", topic)[0]
    # Drop explicit time spans already captured by the period resolver.
    topic = _CN_YEAR_SPAN.sub("", topic)
    topic = _TAIL_NOUNS.sub("", topic)
    topic = re.sub(r"^(?:与|及|和|、)*", "", topic)
    return topic.strip().strip("的")


@dataclass(frozen=True)
class BriefPlan:
    """Normalized brief plus human-readable notes for the run display."""

    brief: ResearchBrief
    notes: list[str] = field(default_factory=list)

    def display_lines(self) -> list[str]:
        lines = [f"主题：{self.brief.topic}（用户）"]
        resolved = {item.field: item for item in self.brief.defaults_applied}
        ordered = [
            "region",
            "audience",
            "writing_style",
            "language",
            "length",
            "period",
            "structure",
            "output",
        ]
        for key in ordered:
            item = resolved.get(key)
            if item is not None:
                label = item.value
                for field, labels in PREFERENCE_OPTION_LABELS.items():
                    if item.field == field.value:
                        label = labels.get(item.value, item.value)
                        break
                lines.append(f"{item.field}：{label}（{item.origin.value}）")
        return lines


def _set(
    target: dict[str, DefaultedField],
    field_name: str,
    value: str,
    origin: OriginKind,
) -> None:
    target[field_name] = DefaultedField(field=field_name, value=value, origin=origin)


def _selection_map(
    preferences: Sequence[PreferenceSelection] | None,
) -> dict[PreferenceField, PreferenceSelection]:
    selections = list(preferences or ())
    result: dict[PreferenceField, PreferenceSelection] = {}
    for selection in selections:
        if selection.field in result:
            raise ValueError(f"duplicate preference: {selection.field.value}")
        result[selection.field] = selection
    return result


def normalize_brief(
    text: str,
    clock: Clock,
    defaults: DefaultsFile,
    *,
    preferences: Sequence[PreferenceSelection] | None = None,
    run_id: str | None = None,
) -> BriefPlan:
    """Build a complete brief from any report request text.

    ``text`` must already be classified as a report request; otherwise a
    ValueError is raised so callers never invent a topic silently.
    """
    cleaned = " ".join(text.split())
    decision = classify(cleaned)
    if decision.kind.value != "report_request":
        raise ValueError("not a report request")

    resolution = resolve_period(
        cleaned,
        clock.today(),
        defaults.period_default_months,
        defaults.default_relative_expression,
    )
    if not resolution.valid:
        raise ValueError(f"period unresolved: {','.join(resolution.conflicts)}")

    topic = extract_topic(cleaned) or "人工智能现状"

    selected = _selection_map(preferences)
    applied: dict[str, DefaultedField] = {}
    _set(applied, "language", defaults.language, OriginKind.DEFAULT)

    writing_selection = selected.get(PreferenceField.WRITING_STYLE)
    writing_style = DEFAULT_WRITING_STYLE
    custom_writing_style = None
    if writing_selection is None:
        _set(applied, "writing_style", writing_style.value, OriginKind.DEFAULT)
    elif writing_selection.value == OTHER_OPTION:
        custom_writing_style = writing_selection.resolved_text
        _set(applied, "writing_style", custom_writing_style, OriginKind.USER)
    else:
        writing_style = WritingStyle(writing_selection.value)
        _set(applied, "writing_style", writing_style.value, OriginKind.USER)

    scope = _detect_geoscope(cleaned)
    geography = scope if scope is not None else defaults.geography
    _set(
        applied,
        "region",
        GEO_LABELS[geography],
        OriginKind.USER if scope is not None else OriginKind.DEFAULT,
    )

    usage_selection = selected.get(PreferenceField.USAGE_SCENARIO)
    custom_usage_scenario = None
    if usage_selection is not None and usage_selection.value != OTHER_OPTION:
        reader_style = ReaderStyle(usage_selection.value)
        audience = reader_style
    elif usage_selection is not None:
        reader_style = defaults.audience
        audience = None
        custom_usage_scenario = usage_selection.resolved_text
    else:
        audience = _detect_audience(cleaned)
        reader_style = audience if audience is not None else defaults.audience
    _set(
        applied,
        "audience",
        custom_usage_scenario or reader_style.value,
        OriginKind.USER if usage_selection is not None or audience is not None else OriginKind.DEFAULT,
    )

    length = _detect_length(cleaned)
    if length is not None:
        low, high = length
        min_words, max_words = min(low, high), max(low, high)
        label = f"约{min_words}字" if min_words == max_words else f"{min_words}-{max_words}字"
        length_target = LengthTarget(min_words=min_words, max_words=max_words, label=label)
        _set(applied, "length", label, OriginKind.USER)
    else:
        length_target = LengthTarget(
            min_words=defaults.length_min_words,
            max_words=defaults.length_max_words,
            label=f"{defaults.length_min_words}-{defaults.length_max_words}字",
        )
        _set(applied, "length", length_target.label, OriginKind.DEFAULT)

    _set(applied, "structure", "、".join(defaults.report_structure), OriginKind.DEFAULT)

    period_label = (
        f"{resolution.start_date.isoformat()} 至 {resolution.end_date.isoformat()}"
        f"（{resolution.expression}，{defaults.timezone}）"
    )
    period_is_default = not resolution.explicit
    _set(applied, "period", period_label, OriginKind.DEFAULT if period_is_default else OriginKind.USER)
    _set(applied, "output", defaults.output_format, OriginKind.DEFAULT)

    now_local = clock.now()
    brief = ResearchBrief(
        run_id=run_id or new_id("run", now=now_local),
        topic=topic,
        audience=reader_style,
        custom_usage_scenario=custom_usage_scenario,
        writing_style=writing_style,
        custom_writing_style=custom_writing_style,
        geography=geography,
        region=GEO_LABELS[geography],
        period=PeriodSpec(
            start_date=resolution.start_date,
            end_date=resolution.end_date,
            relative_expression=resolution.expression,
            zone=defaults.timezone,
        ),
        language=defaults.language,
        length_target=length_target,
        report_structure=list(defaults.report_structure),
        output_format=defaults.output_format,
        defaults_applied=[applied[key] for key in sorted(applied)],
        created_at=now_local.astimezone(UTC),
    )

    notes = [
        (
            f"本次将按以下范围生成报告：主题「{topic}」，{GEO_LABELS[geography]}，"
            f"读者 {reader_style.value}，研究窗口 {period_label}。"
        )
    ]
    if custom_usage_scenario:
        notes.append(f"使用场景采用用户补充：{custom_usage_scenario}。")
    if custom_writing_style:
        notes.append(f"写作风格采用用户补充：{custom_writing_style}。")
    if scope is not None:
        notes.append("检测到明确的地域范围，已采用用户指定。")
    return BriefPlan(brief=brief, notes=notes)
