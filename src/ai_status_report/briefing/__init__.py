"""Briefing: entry intent, defaults and calendar-safe brief normalization."""

from ai_status_report.briefing.clock import Clock, FixedClock, SystemClock
from ai_status_report.briefing.defaults import (
    GEO_LABELS,
    DefaultsFile,
    defaults_for_root,
    load_defaults_file,
)
from ai_status_report.briefing.intents import Intent, IntentDecision, classify
from ai_status_report.briefing.normalize import BriefPlan, extract_topic, normalize_brief
from ai_status_report.briefing.timewindows import (
    DEFAULT_EXPRESSION,
    DEFAULT_MONTHS,
    PeriodResolution,
    resolve_period,
    shift_months,
)

__all__ = [
    "DEFAULT_EXPRESSION",
    "DEFAULT_MONTHS",
    "GEO_LABELS",
    "BriefPlan",
    "Clock",
    "DefaultsFile",
    "FixedClock",
    "Intent",
    "IntentDecision",
    "PeriodResolution",
    "SystemClock",
    "classify",
    "defaults_for_root",
    "extract_topic",
    "load_defaults_file",
    "normalize_brief",
    "resolve_period",
    "shift_months",
]
