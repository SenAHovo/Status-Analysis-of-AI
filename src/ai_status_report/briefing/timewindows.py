"""Calendar-safe period computation for relative time expressions.

The default scope keeps the run start date as the cutoff and covers the
previous twelve months. Expressions such as ``近三年``, ``最近十二个月`` and
explicit cutoffs use calendar arithmetic, so month ends and leap days behave
like a real calendar: 2024-02-29 minus three years is 2021-02-28. Fixed
30-day spans are never used for month and year expressions.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date

DEFAULT_EXPRESSION = "最近十二个月"
DEFAULT_MONTHS = 12

_CN_DIGITS = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}

_NUM = r"\d{1,3}|[零一二两三四五六七八九十]{1,3}"
_DURATION_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(rf"(?:近|最近|过去|前)\s*({_NUM})\s*(?:个)?月"), 1),
    (re.compile(rf"(?:近|最近|过去|前)\s*({_NUM})\s*年"), 12),
)
_CUTOFF_FULL = re.compile(r"(?:截至|截止)\s*(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?")
# A bare "截至2026年" means end of that calendar year; it must not also
# consume the prefix of a fuller date such as 截至2026年6月30日.
_CUTOFF_YEAR = re.compile(r"(?:截至|截止)\s*(\d{4})\s*年(?!\s*[\d月])")
_START_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:自|从)\s*(\d{4})\s*年(?:起|开始|至今)"),
    re.compile(r"(\d{4})\s*年\s*(?:以来|至今)"),
)


def _cn_to_int(text: str) -> int | None:
    if text.isdigit():
        return int(text)
    if not text or not all(ch in _CN_DIGITS or ch == "十" for ch in text):
        return None
    if text == "十":
        return 10
    if text.startswith("十"):
        ones = _CN_DIGITS.get(text[1])
        return 10 + ones if ones is not None else None
    if "十" in text:
        head, _, tail = text.partition("十")
        tens = _CN_DIGITS.get(head)
        ones = _CN_DIGITS.get(tail) if tail else None
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    return _CN_DIGITS.get(text)


def shift_months(value: date, months: int) -> date:
    """Shift by whole calendar months, clamping the day to the month end."""
    index = value.year * 12 + (value.month - 1) + months
    year, zero_based = divmod(index, 12)
    month = zero_based + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


@dataclass(frozen=True)
class PeriodResolution:
    start_date: date
    end_date: date
    expression: str
    conflicts: tuple[str, ...] = ()
    # True when the user's text supplied an explicit cutoff, start or duration;
    # False when the default window was applied.
    explicit: bool = False

    @property
    def valid(self) -> bool:
        return self.start_date <= self.end_date and not self.conflicts


def _cutoffs(text: str) -> list[date]:
    found: list[date] = []
    for match in _CUTOFF_FULL.finditer(text):
        try:
            found.append(date(int(match.group(1)), int(match.group(2)), int(match.group(3))))
        except ValueError:
            continue
    for match in _CUTOFF_YEAR.finditer(text):
        found.append(date(int(match.group(1)), 12, 31))
    # Keep distinct values in first-seen order.
    return list(dict.fromkeys(found))


def _durations(text: str) -> list[tuple[int, str]]:
    """Return (duration in months, matched expression) pairs in text order."""
    found: list[tuple[int, str]] = []
    for pattern, multiplier in _DURATION_PATTERNS:
        for match in pattern.finditer(text):
            number = _cn_to_int(match.group(1))
            if number is None or number < 1:
                continue
            found.append((number * multiplier, match.group(0).strip()))
    return found


def _start_bounds(text: str) -> list[date]:
    found: list[date] = []
    for pattern in _START_PATTERNS:
        for match in pattern.finditer(text):
            found.append(date(int(match.group(1)), 1, 1))
    return list(dict.fromkeys(found))


def resolve_period(
    text: str,
    anchor: date,
    default_months: int = DEFAULT_MONTHS,
    default_expression: str = DEFAULT_EXPRESSION,
) -> PeriodResolution:
    """Resolve the fixed research window from user text and the run date.

    Precedence: an explicit cutoff fixes the end; a ``YYYY年以来`` bound fixes
    the start; otherwise the duration (or the default twelve months) is applied
    backwards from the resolved end with calendar-safe month arithmetic.
    """
    cutoffs = _cutoffs(text)
    start_bounds = _start_bounds(text)
    durations = _durations(text)
    # User-supplied window clues make the range explicit; otherwise the default
    # window applies and normalize_brief should mark the field as a default.
    explicit = bool(cutoffs or start_bounds or durations)

    if len(cutoffs) > 1:
        return PeriodResolution(anchor, anchor, "ambiguous", ("multiple_cutoff_dates",), explicit)
    if len(start_bounds) > 1:
        return PeriodResolution(anchor, anchor, "ambiguous", ("multiple_start_dates",), explicit)
    if len(durations) > 1 and not cutoffs:
        return PeriodResolution(anchor, anchor, "ambiguous", ("multiple_relative_ranges",), explicit)

    duration = durations[0][0] if durations else default_months
    duration_label = durations[0][1] if durations else default_expression

    end = cutoffs[0] if cutoffs else anchor
    start = start_bounds[0] if start_bounds else shift_months(end, -duration)

    if cutoffs:
        expression = (
            f"{duration_label}，截至{end.isoformat()}" if durations else f"截至{end.isoformat()}"
        )
    elif start_bounds:
        expression = f"自{start_bounds[0].isoformat()}起，{duration_label}"
    else:
        expression = duration_label

    if start > end:
        return PeriodResolution(start, end, expression, ("window_precedes_cutoff",), explicit)
    return PeriodResolution(start, end, expression, explicit=explicit)
