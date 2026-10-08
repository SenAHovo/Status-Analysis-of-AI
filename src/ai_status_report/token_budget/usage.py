"""Provider usage parsing: actuals are separated from estimates."""

from __future__ import annotations

ALLOWED = ("prompt_tokens", "completion_tokens", "total_tokens")


def usage_tokens(usage: dict) -> int:
    """Total billed tokens when reported; 0 when absent or malformed."""
    value = usage.get("total_tokens")
    if type(value) is int and value >= 0:
        return value
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if type(prompt) is int and type(completion) is int and prompt >= 0 and completion >= 0:
        return prompt + completion
    return 0


def parse_usage(result: dict) -> dict:
    """Return only allow-listed, non-negative integer usage counters.

    A strict ``int`` type check keeps booleans (an ``int`` subclass) out of
    the accounting counters.
    """
    usage = result.get("usage") or {}
    if not isinstance(usage, dict):
        return {}
    return {
        key: usage[key]
        for key in ALLOWED
        if type(usage.get(key)) is int and usage[key] >= 0
    }
