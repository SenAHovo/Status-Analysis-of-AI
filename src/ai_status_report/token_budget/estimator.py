"""Token estimation heuristics.

Estimates are intentionally conservative and separate from provider usage.
They size model input (context assembly) and reserve budgets; the provider
reports the true numbers that are settled afterwards. These are project
heuristics, not claims about a specific tokenizer.
"""

from __future__ import annotations

import math
import re

_ASCII_WORD = re.compile(r"[A-Za-z0-9_]+")
# Characters roughly one token each for CJK; western words average 4 chars.
_CJK_CHAR_RANGE = (0x2E80, 0x9FFF)
_MESSAGE_OVERHEAD = 4
_BLOCK_OVERHEAD = 2


def _count_cjk(text: str) -> int:
    return sum(1 for ch in text if _CJK_CHAR_RANGE[0] <= ord(ch) <= _CJK_CHAR_RANGE[1])


def estimate_tokens(text: str) -> int:
    """Estimate tokens for one text block (message or instruction)."""
    ascii_tokens = sum(math.ceil(len(word) / 4) for word in _ASCII_WORD.findall(text))
    cjk_tokens = _count_cjk(text)
    return max(1, ascii_tokens + cjk_tokens + _BLOCK_OVERHEAD)


def estimate_messages(messages: list[dict]) -> int:
    """Estimate a full message list including per-message overhead."""
    return sum(estimate_tokens(str(message.get("content") or "")) + _MESSAGE_OVERHEAD for message in messages)
