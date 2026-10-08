"""Shared identifiers, version handling and base model behaviour.

All internal business contracts derive from :class:`ProjectModel` so that a
structure-level ``schema_version`` is always present and unknown extra fields
are rejected instead of silently passing through.
"""

from __future__ import annotations

import re
import secrets
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = "1"
_VERSION_PATTERN = re.compile(r"^\d+$")


class ProjectModel(BaseModel):
    """Frozen contract base; every payload records its schema version."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = SCHEMA_VERSION


def new_id(prefix: str, *, now: datetime | None = None, entropy_bytes: int = 6) -> str:
    """Produce a readable unique identifier for a run, job or artifact.

    The stamp is derived from the supplied clock (UTC when omitted) so tests
    can make identifiers deterministic by passing a fixed time.
    """
    at = now or datetime.now(UTC)
    stamp = at.strftime("%Y%m%dT%H%M%S")
    return f"{prefix}-{stamp}-{secrets.token_hex(entropy_bytes)}"


def validate_version(value: str) -> str:
    """Allow only non-negative integer version labels such as ``"1"``."""
    if not _VERSION_PATTERN.fullmatch(value):
        raise ValueError("version must be a non-negative integer string")
    return value


def next_version(value: str) -> str:
    """Return the next integer version label."""
    return str(int(value) + 1)
