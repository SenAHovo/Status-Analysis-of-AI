"""Structured output validation for model responses.

A completion string is parsed as JSON and validated against a project
contract (pydantic model). The layer returns typed data or a fixed error code;
content is never echoed into the error message.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class StructuredOutputError(ValueError):
    """Safe reason code only; never embeds the model output."""


def coerce_json(content: str) -> object:
    """Parse strict JSON; tolerate a surrounding code fence and stray text."""
    text = content.strip()
    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    # Fallback: extract the outermost JSON object when the model padded text.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise StructuredOutputError("invalid_json")
    try:
        return json.loads(text[start : end + 1])
    except ValueError as exc:
        raise StructuredOutputError("invalid_json") from exc


def parse_structured[T: BaseModel](content: str, model: type[T]) -> T:
    """Validate ``content`` against ``model``; raises StructuredOutputError."""
    payload = coerce_json(content)
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        # Only schema paths and fixed codes are safe to surface.
        paths = sorted({".".join(str(part) for part in error["loc"]) for error in exc.errors()})
        code = "schema_mismatch" + (f":{','.join(paths)}" if paths else "")
        raise StructuredOutputError(code) from exc
    except (TypeError, ValueError) as exc:
        raise StructuredOutputError("schema_mismatch") from exc


def schema_hint(model: type[BaseModel]) -> str:
    """Compact prompt hint describing the expected JSON contract."""
    schema = model.model_json_schema()
    required = schema.get("required", [])
    props = {name: value.get("type", "object") for name, value in schema.get("properties", {}).items()}
    return f"JSON 对象，字段：{', '.join(props)}；必填：{', '.join(required)}"
