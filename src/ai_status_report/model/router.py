"""Real model routing with a local response cache.

The deterministic intent classifier is the zero-token fast path. DeepSeek is
called only when the caller explicitly requests model routing for an uncertain
utterance. Cache keys include the complete normalized request and provider
profile, so a changed prompt/model cannot reuse an incompatible answer.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from pydantic import ValidationError

from ai_status_report.briefing.intents import Intent, classify
from ai_status_report.model.client import DeepSeekClient, ProviderError
from ai_status_report.model.structured import StructuredOutputError, parse_structured
from ai_status_report.schemas.common import ProjectModel


class ModelRoute(ProjectModel):
    intent: Intent
    reason: str
    confidence: float
    source: str = "model"


class ModelRouteError(RuntimeError):
    """Safe model-routing failure for callers that require strict routing."""


def _cache_key(model: str, messages: list[dict], options: dict) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "options": options},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ResponseCache:
    """Small SQLite cache for completed, non-streaming, no-tool requests."""

    def __init__(self, path: Path, ttl_seconds: int = 86_400, max_entries: int = 500):
        self.path = path
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS responses (cache_key TEXT PRIMARY KEY, created REAL NOT NULL, payload TEXT NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS responses_created ON responses(created)")

    def get(self, key: str) -> dict | None:
        cutoff = time.time() - self.ttl_seconds
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT created, payload FROM responses WHERE cache_key = ?", (key,)).fetchone()
            if row is None:
                return None
            if row[0] < cutoff:
                db.execute("DELETE FROM responses WHERE cache_key = ?", (key,))
                return None
            try:
                payload = json.loads(row[1])
            except json.JSONDecodeError:
                db.execute("DELETE FROM responses WHERE cache_key = ?", (key,))
                return None
            if not isinstance(payload, dict):
                db.execute("DELETE FROM responses WHERE cache_key = ?", (key,))
                return None
            return payload

    def put(self, key: str, payload: dict) -> None:
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT OR REPLACE INTO responses(cache_key, created, payload) VALUES (?, ?, ?)",
                (key, time.time(), serialized),
            )
            db.execute(
                "DELETE FROM responses WHERE cache_key IN "
                "(SELECT cache_key FROM responses ORDER BY created DESC LIMIT -1 OFFSET ?)",
                (self.max_entries,),
            )

    def delete(self, key: str) -> None:
        """Remove one invalid entry before retrying the normal route."""

        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM responses WHERE cache_key = ?", (key,))


class DeepSeekRouter:
    def __init__(self, client: DeepSeekClient, cache: ResponseCache, *, max_output_tokens: int = 96):
        self.client = client
        self.cache = cache
        self.max_output_tokens = max_output_tokens

    def route_intent(
        self,
        text: str,
        *,
        allow_model: bool = False,
        fallback_to_deterministic: bool = True,
    ) -> ModelRoute:
        deterministic = classify(text)
        if not allow_model:
            return ModelRoute(
                intent=deterministic.kind,
                reason=deterministic.reason,
                confidence=1.0,
                source="deterministic",
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "你是项目入口路由器。只输出 JSON。将用户输入分类为 report_request、"
                    "conversation、unsupported。必须包含 intent、reason、confidence。"
                    "凡是询问人工智能、大模型、机器学习等领域的当前现状、行业变化、发展趋势、"
                    "技术进展、市场格局、应用、影响、机会或挑战，都属于 report_request，"
                    "即使用户没有使用‘报告’或‘分析’这些词。只有闲聊、概念解释、问候、"
                    "能力询问才属于 conversation；天气、订票、邮件等范围外任务属于 unsupported。"
                ),
            },
            {"role": "user", "content": text},
        ]
        options = {
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "max_tokens": self.max_output_tokens,
        }
        key = _cache_key(self.client.profile.model, messages, options)
        cached = self.cache.get(key)
        if cached is not None:
            try:
                return ModelRoute.model_validate({**cached, "source": "model_cache"})
            except (TypeError, ValidationError):
                # A stale or manually damaged cache entry must never bypass
                # the normal provider/fallback boundary.
                self.cache.delete(key)
        try:
            result = self.client.chat(messages, **options)
            content = result["choices"][0]["message"]["content"]
            parsed = parse_structured(content, ModelRoute)
            route = parsed.model_copy(update={"source": "model"})
        except (KeyError, IndexError, TypeError, ProviderError, StructuredOutputError):
            if not fallback_to_deterministic:
                raise ModelRouteError("intent_model_failed") from None
            # Batch/report compatibility keeps the deterministic safety boundary.
            return ModelRoute(
                intent=deterministic.kind,
                reason="model_route_fallback:" + deterministic.reason,
                confidence=0.0,
                source="deterministic_fallback",
            )
        self.cache.put(key, route.model_dump(mode="json"))
        return route
