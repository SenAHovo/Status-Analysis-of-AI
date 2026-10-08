"""TTL cache dedicated to DeepSeek native web-search responses."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class WebSearchCache:
    def __init__(self, path: Path, ttl_seconds: int = 3600, max_entries: int = 200):
        self.path = path
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS web_search (cache_key TEXT PRIMARY KEY, created REAL NOT NULL, payload TEXT NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS web_search_created ON web_search(created)")

    @staticmethod
    def key(*, model: str, query: str, options: dict[str, Any]) -> str:
        payload = json.dumps(
            {"model": model, "query": query, "options": options},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT created, payload FROM web_search WHERE cache_key = ?", (key,)).fetchone()
            if row is None or row[0] < time.time() - self.ttl_seconds:
                if row is not None:
                    db.execute("DELETE FROM web_search WHERE cache_key = ?", (key,))
                return None
            return json.loads(row[1])

    def put(self, key: str, payload: dict[str, Any]) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT OR REPLACE INTO web_search(cache_key, created, payload) VALUES (?, ?, ?)", (key, time.time(), json.dumps(payload, ensure_ascii=False)))
            db.execute("DELETE FROM web_search WHERE cache_key IN (SELECT cache_key FROM web_search ORDER BY created DESC LIMIT -1 OFFSET ?)", (self.max_entries,))


def cached_web_search(client, cache: WebSearchCache, query: str, *, max_output_tokens: int = 65536) -> dict[str, Any]:
    options = {"max_output_tokens": max_output_tokens, "tool_choice": "web_search"}
    key = cache.key(model=client.profile.model, query=query, options=options)
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = client.web_search(query, max_output_tokens=max_output_tokens)
    cache.put(key, result)
    return result
