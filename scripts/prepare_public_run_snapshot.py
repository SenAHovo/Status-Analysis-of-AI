"""Copy one completed run into ``examples/`` and redact local-only strings.

The command is intentionally explicit: it never discovers or deletes runs.
It copies the selected source run into the selected public example directory,
keeps the process artifacts needed for teaching, and rewrites text artifacts so
local absolute paths and credential-shaped values are not published.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

DIRECTORIES = ("chapters", "outlines", "reviews", "traces")
REPORT_FILES = ("report.md", "report.pdf")
ROOT_FILES = ("token_usage.json",)
PRIVATE_ARTIFACT_DIRECTORIES = ("raw", "evidence", "evidence_bundles")
SOURCE_MANIFEST_FILE = "sources.json"
WINDOWS_PATH = re.compile(r"(?i)[c-e]:(?:\\+(?!/)[^\r\n\"']+|/(?!/)[^\r\n\"']+)")
CREDENTIAL_FIELD = re.compile(r"(?i)(api[_-]?key|authorization|token|secret|password)")
CREDENTIAL_VALUE = re.compile(r"(?i)sk-[a-z0-9_-]{8,}")
EXTERNAL_TEXT_FIELDS = {"snippet", "raw_content", "content"}
PRIVATE_REFERENCE_FIELDS = {
    "content_ref",
    "content_refs",
    "evidence_bundle_ref",
    "raw_ref",
    "raw_refs",
}
SENSITIVE_URL_QUERY_KEY = re.compile(
    r"(?i)^(?:api[_-]?key|authorization|token|secret|password|signature|sig|x-amz-signature|x-goog-signature)$"
)


def _redact_text(value: str, source: Path) -> tuple[str, int]:
    result = value.replace(str(source), ".").replace(source.as_posix(), ".")
    result, path_count = WINDOWS_PATH.subn("[local-path]", result)
    result, credential_count = CREDENTIAL_VALUE.subn("[redacted]", result)
    return result, path_count + credential_count


def _public_url(value: str) -> str:
    """Keep a source location while removing credential-bearing query parameters."""

    try:
        parsed = urlsplit(value)
    except ValueError:
        return "[not-published]"
    query = [(key, item) for key, item in parse_qsl(parsed.query, keep_blank_values=True) if not SENSITIVE_URL_QUERY_KEY.fullmatch(key)]
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query, doseq=True), ""))


def _redact_value(value: Any, source: Path, *, field_name: str = "") -> tuple[Any, int]:
    if field_name in EXTERNAL_TEXT_FIELDS:
        return "[not-published]", int(bool(value))
    if field_name in PRIVATE_REFERENCE_FIELDS:
        return ([] if isinstance(value, list) else ""), int(bool(value))
    if isinstance(value, str):
        stripped = value.lstrip()
        if field_name in {"summary", "text"} and stripped.startswith(("{", "[")):
            try:
                embedded = json.loads(value)
            except json.JSONDecodeError:
                pass
            else:
                safe_embedded, count = _redact_value(embedded, source)
                return json.dumps(safe_embedded, ensure_ascii=False, separators=(",", ":")), count
        return _redact_text(value, source)
    if isinstance(value, list):
        count = 0
        items = []
        for item in value:
            safe_item, item_count = _redact_value(item, source)
            items.append(safe_item)
            count += item_count
        return items, count
    if isinstance(value, dict):
        count = 0
        safe: dict[str, Any] = {}
        for key, item in value.items():
            if CREDENTIAL_FIELD.fullmatch(str(key)):
                if item:
                    safe[str(key)] = "[redacted]"
                    count += 1
                else:
                    safe[str(key)] = item
                continue
            safe_item, item_count = _redact_value(item, source, field_name=str(key))
            safe[str(key)] = safe_item
            count += item_count
        return safe, count
    return value, 0


def _rewrite_json(path: Path, source: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _rewrite_text(path, source)
    safe, count = _redact_value(data, source)
    path.write_text(json.dumps(safe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return count


def _rewrite_jsonl(path: Path, source: Path) -> int:
    lines: list[str] = []
    count = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            data = json.loads(raw_line)
        except json.JSONDecodeError:
            safe_line, item_count = _redact_text(raw_line, source)
        else:
            safe_data, item_count = _redact_value(data, source)
            safe_line = json.dumps(safe_data, ensure_ascii=False, separators=(",", ":"))
        lines.append(safe_line)
        count += item_count
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return count


def _rewrite_text(path: Path, source: Path) -> int:
    text = path.read_text(encoding="utf-8")
    safe, count = _redact_text(text, source)
    path.write_text(safe, encoding="utf-8")
    return count


def _source_manifest(source: Path) -> list[dict[str, str]]:
    """Return source metadata without copying provider excerpts or page content."""

    records: dict[tuple[str, str], dict[str, str]] = {}
    for path in sorted((source / "evidence").glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(item, dict):
            continue
        source_id = item.get("source_id")
        url = item.get("url")
        if not isinstance(source_id, str) or not isinstance(url, str) or not source_id or not url:
            continue
        records.setdefault(
            (source_id, url),
            {
                "source_id": source_id,
                "title": str(item.get("title") or ""),
                "url": _public_url(url),
                "provider": str(item.get("provider") or ""),
                "verification_status": str(item.get("verification_status") or ""),
                "content_sha256": str(item.get("content_hash") or ""),
            },
        )
    return sorted(records.values(), key=lambda item: (item["source_id"], item["url"]))


def _clear_managed_artifacts(target: Path) -> None:
    """Remove only snapshot artifacts that this script owns before re-copying."""

    for name in (*DIRECTORIES, *PRIVATE_ARTIFACT_DIRECTORIES, "reports", "service_logs"):
        path = target / name
        if path.is_dir():
            shutil.rmtree(path)
    for name in (*ROOT_FILES, SOURCE_MANIFEST_FILE, "public_snapshot_manifest.json"):
        path = target / name
        if path.is_file():
            path.unlink()


def prepare_snapshot(
    source: Path, target: Path, service_logs: Path | None = None
) -> dict[str, Any]:
    source = source.resolve()
    target = target.resolve()
    if not source.is_dir():
        raise ValueError("source_run_not_found")
    if source == target:
        raise ValueError("source_and_target_must_differ")

    target.mkdir(parents=True, exist_ok=True)
    _clear_managed_artifacts(target)
    copied = 0
    for name in DIRECTORIES:
        origin = source / name
        if origin.is_dir():
            shutil.copytree(origin, target / name, dirs_exist_ok=True)
            copied += sum(1 for item in origin.rglob("*") if item.is_file())
    for name in REPORT_FILES:
        origin = source / "reports" / name
        if origin.is_file():
            (target / "reports").mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target / "reports" / name)
            copied += 1
    for name in ROOT_FILES:
        origin = source / name
        if origin.is_file():
            shutil.copy2(origin, target / name)
            copied += 1

    included_directories = list(DIRECTORIES)
    if service_logs is not None:
        service_logs = service_logs.resolve()
        if not service_logs.is_dir():
            raise ValueError("service_logs_not_found")
        shutil.copytree(service_logs, target / "service_logs", dirs_exist_ok=True)
        copied += sum(1 for item in service_logs.rglob("*") if item.is_file())
        included_directories.append("service_logs")

    redactions = 0
    for path in target.rglob("*"):
        if not path.is_file() or path.name == "README.md":
            continue
        if path.suffix == ".json":
            redactions += _rewrite_json(path, source)
        elif path.suffix == ".jsonl":
            redactions += _rewrite_jsonl(path, source)
        elif path.suffix in {".md", ".txt", ".log"}:
            redactions += _rewrite_text(path, source)

    source_manifest = _source_manifest(source)
    (target / SOURCE_MANIFEST_FILE).write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    copied += 1

    manifest = {
        "snapshot_type": "public_teaching_run",
        "source_run_id": source.name.split("__", maxsplit=1)[0],
        "copied_files": copied,
        "redacted_values": redactions,
        "included_directories": included_directories,
        "included_report_files": list(REPORT_FILES),
        "included_root_files": list(ROOT_FILES),
        "source_manifest": SOURCE_MANIFEST_FILE,
        "source_count": len(source_manifest),
        "excluded": [
            "raw provider responses and crawled web pages",
            "EvidenceChunk text and chapter evidence bundles",
            "search report snippets",
            "token_usage.lock",
            "local virtual environments",
            "credential files",
        ],
    }
    (target / "public_snapshot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="prepare one public, redacted run snapshot")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument(
        "--service-logs",
        type=Path,
        help="optional teaching-session service log directory associated with the run",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_snapshot(args.source_run, args.target_run, args.service_logs),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
