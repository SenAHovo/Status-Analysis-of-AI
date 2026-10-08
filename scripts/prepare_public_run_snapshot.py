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

DIRECTORIES = (
    "chapters",
    "evidence",
    "evidence_bundles",
    "outlines",
    "raw",
    "reports",
    "reviews",
    "traces",
)
ROOT_FILES = ("token_usage.json",)
WINDOWS_PATH = re.compile(r"(?i)[c-e]:(?:\\+(?!/)[^\r\n\"']+|/(?!/)[^\r\n\"']+)")
CREDENTIAL_FIELD = re.compile(r"(?i)(api[_-]?key|authorization|token|secret|password)")
CREDENTIAL_VALUE = re.compile(r"(?i)sk-[a-z0-9_-]{8,}")


def _redact_text(value: str, source: Path) -> tuple[str, int]:
    result = value.replace(str(source), ".").replace(source.as_posix(), ".")
    result, path_count = WINDOWS_PATH.subn("[local-path]", result)
    result, credential_count = CREDENTIAL_VALUE.subn("[redacted]", result)
    return result, path_count + credential_count


def _redact_value(value: Any, source: Path) -> tuple[Any, int]:
    if isinstance(value, str):
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
            safe_item, item_count = _redact_value(item, source)
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
    copied = 0
    for name in DIRECTORIES:
        origin = source / name
        if origin.is_dir():
            shutil.copytree(origin, target / name, dirs_exist_ok=True)
            copied += sum(1 for item in origin.rglob("*") if item.is_file())
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

    manifest = {
        "snapshot_type": "public_teaching_run",
        "source_run_id": source.name.split("__", maxsplit=1)[0],
        "copied_files": copied,
        "redacted_values": redactions,
        "included_directories": included_directories,
        "included_root_files": list(ROOT_FILES),
        "excluded": ["token_usage.lock", "local virtual environments", "credential files"],
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
