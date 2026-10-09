"""Read-only public and local credential audit without printing secret values."""

from __future__ import annotations

import argparse
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path

from dotenv import dotenv_values

from ai_status_report.settings import KEYS

PUBLIC_SECRET_PATTERNS = (
    re.compile(rb"(?i)\bsk-(?!(?:p\d+-placeholder-only|test-|example-))[a-z0-9_-]{16,}\b"),
    re.compile(rb"(?i)\btvly-[a-z0-9_-]{16,}\b"),
    re.compile(rb"-----BEGIN (?:RSA|EC|OPENSSH|DSA|PGP) PRIVATE KEY-----"),
    re.compile(rb"(?i)authorization\s*[:=]\s*[\"']?bearer\s+[a-z0-9._~+/-]{16,}"),
)
KNOWN_CREDENTIAL_ASSIGNMENT = re.compile(
    rb"(?im)\b(?:DEEPSEEK_API_KEY|GLM_OCR_API_KEY|GLM_EMBEDDING_API_KEY|TAVILY_API_KEY)"
    rb"[ \t]*=[ \t]*[\"']?([a-z0-9_.-]{12,})"
)
PLACEHOLDER_VALUE_MARKERS = (b"placeholder", b"test", b"example", b"keep-existing")


def _candidate_paths(root: Path) -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "."],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    paths = [root / name.decode("utf-8") for name in listed.split(b"\0") if name]
    return paths + list((root / "outputs" / "d01").glob("*.json"))


def _archive_members(root: Path) -> list[bytes]:
    contents: list[bytes] = []
    for archive in (root / "dist").glob("*"):
        if archive.suffix == ".whl":
            with zipfile.ZipFile(archive) as package:
                contents.extend(package.read(name) for name in package.namelist())
        elif archive.name.endswith(".tar.gz"):
            with tarfile.open(archive) as package:
                for member in package.getmembers():
                    if member.isfile():
                        file = package.extractfile(member)
                        if file is not None:
                            contents.append(file.read())
    return contents


def _generic_hits(content: bytes) -> int:
    if any(pattern.search(content) for pattern in PUBLIC_SECRET_PATTERNS):
        return 1
    assignments = KNOWN_CREDENTIAL_ASSIGNMENT.findall(content)
    return int(any(not any(marker in value.lower() for marker in PLACEHOLDER_VALUE_MARKERS) for value in assignments))


def _local_secrets(root: Path) -> list[bytes]:
    values = dotenv_values(root / ".env", interpolate=False)
    return [values[key].encode() for key in KEYS if values.get(key)]


def audit(root: Path, *, mode: str = "auto") -> tuple[int, int, bool]:
    """Return checked item count, finding count, and whether exact matching ran."""

    if mode not in {"auto", "public", "local"}:
        raise ValueError("audit_mode_invalid")
    paths = _candidate_paths(root)
    contents = [path.read_bytes() for path in paths if path.is_file()]
    contents.extend(_archive_members(root))
    findings = sum(_generic_hits(content) for content in contents)

    secrets = _local_secrets(root)
    exact_enabled = len(secrets) == len(KEYS) and mode != "public"
    if mode == "local" and not exact_enabled:
        raise ValueError("local_credential_slots_missing")
    if exact_enabled:
        findings += sum(int(any(secret in content for secret in secrets)) for content in contents)

    for path in (".env",):
        ignored = subprocess.run(["git", "check-ignore", "-q", "--", path], cwd=root, check=False)
        tracked = subprocess.run(
            ["git", "ls-files", "-z", "--", path], cwd=root, capture_output=True, check=True
        )
        findings += int(ignored.returncode != 0 or bool(tracked.stdout))
    return len(contents), findings, exact_enabled


def main() -> int:
    parser = argparse.ArgumentParser(description="scan public files for secret exposure")
    parser.add_argument("--mode", choices=("auto", "public", "local"), default="auto")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        checked, findings, exact_enabled = audit(root, mode=args.mode)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    exact_mode = "enabled" if exact_enabled else "skipped"
    print(
        f"Checked items: {checked}; secret/security findings: {findings}; exact local scan: {exact_mode}"
    )
    return int(findings > 0)


if __name__ == "__main__":
    raise SystemExit(main())
