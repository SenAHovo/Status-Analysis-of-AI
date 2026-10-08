"""Read-only audit: print counts only, never matching strings or file content."""

import subprocess
import tarfile
import zipfile
from pathlib import Path

from dotenv import dotenv_values

from ai_status_report.settings import KEYS


def main():
    root = Path(__file__).resolve().parents[1]
    values = dotenv_values(root / ".env", interpolate=False)
    secrets = [values[k].encode() for k in KEYS if values.get(k)]
    if len(secrets) != len(KEYS):
        raise SystemExit("Audit requires all local credential slots; values withheld.")
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "."],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    paths = [root / name.decode("utf-8") for name in listed.split(b"\0") if name]
    paths += list((root / "outputs" / "d01").glob("*.json"))
    paths += [root.parent / "项目方案v1.md", root.parent / "Agent开发技术方案v1.md"]
    hits = 0
    checked = 0
    for path in paths:
        if path.is_file():
            checked += 1
            hits += int(any(key in path.read_bytes() for key in secrets))
    for archive in (root / "dist").glob("*"):
        if archive.suffix == ".whl":
            with zipfile.ZipFile(archive) as package:
                for name in package.namelist():
                    checked += 1
                    hits += int(any(key in package.read(name) for key in secrets))
        elif archive.name.endswith(".tar.gz"):
            with tarfile.open(archive) as package:
                for member in package.getmembers():
                    if member.isfile():
                        checked += 1
                        with package.extractfile(member) as file:
                            content = file.read()
                        hits += int(any(key in content for key in secrets))
    for path in (".env", "../deepseekAPI.txt"):
        ignored = subprocess.run(["git", "check-ignore", "-q", "--", path], cwd=root, check=False)
        tracked = subprocess.run(
            ["git", "ls-files", "-z", "--", path], cwd=root, capture_output=True, check=True
        )
        hits += int(ignored.returncode != 0 or bool(tracked.stdout))
    print(f"Checked items: {checked}; secret/security findings: {hits}")
    return int(hits > 0)


if __name__ == "__main__":
    raise SystemExit(main())
