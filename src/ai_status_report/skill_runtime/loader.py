"""Skill discovery and loading from the project ``skills`` directory.

A skill package is a directory containing ``SKILL.md`` with a YAML front
matter block carrying ``name`` and ``description``, followed by the guidance
body and any reference resources. Only names matching the project convention
(lowercase letters, digits and hyphens) are accepted, and the directory name
must equal the declared name. No credential is ever expected inside skills.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

_NAME_PATTERN = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")


def _valid_name(name: str) -> bool:
    return bool(name) and all(ch in _NAME_PATTERN for ch in name)


class SkillError(RuntimeError):
    """Fixed reason codes; messages never contain credential material."""


class SkillNotFoundError(SkillError):
    pass


@dataclass(frozen=True)
class SkillInfo:
    name: str
    description: str
    path: Path


@dataclass(frozen=True)
class LoadedSkill:
    name: str
    description: str
    body: str
    content_hash: str
    source_path: Path

    def guidance(self) -> str:
        return self.body


class SkillLoader:
    """Discover packages under a skill root and load them by name."""

    def __init__(self, skill_root: Path):
        self.skill_root = skill_root

    def list_skills(self) -> list[SkillInfo]:
        packages: list[SkillInfo] = []
        for path in sorted(self.skill_root.iterdir()) if self.skill_root.is_dir() else []:
            if not path.is_dir():
                continue
            skill_file = path / "SKILL.md"
            if not skill_file.is_file():
                continue
            info = self._inspect(path.name, skill_file)
            if info is not None:
                packages.append(info)
        return packages

    def load(self, name: str) -> LoadedSkill:
        if not _valid_name(name):
            raise SkillError("invalid_skill_name")
        path = self.skill_root / name
        if not path.is_dir() or not (path / "SKILL.md").is_file():
            raise SkillNotFoundError(f"skill_not_found:{name}")
        info = self._inspect(name, path / "SKILL.md")
        if info is None:
            raise SkillError(f"skill_metadata_invalid:{name}")
        raw = (path / "SKILL.md").read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        body = _body_of(raw.decode("utf-8"))
        return LoadedSkill(
            name=info.name,
            description=info.description,
            body=body,
            content_hash=digest,
            source_path=path,
        )

    def load_many(self, names: Iterable[str]) -> list[LoadedSkill]:
        return [self.load(name) for name in names]

    def _inspect(self, folder_name: str, skill_file: Path) -> SkillInfo | None:
        if not _valid_name(folder_name):
            return None
        text = skill_file.read_text(encoding="utf-8")
        meta = _front_matter(text)
        if not isinstance(meta, dict):
            return None
        name, description = meta.get("name"), meta.get("description")
        if not isinstance(name, str) or not isinstance(description, str):
            return None
        if not _valid_name(name) or name != folder_name or not description.strip():
            return None
        return SkillInfo(name=name, description=description.strip(), path=skill_file.parent)


def _front_matter(text: str) -> object:
    if not text.startswith("---"):
        return None
    try:
        _, block, _ = text.split("---", 2)
    except ValueError:
        return None
    try:
        return yaml.safe_load(block)
    except yaml.YAMLError:
        return None


def _body_of(text: str) -> str:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2].strip()
    return text.strip()
