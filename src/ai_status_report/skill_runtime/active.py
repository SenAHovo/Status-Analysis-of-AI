"""Active skill registry for one task run.

Tracks which skill versions are loaded, for which phase, so the context
builder can prove that required guidance is present in a model request.
Registration is de-duplicated by content hash; conflicting versions of the
same skill inside one run are refused. Snapshots support restoring the fixed
versions after context compression.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from ai_status_report.skill_runtime.loader import LoadedSkill, SkillError


class MissingSkillError(RuntimeError):
    """Blocked business stage; message lists missing skill names only."""


@dataclass(frozen=True)
class SkillUse:
    name: str
    description: str
    body: str
    version_hash: str
    phase: str
    registered_at: str


class ActiveSkills:
    def __init__(self):
        self._skills: dict[str, SkillUse] = {}
        self._by_hash: dict[str, str] = {}
        self._phases: dict[str, set[str]] = {}

    def register(self, loaded: LoadedSkill, phase: str) -> bool:
        """Register one skill version for a phase; True means newly added."""
        key = loaded.name
        existing = self._skills.get(key)
        if existing is not None:
            if existing.version_hash == loaded.content_hash and phase in self._phases.get(key, set()):
                return False  # already loaded for this phase; deduplicated
            if existing.version_hash != loaded.content_hash:
                raise SkillError(f"skill_version_conflict:{key}")
            # The same version can be reused, while only the current phase is active.
            self._phases[key] = {phase}
            return True
        skill_use = SkillUse(
            name=loaded.name,
            description=loaded.description,
            body=loaded.body,
            version_hash=loaded.content_hash,
            phase=phase,
            registered_at=_now(),
        )
        self._skills[key] = skill_use
        self._by_hash[loaded.content_hash] = key
        self._phases[key] = {phase}
        return True

    def is_active(self, name: str, phase: str | None = None) -> bool:
        use = self._skills.get(name)
        return use is not None and (phase is None or phase in self._phases.get(name, {use.phase}))

    def version_hash(self, name: str) -> str | None:
        use = self._skills.get(name)
        return use.version_hash if use is not None else None

    def ensure_required(self, required: Iterable[tuple[str, str]]) -> None:
        """Raise MissingSkillError listing every missing or wrong-phase skill."""
        missing = []
        for name, phase in required:
            if not self.is_active(name, phase):
                missing.append(f"{name}@{phase}")
        if missing:
            raise MissingSkillError("required_skills_missing:" + ",".join(sorted(missing)))

    def guidance(self, names: Iterable[str]) -> list[SkillUse]:
        """Return loaded skill uses in stable order for context assembly."""
        result = []
        seen: set[str] = set()
        for name in names:
            use = self._skills.get(name)
            if use is None or use.version_hash in seen:
                continue
            seen.add(use.version_hash)
            result.append(use)
        return result

    def items(self) -> list[SkillUse]:
        return list(self._skills.values())

    def snapshot(self) -> dict:
        return {
            "skills": [self._skill_dict(use) for use in self._skills.values()],
            "phases": {name: sorted(phases) for name, phases in self._phases.items()},
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> ActiveSkills:
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("skills"), list):
            raise SkillError("invalid_skill_snapshot")
        raw_phases = snapshot.get("phases", {})
        if not isinstance(raw_phases, dict):
            raise SkillError("invalid_skill_snapshot_phases")
        registry = cls()
        for raw in snapshot["skills"]:
            if not isinstance(raw, dict):
                raise SkillError("invalid_skill_snapshot_entry")
            required = {"name", "description", "body", "version_hash", "phase", "registered_at"}
            if set(raw) != required or not isinstance(raw["version_hash"], str) or len(raw["version_hash"]) != 64:
                raise SkillError("invalid_skill_snapshot_entry")
            use = SkillUse(**raw)
            if use.name in registry._skills:
                raise SkillError(f"duplicate_skill_snapshot:{use.name}")
            registry._skills[use.name] = use
            registry._by_hash[use.version_hash] = use.name
        registry._phases = {}
        for name, items in raw_phases.items():
            if name not in registry._skills or not isinstance(items, list) or not all(isinstance(item, str) for item in items):
                raise SkillError("invalid_skill_snapshot_phases")
            registry._phases[name] = set(items)
        for name, use in registry._skills.items():
            registry._phases.setdefault(name, {use.phase})
            if not registry._phases[name]:
                raise SkillError("invalid_skill_snapshot_phases")
        return registry

    @staticmethod
    def _skill_dict(use: SkillUse) -> dict:
        return {
            "name": use.name,
            "description": use.description,
            "body": use.body,
            "version_hash": use.version_hash,
            "phase": use.phase,
            "registered_at": use.registered_at,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat()
