"""Skill runtime and context assembly: loader, registry and budgets."""

from pathlib import Path

import pytest

from ai_status_report.context.builder import ContextAssembly, ContextBuilder, ContextError, assemble
from ai_status_report.skill_runtime import (
    ActiveSkills,
    LoadedSkill,
    MissingSkillError,
    SkillError,
    SkillInfo,
    SkillLoader,
    SkillNotFoundError,
)

SKILL_MD = """---
name: research
description: 网络研究执行指导
---
来源优先级：官方报告优先。空结果时更换关键词补查。
"""


def write_skill(root: Path, folder: str, content: str = SKILL_MD) -> Path:
    target = root / folder
    target.mkdir(parents=True, exist_ok=True)
    (target / "SKILL.md").write_text(content, encoding="utf-8")
    return target


def loaded(root: Path) -> LoadedSkill:
    write_skill(root, "research")
    return SkillLoader(root).load("research")


# --- loader ---

def test_list_and_load_valid_skill(tmp_path):
    write_skill(tmp_path, "research")
    loader = SkillLoader(tmp_path)
    infos = loader.list_skills()
    assert infos == [SkillInfo(name="research", description="网络研究执行指导", path=tmp_path / "research")]
    loaded_skill = loader.load("research")
    assert loaded_skill.name == "research"
    assert "来源优先级" in loaded_skill.body
    assert len(loaded_skill.content_hash) == 64


def test_loader_ignores_folders_without_valid_metadata(tmp_path):
    write_skill(tmp_path, "research", content="no front matter")
    write_skill(tmp_path, "research", content="")
    bad = tmp_path / "Bad_Name"
    bad.mkdir()
    (bad / "SKILL.md").write_text("---\nname: Bad_Name\ndescription: x\n---\nbody\n", encoding="utf-8")
    write_skill(tmp_path, "mismatch", content="---\nname: other\ndescription: x\n---\nbody\n")
    assert SkillLoader(tmp_path).list_skills() == []


def test_loader_missing_and_invalid_name(tmp_path):
    loader = SkillLoader(tmp_path)
    with pytest.raises(SkillNotFoundError):
        loader.load("research")
    with pytest.raises(SkillError):
        loader.load("Bad_Name")


def test_load_many_keeps_order(tmp_path):
    write_skill(tmp_path, "research")
    write_skill(tmp_path, "report-writing", content=SKILL_MD.replace("research", "report-writing"))
    loader = SkillLoader(tmp_path)
    loaded_skills = loader.load_many(["research", "report-writing"])
    assert [s.name for s in loaded_skills] == ["research", "report-writing"]


# --- active registry ---

def test_register_dedup_and_version_conflict(tmp_path):
    registry = ActiveSkills()
    first = loaded(tmp_path)
    assert registry.register(first, phase="research") is True
    assert registry.register(first, phase="research") is False  # same phase, dedup

    changed = LoadedSkill(
        name="research",
        description="x",
        body="changed",
        content_hash="a" * 64,
        source_path=tmp_path,
    )
    with pytest.raises(SkillError):
        registry.register(changed, phase="research")


def test_register_same_version_for_new_phase_reuses_without_duplication(tmp_path):
    registry = ActiveSkills()
    skill = loaded(tmp_path)
    assert registry.register(skill, phase="collect") is True
    # Same version in a new phase is allowed and tracked under the new phase.
    assert registry.register(skill, phase="verify") is True
    assert len(registry.items()) == 1
    assert registry.is_active("research", phase="verify")
    assert not registry.is_active("research", phase="collect")


def test_ensure_required_reports_missing(tmp_path):
    registry = ActiveSkills()
    with pytest.raises(MissingSkillError):
        registry.ensure_required([("research", "collect")])

    skill = loaded(tmp_path)
    registry.register(skill, phase="collect")
    registry.ensure_required([("research", "collect")])  # no raise
    with pytest.raises(MissingSkillError):
        registry.ensure_required([("evidence-review", "review")])


def test_guidance_is_stable_and_deduped(tmp_path):
    registry = ActiveSkills()
    skill = loaded(tmp_path)
    registry.register(skill, phase="collect")
    uses = registry.guidance(["research", "research", "missing"])
    assert [use.name for use in uses] == ["research"]


def test_active_skills_snapshot_round_trip(tmp_path):
    registry = ActiveSkills()
    skill = loaded(tmp_path)
    registry.register(skill, phase="collect")
    restored = ActiveSkills.from_snapshot(registry.snapshot())
    assert restored.is_active("research", phase="collect")
    assert restored.version_hash("research") == skill.content_hash


# --- context assembly ---

def _skill_use(body: str = "必需写作指导正文", name: str = "report-writing") -> "object":
    registry = ActiveSkills()
    skill = LoadedSkill(
        name=name,
        description="desc",
        body=body,
        content_hash="b" * 64,
        source_path=Path("unused"),
    )
    registry.register(skill, phase="write")
    return registry.guidance([name])[0]


def test_assemble_includes_required_skill_in_system():
    skill = _skill_use()
    result = assemble(role="你是报告撰写者", task="撰写第一章", skills=[skill], window_tokens=8000)
    assert any("必需写作指导正文" in str(message.get("content", "")) for message in result.messages)
    assert result.included_skills == (skill,)
    assert result.skill_versions() == {"report-writing": "b" * 64}


def test_assemble_drops_overflowing_evidence_but_keeps_guidance():
    skill = _skill_use()
    huge = ["证据" * 2000]
    result = assemble(
        role="角色",
        task="任务",
        skills=[skill],
        evidence=huge,
        window_tokens=120,
    )
    assert "evidence_overflow" in result.dropped
    assert len(result.messages) == 2  # system + user task only
    assert any("必需写作指导正文" in str(message.get("content", "")) for message in result.messages)


def test_assemble_raises_when_window_cannot_fit_mandatory():
    skill = _skill_use()
    with pytest.raises(ContextError, match="context_too_small_for_required_guidance"):
        assemble(role="很长的角色说明" * 500, task="任务", skills=[skill], window_tokens=10)


def test_context_builder_binds_registry_and_phase(tmp_path):
    registry = ActiveSkills()
    loader = SkillLoader(tmp_path)
    skill_md = SKILL_MD.replace("research", "research").replace("网络研究执行指导", "研究规划执行指导")
    (tmp_path / "research").mkdir(parents=True, exist_ok=True)
    (tmp_path / "research" / "SKILL.md").write_text(skill_md, encoding="utf-8")
    registry.register(loader.load("research"), phase="plan")
    builder = ContextBuilder("你是研究负责人", registry, window_tokens=8000)
    assembly = builder.build(task="制定大纲", phase="plan", required_skills=["research"])
    assert isinstance(assembly, ContextAssembly)
    assert "官方报告优先" in str(assembly.messages[0]["content"])


def test_context_builder_blocks_when_required_skill_missing():
    registry = ActiveSkills()
    builder = ContextBuilder("你是研究负责人", registry, window_tokens=8000)
    with pytest.raises(MissingSkillError):
        builder.build(task="制定大纲", phase="plan", required_skills=["research"])
