"""Context assembly for one model request.

The builder composes role instructions, required skill guidance, the current
task and bounded evidence into model messages. Required guidance is verified
against the active skill registry before assembly, counted into the estimate,
and never duplicated. Optional evidence and history are trimmed to fit the
window; the assembler never pulls whole corpora into the request.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ai_status_report.skill_runtime.active import ActiveSkills, SkillUse
from ai_status_report.token_budget.estimator import estimate_messages, estimate_tokens

WINDOW_DEFAULT = 16_000
HEADROOM_RATIO = 0.9


class ContextError(RuntimeError):
    """Safe message for capacity or guidance problems."""


@dataclass(frozen=True)
class ContextAssembly:
    messages: list[dict]
    estimate: int
    included_skills: tuple[SkillUse, ...] = ()
    dropped: tuple[str, ...] = ()

    def skill_versions(self) -> dict[str, str]:
        return {skill.name: skill.version_hash for skill in self.included_skills}


def assemble(
    *,
    role: str,
    task: str,
    skills: Sequence[SkillUse] = (),
    evidence: Sequence[str] = (),
    recent_messages: Sequence[dict] = (),
    window_tokens: int = WINDOW_DEFAULT,
) -> ContextAssembly:
    """Build a message list that fits ``window_tokens``.

    Priority is fixed: role and skill guidance cannot be trimmed; the task is
    mandatory; evidence and history are optional and are dropped (oldest first,
    evidence before history) when they would exceed the effective window.
    """
    effective = int(window_tokens * HEADROOM_RATIO)
    dropped: list[str] = []

    skill_text = "\n\n".join(skill.body for skill in skills)
    system_content = role if not skill_text else f"{role}\n\n适用业务指导：\n{skill_text}"
    system = {"role": "system", "content": system_content}
    mandatory_estimate = estimate_tokens(system_content) + estimate_tokens(task)
    if mandatory_estimate > effective:
        raise ContextError("context_too_small_for_required_guidance")

    remaining = effective - mandatory_estimate
    user_parts = [f"当前任务：\n{task}"]
    evidence_text = "\n\n".join(evidence)
    if evidence_text:
        estimate_evidence = estimate_tokens(evidence_text)
        if estimate_evidence <= remaining:
            user_parts.append(f"可参考的证据（按优先级选择，不可当作事实全部采信）：\n{evidence_text}")
            remaining -= estimate_evidence
        else:
            dropped.append("evidence_overflow")

    messages = [system, {"role": "user", "content": "\n\n".join(user_parts)}]
    history = list(recent_messages)
    estimate_history = estimate_messages(history) if history else 0
    if estimate_history > remaining:
        dropped.append("history_overflow")
        while history and estimate_messages(history) > remaining:
            history.pop(0)
        dropped.append(f"history_trimmed_to_{len(history)}")
    messages.extend(history)

    return ContextAssembly(
        messages=messages,
        estimate=estimate_messages(messages),
        included_skills=tuple(skills),
        dropped=tuple(dropped),
    )


class ContextBuilder:
    """Phased context builder bound to one agent's active skill registry."""

    def __init__(
        self,
        role_instruction: str,
        active: ActiveSkills,
        window_tokens: int = WINDOW_DEFAULT,
    ):
        self.role_instruction = role_instruction
        self.active = active
        self.window_tokens = window_tokens

    def build(
        self,
        *,
        task: str,
        phase: str,
        required_skills: Iterable[str],
        evidence: Sequence[str] = (),
        recent_messages: Sequence[dict] = (),
    ) -> ContextAssembly:
        """Ensure required skills are active for ``phase``, then assemble."""
        required = [(name, phase) for name in required_skills]
        self.active.ensure_required(required)
        skills = self.active.guidance(name for name, _ in required)
        return assemble(
            role=self.role_instruction,
            task=task,
            skills=skills,
            evidence=evidence,
            recent_messages=recent_messages,
            window_tokens=self.window_tokens,
        )
