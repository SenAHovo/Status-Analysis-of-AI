"""Validated non-sensitive ledger, model, and task budget configuration."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field, field_validator, model_validator

from ai_status_report.schemas.common import ProjectModel
from ai_status_report.token_budget.allocator import BudgetLimits

_REQUIRED_CAPACITIES = {"run", "window", "output", "search_requests", "retrieval_requests"}
_REQUIRED_TASKS = {
    "document_write", "document_revision", "evidence_review", "search_planning",
    "search_synthesis", "search_execution", "intent_routing", "report_title", "token_probe",
}


class LedgerConfig(ProjectModel):
    capacities: dict[str, int]

    @field_validator("capacities")
    @classmethod
    def _capacities(cls, value: dict[str, int]) -> dict[str, int]:
        if set(value) != _REQUIRED_CAPACITIES or any(
            type(amount) is not int or amount < 0 for amount in value.values()
        ):
            raise ValueError("invalid ledger capacities")
        return value


class ModelBudgetProfile(ProjectModel):
    context_window_tokens: int = Field(ge=1024, le=1_000_000)
    max_output_tokens: int = Field(ge=1, le=384_000)

    @model_validator(mode="after")
    def _output_fits_window(self) -> ModelBudgetProfile:
        if self.max_output_tokens > self.context_window_tokens:
            raise ValueError("model output exceeds context window")
        return self


class TaskBudgetProfile(ProjectModel):
    max_output_tokens: int = Field(ge=1, le=384_000)
    evidence_tokens: int | None = Field(default=None, ge=100, le=1_000_000)
    reserved_input_tokens: int | None = Field(default=None, ge=0, le=1_000_000)

    @model_validator(mode="after")
    def _document_fields_pair(self) -> TaskBudgetProfile:
        if (self.evidence_tokens is None) != (self.reserved_input_tokens is None):
            raise ValueError("incomplete document task budget")
        return self


class BudgetConfiguration(ProjectModel):
    ledger: LedgerConfig
    models: dict[str, ModelBudgetProfile]
    tasks: dict[str, TaskBudgetProfile]

    @field_validator("models")
    @classmethod
    def _models(cls, value: dict[str, ModelBudgetProfile]) -> dict[str, ModelBudgetProfile]:
        if not value:
            raise ValueError("missing model budget profiles")
        return value

    @field_validator("tasks")
    @classmethod
    def _tasks(cls, value: dict[str, TaskBudgetProfile]) -> dict[str, TaskBudgetProfile]:
        if set(value) != _REQUIRED_TASKS:
            raise ValueError("invalid task budget profiles")
        if any(name.startswith("document_") != (profile.evidence_tokens is not None) for name, profile in value.items()):
            raise ValueError("invalid document task budget profile")
        return value


def _budget_path(root: Path) -> Path:
    project_path = root / "config" / "budgets.yaml"
    return project_path if project_path.is_file() else Path(__file__).resolve().parents[3] / "config" / "budgets.yaml"


def load_budget_configuration(root: Path) -> BudgetConfiguration:
    try:
        raw = yaml.safe_load(_budget_path(root).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError("budget config unavailable") from exc
    if not isinstance(raw, dict):
        raise TypeError("budget config invalid")
    try:
        return BudgetConfiguration.model_validate(raw)
    except ValueError as exc:
        raise ValueError("budget config invalid") from exc


def load_budget_limits(root: Path) -> BudgetLimits:
    return BudgetLimits(capacities=load_budget_configuration(root).ledger.capacities)


def model_budget(root: Path, model: str) -> ModelBudgetProfile:
    try:
        return load_budget_configuration(root).models[model]
    except KeyError as exc:
        raise ValueError("model budget profile unavailable") from exc


def task_budget(root: Path, model: str, task_name: str) -> TaskBudgetProfile:
    configuration = load_budget_configuration(root)
    try:
        model_profile = configuration.models[model]
        profile = configuration.tasks[task_name]
    except KeyError as exc:
        raise ValueError("task budget profile unavailable") from exc
    if profile.max_output_tokens > model_profile.max_output_tokens:
        raise ValueError("task output exceeds model budget")
    if profile.evidence_tokens is not None and (
        profile.evidence_tokens + profile.reserved_input_tokens + profile.max_output_tokens
        > model_profile.context_window_tokens
    ):
        raise ValueError("document task exceeds model context window")
    return profile
