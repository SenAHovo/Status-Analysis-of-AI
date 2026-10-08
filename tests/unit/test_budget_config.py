import shutil
from pathlib import Path

import pytest

from ai_status_report.token_budget.config import load_budget_configuration, task_budget


def test_project_budget_configuration_exposes_teaching_profiles(tmp_path):
    source = Path(__file__).resolve().parents[2] / "config"
    shutil.copytree(source, tmp_path / "config")

    config = load_budget_configuration(tmp_path)

    assert config.ledger.capacities["run"] == 3_000_000
    assert task_budget(tmp_path, "deepseek-v4-flash", "document_write").max_output_tokens == 64_000
    assert task_budget(tmp_path, "deepseek-v4-flash", "search_execution").max_output_tokens == 65_536


def test_task_budget_rejects_unknown_model(tmp_path):
    source = Path(__file__).resolve().parents[2] / "config"
    shutil.copytree(source, tmp_path / "config")

    with pytest.raises(ValueError, match="task budget profile unavailable"):
        task_budget(tmp_path, "unknown", "document_write")
