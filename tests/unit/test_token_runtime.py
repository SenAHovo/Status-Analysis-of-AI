"""Run-scoped ledger persistence and recovery boundaries."""

import json
import sys

import pytest

from ai_status_report.token_budget import RUN, SEARCH_REQUESTS
from ai_status_report.token_budget.allocator import TokenLedgerStateError
from ai_status_report.token_budget.runtime import new_run_ledger


def test_persistent_ledgers_merge_interleaved_mutations(tmp_path):
    first = new_run_ledger("shared-run", root=tmp_path)
    reservation = first.reserve(RUN, 100)

    second = new_run_ledger("shared-run", root=tmp_path)
    second.meter(SEARCH_REQUESTS)
    first.settle(reservation, 25)

    restored = new_run_ledger("shared-run", root=tmp_path)
    assert restored.actual == {RUN: 25, SEARCH_REQUESTS: 1}
    assert restored.remaining(RUN) == 2_999_975
    assert restored.remaining(SEARCH_REQUESTS) == 59


def test_persistent_ledger_refuses_a_structurally_invalid_snapshot(tmp_path):
    path = tmp_path / "data" / "runs" / "broken-run" / "token_usage.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"run_id": "broken-run", "capacities": {}}), encoding="utf-8")

    with pytest.raises(TokenLedgerStateError, match="token_ledger_snapshot_invalid"):
        new_run_ledger("broken-run", root=tmp_path).reserve(RUN, 1)


def test_show_usage_returns_a_safe_status_for_an_invalid_snapshot(monkeypatch, capsys, tmp_path):
    from ai_status_report import cli

    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    path = tmp_path / "data" / "runs" / "broken-run" / "token_usage.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"run_id": "broken-run", "capacities": []}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["ai-status", "--root", str(tmp_path), "show-usage", "--run-id", "broken-run"],
    )

    assert cli.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "run_id": "broken-run",
        "status": "invalid_usage_file",
    }
