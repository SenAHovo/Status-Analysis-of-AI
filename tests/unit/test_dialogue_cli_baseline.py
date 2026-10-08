"""ADR-0005 P0: existing CLI entries stay compatible, with no network use.

Each case either exercises argparse wiring or a deterministic failure path.
No service is started and no paid provider is called.
"""

from __future__ import annotations

import sys

import pytest

from ai_status_report import cli

SUBCOMMANDS = (
    "bootstrap",
    "check-config",
    "check-rag-store",
    "reset-rag-store",
    "smoke",
    "route",
    "index-evidence",
    "retrieve-evidence",
    "retrieve-section",
    "evaluate-retrieval",
    "export-pdf",
    "run-report",
    "teach",
    "token-probe",
    "show-usage",
)


def project_root(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='p0-cli'\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("command", SUBCOMMANDS)
def test_subcommand_is_registered(tmp_path, monkeypatch, command) -> None:
    root = project_root(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ai-status", "--root", str(root), command, "--help"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0


def test_show_usage_reports_a_missing_run(tmp_path, monkeypatch, capsys) -> None:
    root = project_root(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["ai-status", "--root", str(root), "show-usage", "--run-id", "run-absent"]
    )

    assert cli.main() == 1
    assert "not_found" in capsys.readouterr().out


def test_check_config_succeeds_offline_without_echoing_values(tmp_path, monkeypatch, capsys) -> None:
    root = project_root(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-p0-placeholder-only")
    monkeypatch.setenv("GLM_OCR_API_KEY", "glm-p0-placeholder-only")
    monkeypatch.setenv("GLM_EMBEDDING_API_KEY", "emb-p0-placeholder-only")
    monkeypatch.setattr(sys, "argv", ["ai-status", "--root", str(root), "check-config"])

    assert cli.main() == 0

    out = capsys.readouterr().out
    assert "withheld" in out
    for placeholder in ("sk-p0-placeholder-only", "glm-p0-placeholder-only", "emb-p0-placeholder-only"):
        assert placeholder not in out


def test_token_probe_requires_a_run_id(tmp_path, monkeypatch) -> None:
    root = project_root(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ai-status", "--root", str(root), "token-probe"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2


def test_export_pdf_rejects_input_outside_the_run_tree(tmp_path, monkeypatch) -> None:
    root = project_root(tmp_path)
    source = tmp_path / "report.md"
    source.write_text("# Report\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["ai-status", "--root", str(root), "export-pdf", "--input-markdown", str(source)],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
