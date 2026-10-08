"""Model-backed controller review tests without network requests."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import ai_status_report.harness.review as review_module
from ai_status_report.harness.review import ModelEvidenceReviewer
from ai_status_report.schemas.document import (
    DocumentBudget,
    DocumentSectionResult,
    DocumentWriteSectionTask,
)
from ai_status_report.schemas.evidence import EvidenceBundle, Excerpt


def _write_review_inputs(root: Path) -> tuple[DocumentWriteSectionTask, DocumentSectionResult]:
    run_id = "run-review-001"
    chapter = root / "data" / "runs" / run_id / "chapters" / "s1" / "v1.md"
    chapter.parent.mkdir(parents=True)
    chapter.write_text(
        "# 智能体交互方式\n\n## 本章摘要\n\n摘要。\n\n## 正文\n\n当前结论。[1]\n",
        encoding="utf-8",
    )
    citation_map = chapter.with_suffix(".citations.json")
    citation_map.write_text(
        json.dumps({"citations": [{"number": 1, "source_ids": ["source-1"]}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    bundle = EvidenceBundle(
        retrieval_id="retrieval-review-001",
        run_id=run_id,
        section_id="s1",
        queries=["智能体交互方式"],
        excerpts=[
            Excerpt(
                chunk_id="chunk-1",
                source_id="source-1",
                text="可追溯的证据正文。",
                title="来源一",
                url="https://example.com/source-1",
            )
        ],
    )
    bundle_path = root / "data" / "runs" / run_id / "evidence_bundles" / "bundle.json"
    bundle_path.parent.mkdir(parents=True)
    bundle_path.write_text(
        json.dumps(bundle.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8"
    )
    task = DocumentWriteSectionTask(
        run_id=run_id,
        research_report_ref=f"data/runs/{run_id}/reports/research.json",
        research_summary="研究摘要。",
        outline_ref=f"data/runs/{run_id}/outlines/v1.json",
        outline_version="1",
        section_id="s1",
        section_title="智能体交互方式",
        section_goal="说明主要交互模式。",
        retrieval_queries=["智能体交互方式"],
        evidence_requirements=["关键结论有来源。"],
        writing_constraints=["使用中性表达。"],
        budget=DocumentBudget(
            context_window_tokens=16000,
            evidence_tokens=4000,
            reserved_input_tokens=3000,
            max_output_tokens=3000,
        ),
    )
    result = DocumentSectionResult(
        task_type="document.write_section",
        status="drafted",
        run_id=run_id,
        section_id="s1",
        draft_version="1",
        draft_artifact={
            "artifact_id": "artifact-chapter-1",
            "run_id": run_id,
            "version": "1",
            "kind": "chapter_markdown",
            "mime_type": "text/markdown",
            "hash": hashlib.sha256(chapter.read_bytes()).hexdigest(),
            "access_ref": str(chapter),
        },
        evidence_bundle_ref=str(bundle_path),
        citation_map_ref=str(citation_map),
    )
    return task, result


def test_model_evidence_reviewer_loads_skill_binds_identity_and_caches(monkeypatch, tmp_path):
    skill = tmp_path / "skills" / "evidence-review" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: evidence-review\ndescription: review test\n---\n\n必须审核来源。\n",
        encoding="utf-8",
    )
    task, result = _write_review_inputs(tmp_path)
    calls: list[tuple[list[dict], dict]] = []

    class FakeDeepSeekClient:
        def __init__(self, generation, timeout):
            assert generation.model == "review-model"
            assert timeout == 20

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def chat(self, messages, **options):
            calls.append((messages, options))
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": "need_revision",
                                    "issues": [
                                        {
                                            "issue_id": "style-1",
                                            "section_id": "s1",
                                            "draft_version": "1",
                                            "type": "style",
                                            "severity": "low",
                                            "requested_change": "收紧结论措辞。",
                                        }
                                    ],
                                    "supplement_queries": [],
                                    "reason": "现有证据足够，需调整表述。",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr(
        review_module,
        "load_settings",
        lambda root: SimpleNamespace(
            generation=SimpleNamespace(model="review-model"), timeout=20
        ),
    )
    monkeypatch.setattr(review_module, "DeepSeekClient", FakeDeepSeekClient)

    reviewer = ModelEvidenceReviewer(tmp_path)
    first = reviewer.review(task=task, result=result, review_round=0)
    second = reviewer.review(task=task, result=result, review_round=0)

    assert first.decision == "need_revision"
    assert first.source == "model"
    assert first.issues[0].section_id == "s1"
    assert second.source == "model_cache"
    assert len(calls) == 1
    assert "必须审核来源。" in calls[0][0][0]["content"]
    assert "当前结论。[1]" in calls[0][0][1]["content"]
    assert calls[0][1] == {
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "temperature": 0.1,
        "max_tokens": 4096,
    }


def test_model_evidence_reviewer_falls_back_when_model_findings_mismatch_task(monkeypatch, tmp_path):
    skill = tmp_path / "skills" / "evidence-review" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: evidence-review\ndescription: review test\n---\n\n必须审核来源。\n",
        encoding="utf-8",
    )
    task, result = _write_review_inputs(tmp_path)
    result = DocumentSectionResult.model_validate(
        result.model_dump()
        | {
            "evidence_gaps": [
                {
                        "gap_id": "gap-1",
                        "kind": "content_issue",
                        "severity": "hard",
                    "description": "缺少量化数据。",
                    "suggested_queries": ["智能体采用率数据"],
                }
            ]
        }
    )

    class FakeDeepSeekClient:
        def __init__(self, generation, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def chat(self, messages, **options):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": "need_revision",
                                    "issues": [
                                        {
                                            "issue_id": "wrong-identity",
                                            "section_id": "other-section",
                                            "draft_version": "1",
                                            "type": "style",
                                            "requested_change": "错误绑定。",
                                        }
                                    ],
                                    "supplement_queries": [],
                                    "reason": "错误输入。",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr(
        review_module,
        "load_settings",
        lambda root: SimpleNamespace(
            generation=SimpleNamespace(model="review-model"), timeout=20
        ),
    )
    monkeypatch.setattr(review_module, "DeepSeekClient", FakeDeepSeekClient)

    decision = ModelEvidenceReviewer(tmp_path).review(task=task, result=result, review_round=0)

    assert decision.source == "deterministic_fallback"
    assert decision.decision == "need_evidence"
    assert decision.supplement_queries == ["智能体采用率数据"]
    assert decision.fallback_reason == "review_issue_identity_mismatch"
