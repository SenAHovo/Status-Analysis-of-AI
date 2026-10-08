from ai_status_report.harness.review_policy import enforce_hard_gap_policy
from ai_status_report.schemas.document import DocumentSectionResult
from ai_status_report.schemas.review import ReviewDecision


def _result(*, severity: str = "hard") -> DocumentSectionResult:
    return DocumentSectionResult(
        task_type="document.write_section",
        status="drafted",
        run_id="run-policy-1",
        section_id="current-status",
        draft_version="1",
        draft_artifact={
            "artifact_id": "artifact-1",
            "run_id": "run-policy-1",
            "version": "1",
            "kind": "chapter_markdown",
            "mime_type": "text/markdown",
            "access_ref": "data/runs/run-policy-1/chapters/current-status/v1.md",
        },
        evidence_bundle_ref="data/runs/run-policy-1/evidence_bundles/bundle.json",
        citation_map_ref="data/runs/run-policy-1/chapters/current-status/v1.citations.json",
        evidence_gaps=[
            {
                "gap_id": "gap-adoption",
                "kind": "structure_issue",
                "severity": severity,
                "description": "缺少实际采用率数据",
            }
        ],
    )


def test_hard_gap_forces_a_bounded_supplement_when_review_falls_back_to_revision():
    result = _result()
    decision, plan = enforce_hard_gap_policy(
        decision=ReviewDecision(
            decision="need_revision",
            run_id=result.run_id,
            section_id=result.section_id,
            draft_version=result.draft_version,
            reason="存在缺口，但没有可执行的补证查询",
            source="deterministic_fallback",
        ),
        result=result,
        section_title="当前发展现状",
        supplement_round=0,
    )

    assert decision.decision == "need_evidence"
    assert decision.supplement_queries == ["当前发展现状 缺少实际采用率数据 官方 数据 研究"]
    assert plan is not None
    assert plan.supplement_round == 1
    assert plan.source == "deterministic"


def test_hard_gap_at_the_evidence_limit_cannot_be_approved():
    result = _result()
    decision, plan = enforce_hard_gap_policy(
        decision=ReviewDecision(
            decision="approve",
            run_id=result.run_id,
            section_id=result.section_id,
            draft_version=result.draft_version,
            source="model",
        ),
        result=result,
        section_title="当前发展现状",
        supplement_round=2,
    )

    assert decision.decision == "need_evidence"
    assert plan is None
    assert "上限" in decision.reason


def test_soft_gap_keeps_the_reviewers_revision_decision():
    result = _result(severity="soft")
    original = ReviewDecision(
        decision="need_revision",
        run_id=result.run_id,
        section_id=result.section_id,
        draft_version=result.draft_version,
    )
    decision, plan = enforce_hard_gap_policy(
        decision=original,
        result=result,
        section_title="当前发展现状",
        supplement_round=0,
    )

    assert decision == original
    assert plan is None


def test_soft_evidence_limitation_allows_limited_delivery():
    decision = ReviewDecision.from_document_result(
        run_id="run-policy-1",
        section_id="current-status",
        draft_version="1",
        evidence_gaps=[
            {
                "gap_id": "gap-source-diversity",
                "severity": "soft",
                "description": "来源多样性有限。",
            }
        ],
    )

    assert decision.decision == "deliver_with_limits"
    assert decision.supplement_queries == []


def test_hard_evidence_limitation_stops_supplement_and_allows_limited_delivery():
    result = DocumentSectionResult.model_validate(
        _result().model_dump(mode="json")
        | {"evidence_gaps": [{"gap_id": "gap-data", "kind": "evidence_limitation", "severity": "hard", "description": "公开资料不足。"}]}
    )
    decision, plan = enforce_hard_gap_policy(
        decision=ReviewDecision(
            decision="approve",
            run_id=result.run_id,
            section_id=result.section_id,
            draft_version=result.draft_version,
            source="deterministic_fallback",
        ),
        result=result,
        section_title="当前发展现状",
        supplement_round=0,
    )

    assert decision.decision == "deliver_with_limits"
    assert plan is None


def test_model_review_evidence_request_keeps_an_executable_plan_without_document_hard_gap():
    result = _result(severity="soft")
    decision, plan = enforce_hard_gap_policy(
        decision=ReviewDecision(
            decision="need_evidence",
            run_id=result.run_id,
            section_id=result.section_id,
            draft_version=result.draft_version,
            issues=[
                {
                    "issue_id": "review-citation",
                    "section_id": result.section_id,
                    "draft_version": result.draft_version,
                    "type": "citation",
                    "severity": "high",
                    "requested_change": "补充可验证的引用来源",
                }
            ],
            supplement_queries=["智能体采用率 权威统计"],
            source="model",
        ),
        result=result,
        section_title="当前发展现状",
        supplement_round=0,
    )

    assert decision.decision == "need_evidence"
    assert plan is not None
    assert plan.queries[0].gap_id == "review-citation"
    assert plan.queries[0].query == "智能体采用率 权威统计"
