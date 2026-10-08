"""LangGraph controller with explicit A2A dispatch and run-level traces."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from ai_status_report.a2a.client import send_text_task
from ai_status_report.documents.pdf_export import export_report_pdf
from ai_status_report.harness.delivery_nodes import (
    assemble_report_node,
    export_report_node,
)
from ai_status_report.harness.graph_helpers import (
    index_evidence_for_document as _index_evidence_for_document_impl,
)
from ai_status_report.harness.graph_helpers import (
    validate_document_result,
    verified_chapter_artifact_hash,
)
from ai_status_report.harness.graph_routes import (
    route_after_assembly,
    route_after_classification,
    route_after_document,
    route_after_index,
    route_after_review,
    route_after_revision,
    route_after_search,
)
from ai_status_report.harness.rag_nodes import index_evidence_node
from ai_status_report.harness.report_title import generate_report_title_node
from ai_status_report.harness.review import ModelEvidenceReviewer
from ai_status_report.harness.review_nodes import review_section_node
from ai_status_report.harness.revision_nodes import revise_section_node
from ai_status_report.harness.search_nodes import (
    classify_node,
    prepare_report_node,
)
from ai_status_report.harness.state import WorkflowState
from ai_status_report.harness.supplement_nodes import supplement_search_node
from ai_status_report.harness.writing_nodes import write_section_node
from ai_status_report.schemas.document import (
    DocumentReviseSectionTask,
    DocumentSectionResult,
    DocumentWriteSectionTask,
)
from ai_status_report.schemas.review import ReviewDecision
from ai_status_report.settings import load_settings

_validate_document_result = validate_document_result
_verified_chapter_artifact_hash = verified_chapter_artifact_hash


def _index_evidence_for_document(root: Path, run_id: str) -> dict[str, object]:
    """Keep the build_graph indexer injection shape stable."""

    return _index_evidence_for_document_impl(root, run_id, load_settings(root))

def build_graph(
    *,
    classify: Callable[[str], str],
    search_agent_url: str | None = None,
    document_agent_url: str | None = None,
    root: Path | None = None,
    evidence_indexer: Callable[[Path, str], dict[str, object]] = _index_evidence_for_document,
    evidence_reviewer: Callable[
        [DocumentWriteSectionTask | DocumentReviseSectionTask, DocumentSectionResult, int], ReviewDecision
    ]
    | None = None,
    whole_report: bool = False,
):
    """Build the controller with optional real A2A specialist dispatch nodes."""

    project_root = root or Path.cwd()
    model_reviewer: ModelEvidenceReviewer | None = None

    def default_reviewer(
        task: DocumentWriteSectionTask | DocumentReviseSectionTask,
        result: DocumentSectionResult,
        review_round: int,
    ) -> ReviewDecision:
        """Initialize the persistent review cache only when review is reached."""

        nonlocal model_reviewer
        if model_reviewer is None:
            model_reviewer = ModelEvidenceReviewer(project_root)
        return model_reviewer.review(task=task, result=result, review_round=review_round)

    reviewer = evidence_reviewer or default_reviewer

    graph = StateGraph(WorkflowState)
    classify_node_fn = partial(
        classify_node,
        project_root=project_root,
        classify=classify,
        whole_report=whole_report,
    )
    prepare_report_node_fn = partial(
        prepare_report_node,
        project_root=project_root,
        search_agent_url=search_agent_url,
        whole_report=whole_report,
        sender=lambda url, task: send_text_task(url, task),
    )
    write_section_node_fn = partial(
        write_section_node,
        project_root=project_root,
        document_agent_url=document_agent_url,
        whole_report=whole_report,
        sender=lambda url, task: send_text_task(url, task),
    )
    review_node = partial(
        review_section_node,
        project_root=project_root,
        reviewer=reviewer,
        whole_report=whole_report,
    )
    supplement_search_node_fn = partial(
        supplement_search_node,
        project_root=project_root,
        search_agent_url=search_agent_url,
        evidence_indexer=evidence_indexer,
        sender=lambda url, task: send_text_task(url, task),
    )
    revision_node = partial(
        revise_section_node,
        project_root=project_root,
        document_agent_url=document_agent_url,
        sender=lambda url, task: send_text_task(url, task),
    )
    assemble_node = partial(assemble_report_node, project_root=project_root)
    title_node = partial(generate_report_title_node, project_root=project_root)
    export_node = partial(
        export_report_node,
        project_root=project_root,
        exporter=lambda root, **kwargs: export_report_pdf(root, **kwargs),
    )
    graph.add_node("classify", classify_node_fn)
    graph.add_node("prepare_report", prepare_report_node_fn)
    index_node = partial(
        index_evidence_node,
        project_root=project_root,
        evidence_indexer=evidence_indexer,
    )
    graph.add_node("index_evidence", index_node)
    graph.add_node("write_section", write_section_node_fn)
    graph.add_node("review_section", review_node)
    graph.add_node("supplement_search", supplement_search_node_fn)
    graph.add_node("revise_section", revision_node)
    graph.add_node("assemble_report", assemble_node)
    graph.add_node("generate_title", title_node)
    graph.add_node("export_report", export_node)
    graph.add_edge(START, "classify")
    # Conditional edges route the next node from the current State snapshot.
    # Source: https://docs.langchain.com/oss/python/langgraph/graph-api#conditional-edges
    graph.add_conditional_edges(
        "classify", route_after_classification, {"continue": "prepare_report", "end": END}
    )
    graph.add_conditional_edges(
        "prepare_report", lambda state: route_after_search(
            state, document_agent_url=document_agent_url
        ), {"index_evidence": "index_evidence", "end": END}
    )
    graph.add_conditional_edges(
        "index_evidence", route_after_index, {"write_section": "write_section", "end": END}
    )
    graph.add_conditional_edges(
        "write_section", route_after_document, {"review": "review_section", "end": END}
    )
    graph.add_conditional_edges(
        "review_section",
        lambda state: route_after_review(state, whole_report=whole_report),
        {
            "supplement": "supplement_search",
            "revise": "revise_section",
            "next_section": "prepare_report",
            "generate_title": "generate_title",
            "end": END,
        },
    )
    graph.add_edge("generate_title", "assemble_report")
    graph.add_edge("supplement_search", "revise_section")
    graph.add_conditional_edges(
        "revise_section", route_after_revision, {"review": "review_section", "end": END}
    )
    graph.add_conditional_edges(
        "assemble_report", route_after_assembly, {"export": "export_report", "end": END}
    )
    return graph.compile()
