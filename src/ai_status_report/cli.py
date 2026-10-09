"""Command-line entry points for configuration, validation and teaching runs."""

import argparse
import asyncio
import hashlib
import json
import os
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from ai_status_report.harness.dialogue import (
    DialogueModelError,
    DialogueSession,
    StructuredDialogueModel,
)
from ai_status_report.harness.local_services import LocalServiceError, LocalServiceManager
from ai_status_report.model.client import DeepSeekClient
from ai_status_report.presentation.terminal_trace import TerminalTraceFollower
from ai_status_report.rag.chroma import ChromaEvidenceIndex, RagIndexError
from ai_status_report.rag.index_versions import DEFAULT_INDEX_VERSION, is_legacy_read_only
from ai_status_report.rag.maintenance import RagStoreMaintenanceError, reset_rag_store
from ai_status_report.schemas.dialogue import PREFERENCE_OPTION_LABELS, PreferenceField
from ai_status_report.schemas.report import ArtifactRef
from ai_status_report.settings import ConfigError, bootstrap, load_settings
from ai_status_report.smoke import run_checks
from ai_status_report.storage.search_results import (
    canonicalize_run_id,
    register_run_directory,
    run_directory,
    run_id_for_directory,
)
from ai_status_report.token_budget.config import load_budget_limits
from ai_status_report.token_budget.runtime import PersistentTokenLedger


def _run_report_workflow(
    root: Path,
    *,
    topic: str,
    run_id: str,
    search_agent_url: str,
    document_agent_url: str,
    model_route: bool,
    research_brief: dict | None = None,
) -> dict:
    """Run the existing paid four-chapter graph without formatting CLI output."""

    settings = load_settings(root)
    from ai_status_report.briefing.intents import classify
    from ai_status_report.harness.graph import build_graph

    def run_with_classifier(classifier):
        graph = build_graph(
            classify=classifier,
            search_agent_url=search_agent_url,
            document_agent_url=document_agent_url,
            root=root,
            whole_report=True,
        )
        initial_state = {"run_id": run_id, "user_input": topic}
        if research_brief is not None:
            initial_state["research_brief"] = research_brief
        return asyncio.run(graph.ainvoke(initial_state))

    # The interactive dialogue has already produced a validated report brief.
    # Do not spend a second model call to classify the same confirmed request.
    if research_brief is not None:
        return run_with_classifier(lambda _text: "report_request")
    if not model_route:
        return run_with_classifier(classify)
    from ai_status_report.model.client import DeepSeekClient
    from ai_status_report.model.router import DeepSeekRouter, ResponseCache
    from ai_status_report.token_budget.config import task_budget
    from ai_status_report.token_budget.runtime import new_run_ledger

    with DeepSeekClient(
        settings.generation,
        settings.timeout,
        ledger=new_run_ledger(run_id, root=root),
    ) as client:
        router = DeepSeekRouter(
            client,
            ResponseCache(root / "data" / "cache" / "model.sqlite"),
            max_output_tokens=task_budget(
                root, settings.generation.model, "intent_routing"
            ).max_output_tokens,
        )
        return run_with_classifier(lambda text: router.route_intent(text, allow_model=True))


def _brief_preference_lines(brief) -> list[str]:
    """Render the two user-facing teaching preferences in Chinese."""

    origins = {item.field: item.origin.value for item in brief.defaults_applied}
    origin_labels = {"default": "默认", "user": "用户指定", "updated": "已更新"}

    def origin(field: str) -> str:
        return origin_labels.get(origins.get(field, ""), "已确定")

    writing = brief.custom_writing_style or PREFERENCE_OPTION_LABELS[
        PreferenceField.WRITING_STYLE
    ].get(brief.writing_style.value, brief.writing_style.value)
    audience = brief.custom_usage_scenario or PREFERENCE_OPTION_LABELS[
        PreferenceField.USAGE_SCENARIO
    ].get(brief.audience.value, brief.audience.value)
    return [
        f"[偏好] 写作风格：{writing}（{origin('writing_style')}）",
        f"[偏好] 使用场景：{audience}（{origin('audience')}）",
    ]


def _report_summary(root: Path, run_id: str, result: dict) -> dict[str, object]:
    return {
        "run_id": str(result.get("run_id") or run_id),
        "phase": result.get("phase"),
        "error": result.get("error", ""),
        "report_markdown_artifact": result.get("report_markdown_artifact", {}),
        "report_pdf_artifact": result.get("report_pdf_artifact", {}),
        "trace_directory": str(run_directory(root, run_id) / "traces"),
        "event_count": len(result.get("events", [])),
        "a2a_task_count": len(result.get("a2a_tasks", [])),
    }


def _new_teach_run_id() -> str:
    return "teach-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _teaching_run_id_is_occupied(root: Path, run_id: str) -> bool:
    """Reject explicit teaching identifiers before a dialogue call can bill usage."""

    canonical_run_id = canonicalize_run_id(run_id)
    runs_root = root / "data" / "runs"
    label = runs_root / ".run_labels" / f"{canonical_run_id}.json"
    return run_directory(root, canonical_run_id).exists() or label.is_file()


def _new_teaching_dialogue_ledger(
    root: Path, *, run_id: str, session_directory: Path
) -> PersistentTokenLedger:
    """Persist paid dialogue use until it can be attached to its report run.

    The temporary location keeps non-report conversations out of ``data/runs``.
    Its snapshot already carries the eventual stable ``run_id``, so promotion
    never rewrites or loses any settled provider usage.
    """

    return PersistentTokenLedger(
        run_id,
        session_directory / "dialogue_token_usage.json",
        limits=load_budget_limits(root),
    )


def _promote_teaching_dialogue_ledger(
    ledger: PersistentTokenLedger, *, run_directory_path: Path
) -> Path:
    """Move a settled dialogue ledger into the registered report directory."""

    destination = run_directory_path / "token_usage.json"
    if destination.exists():
        raise RuntimeError("teaching_dialogue_ledger_destination_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(ledger.path, destination)
    ledger.lock_path.unlink(missing_ok=True)
    return destination


def _discard_teaching_dialogue_ledger(ledger: PersistentTokenLedger) -> None:
    """Remove transient accounting when the user leaves before creating a report."""

    ledger.path.unlink(missing_ok=True)
    ledger.lock_path.unlink(missing_ok=True)
    try:
        ledger.path.parent.rmdir()
    except OSError:
        pass


def _preserve_failed_teaching_dialogue_ledger(
    root: Path, *, run_id: str, ledger: PersistentTokenLedger
) -> Path:
    """Attach paid dialogue usage to a failed run without inventing a report topic."""

    failed_run_root = register_run_directory(root, run_id, "对话失败")
    return _promote_teaching_dialogue_ledger(ledger, run_directory_path=failed_run_root)


def _artifact_path(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    value = payload.get("access_ref")
    return str(value) if isinstance(value, str) else ""


def _preflight_teaching_rag_store(root: Path) -> int:
    """Read the shared collection before a teaching run can incur provider cost."""

    settings = load_settings(root)
    if settings.chroma_client_mode != "http":
        raise LocalServiceError("teaching_requires_chroma_http")
    index = ChromaEvidenceIndex(
        root,
        dimensions=settings.dimensions,
        client_mode=settings.chroma_client_mode,
        host=settings.chroma_host,
        port=settings.chroma_port,
    )
    return index.healthcheck()


def run_teaching_session(
    root: Path,
    *,
    run_id: str | None,
    search_agent_url: str,
    document_agent_url: str,
    model_route: bool,
    input_reader,
    output,
    service_manager_cls=LocalServiceManager,
    trace_follower_cls=TerminalTraceFollower,
    workflow_runner=_run_report_workflow,
    rag_preflight=_preflight_teaching_rag_store,
    rag_store_reset=reset_rag_store,
) -> dict[str, object]:
    """Run one interactive teaching report using the existing runtime layers."""

    manager = service_manager_cls(root)
    follower = None
    dialogue_ledger: PersistentTokenLedger | None = None
    dialogue_ledger_promoted = False
    actual_run_id = run_id or _new_teach_run_id()
    with ExitStack() as stack:
        try:
            if run_id is not None and _teaching_run_id_is_occupied(root, actual_run_id):
                error_code = "teaching_run_id_already_exists"
                output(f"[对话失败] 错误码：{error_code}，请使用新的 --run-id。")
                return {"status": "failed", "error": error_code}
            dialogue_model = None
            if model_route:
                settings = load_settings(root)
                dialogue_ledger = _new_teaching_dialogue_ledger(
                    root,
                    run_id=actual_run_id,
                    session_directory=manager.log_directory.parent,
                )
                client = stack.enter_context(
                    DeepSeekClient(settings.generation, settings.timeout, ledger=dialogue_ledger)
                )
                dialogue_model = StructuredDialogueModel(client)
            session = DialogueSession(root, model=dialogue_model)
            text = input_reader("\n请输入内容，直接回车退出：").strip()
            if not text:
                output("未输入内容，本次未创建报告运行。")
                return {"status": "cancelled", "service_log_directory": str(manager.log_directory)}
            while True:
                try:
                    turn = session.handle(text)
                except DialogueModelError as exc:
                    error_code = str(exc) or "dialogue_model_failed"
                    if dialogue_ledger is not None:
                        usage_path = _preserve_failed_teaching_dialogue_ledger(
                            root, run_id=actual_run_id, ledger=dialogue_ledger
                        )
                        dialogue_ledger_promoted = True
                        output(f"Token 账本：{usage_path}")
                    output(f"[对话失败] 错误码：{error_code}，本次会话已停止。")
                    return {"status": "failed", "error": error_code}
                if turn.status in {"conversation", "unsupported"}:
                    if turn.message:
                        output(f"\n{turn.message}")
                    text = input_reader("\n请输入下一条内容，直接回车退出：").strip()
                    if not text:
                        output("会话结束，未创建报告运行。")
                        return {"status": "cancelled", "service_log_directory": str(manager.log_directory)}
                    continue
                if turn.status == "needs_preferences":
                    output("\n为了按您的要求生成报告，请一次性补充以下偏好：")
                    default_choices = []
                    field_labels = {
                        PreferenceField.WRITING_STYLE: "写作风格",
                        PreferenceField.USAGE_SCENARIO: "使用场景",
                    }
                    for question in session.state.get("pending_questions", []):
                        field = PreferenceField(question["field"])
                        default_value = PREFERENCE_OPTION_LABELS[field].get(
                            question["default_value"], question["default_value"]
                        )
                        default_choices.append(f"{field_labels[field]}：{default_value}")
                        output(f"- {question['question']}")
                    output(f"直接回复“默认”将使用：{'；'.join(default_choices)}。")
                    text = input_reader("\n请输入您的偏好；不做要求可直接回复‘默认’：").strip()
                    if not text:
                        output("会话结束，未创建报告运行。")
                        return {"status": "cancelled", "service_log_directory": str(manager.log_directory)}
                    continue
                brief = turn.brief
                if brief is None:
                    output("[对话失败] 未能形成有效报告任务，本次会话已停止。")
                    return {"status": "failed", "error": "brief_missing"}
                output("\n[意图确认] 已识别为报告请求，准备生成固定四章报告。")
                for line in _brief_preference_lines(brief):
                    output(line)
                break

            # A confirmed report request owns a stable run before any local
            # service can fail. This preserves already-settled dialogue usage
            # for failed report attempts as well as successful deliveries.
            brief = brief.model_copy(update={"run_id": actual_run_id})
            run_root = register_run_directory(root, actual_run_id, brief.topic)
            if dialogue_ledger is not None:
                _promote_teaching_dialogue_ledger(dialogue_ledger, run_directory_path=run_root)
                dialogue_ledger_promoted = True
            services = manager.ensure_services()
            chroma_service = next(service for service in services if service.spec.name == "chroma")
            try:
                indexed_count = rag_preflight(root)
            except RagIndexError as exc:
                if chroma_service.reused:
                    raise LocalServiceError("chroma_rag_preflight_failed_external") from exc
                output("[服务] chroma：索引预检失败，正在重建本机派生索引。")
                manager.stop()
                rag_store_reset(root)
                services = manager.ensure_services()
                try:
                    indexed_count = rag_preflight(root)
                except RagIndexError as retry_exc:
                    raise LocalServiceError("chroma_rag_preflight_failed") from retry_exc
            for service in services:
                service_state = "复用已有服务" if service.reused else "已启动"
                output(f"[服务] {service.spec.name}：{service_state}")
            output(f"[服务] chroma：索引预检通过（当前索引块：{indexed_count}）。")
            trace_directory = run_root / "traces"
            follower = trace_follower_cls(trace_directory, output=output)
            output(f"\n[运行] run_id：{actual_run_id}")
            follower.start()
            try:
                result = workflow_runner(
                    root,
                    topic=brief.topic,
                    run_id=actual_run_id,
                    search_agent_url=search_agent_url,
                    document_agent_url=document_agent_url,
                    model_route=model_route,
                    research_brief=brief.model_dump(mode="json"),
                )
            except Exception:
                output(f"\n[中止] 主控与 A2A trace：{trace_directory}")
                output(f"Token 账本：{run_root / 'token_usage.json'}")
                output(f"服务日志：{manager.log_directory}")
                raise
            summary = _report_summary(root, actual_run_id, result)
            workflow_completed = summary["phase"] == "report_pdf_exported"
            output(f"\n[{'完成' if workflow_completed else '未完成'}] 阶段：{summary['phase']}")
            markdown_path = _artifact_path(summary["report_markdown_artifact"])
            pdf_path = _artifact_path(summary["report_pdf_artifact"])
            if markdown_path:
                output(f"Markdown：{markdown_path}")
            if pdf_path:
                output(f"PDF：{pdf_path}")
            output(f"主控与 A2A trace：{summary['trace_directory']}")
            output(f"Token 账本：{run_root / 'token_usage.json'}")
            output(f"服务日志：{manager.log_directory}")
            return {
                "status": "completed" if workflow_completed else "incomplete",
                **summary,
                "service_log_directory": str(manager.log_directory),
            }
        finally:
            if follower is not None:
                follower.poll_once()
                follower.stop()
            if dialogue_ledger is not None and not dialogue_ledger_promoted:
                _discard_teaching_dialogue_ledger(dialogue_ledger)
            manager.stop()


def main():
    parser = argparse.ArgumentParser(description="AI status reporting utilities")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="explicit project root")
    sub = parser.add_subparsers(dest="command", required=True)
    importer = sub.add_parser("bootstrap", help="import local TXT without printing secrets")
    importer.add_argument("--source", type=Path)
    sub.add_parser("check-config", help="validate without network calls")
    sub.add_parser("check-rag-store", help="check the shared Chroma service without model calls")
    sub.add_parser("reset-rag-store", help="delete the stopped local Chroma evidence store")
    smoke = sub.add_parser("smoke", help="small paid real-service checks")
    smoke.add_argument("--run-id", help="persist paid probe usage to this run")
    smoke.add_argument("--service", choices=["all", "deepseek", "ocr", "embedding"], default="all")
    route = sub.add_parser("route", help="route one request; model call is opt-in and cached")
    route.add_argument("text")
    route.add_argument("--run-id", help="persist paid routing usage to this run")
    route.add_argument("--model-route", action="store_true", help="use one model routing call")
    index = sub.add_parser("index-evidence", help="embed one run's EvidenceChunk files into Chroma")
    index.add_argument("--run-id", required=True)
    index.add_argument(
        "--index-version",
        default=DEFAULT_INDEX_VERSION,
        help="v2 uses cosine; v1 is read-only legacy L2",
    )
    retrieve = sub.add_parser("retrieve-evidence", help="query Chroma EvidenceChunk index")
    retrieve.add_argument("question")
    retrieve.add_argument("--run-id", required=True)
    retrieve.add_argument(
        "--index-version",
        default=DEFAULT_INDEX_VERSION,
        help="v2 uses cosine; v1 is read-only legacy L2",
    )
    retrieve.add_argument("--n-results", type=int, default=5)
    retrieve.add_argument("--verification-status", choices=["retrieved", "verified"])
    retrieve.add_argument("--distance-threshold", type=float)
    retrieve.add_argument("--rerank", action="store_true")
    section = sub.add_parser("retrieve-section", help="retrieve and persist a section EvidenceBundle")
    section.add_argument("query")
    section.add_argument("--run-id", required=True)
    section.add_argument(
        "--index-version",
        default=DEFAULT_INDEX_VERSION,
        help="v2 uses cosine; v1 is read-only legacy L2",
    )
    section.add_argument("--section-id", required=True)
    section.add_argument("--budget-tokens", type=int, default=1200)
    section.add_argument("--n-results", type=int, default=8)
    section.add_argument("--verification-status", choices=["retrieved", "verified"])
    section.add_argument("--distance-threshold", type=float)
    section.add_argument("--rerank", action="store_true")
    evaluate = sub.add_parser("evaluate-retrieval", help="run a reviewed retrieval evaluation set")
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--eval-set", type=Path, required=True)
    evaluate.add_argument("--n-results", type=int, default=12)
    evaluate.add_argument("--distance-threshold", type=float)
    export_pdf = sub.add_parser("export-pdf", help="convert a report Markdown artifact to PDF")
    export_pdf.add_argument("--input-markdown", type=Path, required=True)
    export_pdf.add_argument("--output-pdf", type=Path)
    run_report = sub.add_parser(
        "run-report", help="run the paid four-chapter report workflow through local A2A agents"
    )
    run_report.add_argument("--topic", required=True, help="report topic or user request")
    run_report.add_argument("--run-id", required=True, help="独立的本次运行标识")
    run_report.add_argument(
        "--search-agent-url", default="http://127.0.0.1:8001/", help="Search Agent A2A URL"
    )
    run_report.add_argument(
        "--document-agent-url", default="http://127.0.0.1:8002/", help="Document Agent A2A URL"
    )
    run_report.add_argument(
        "--model-route",
        action="store_true",
        help="use one optional DeepSeek call for intent routing; default is deterministic routing",
    )
    teach = sub.add_parser(
        "teach", help="start or reuse local services and interactively run one paid teaching report"
    )
    teach.add_argument("--run-id", help="optional explicit run identifier; default is generated after topic input")
    teach.add_argument("--search-agent-url", default="http://127.0.0.1:8001/")
    teach.add_argument("--document-agent-url", default="http://127.0.0.1:8002/")
    teach.add_argument("--model-route", action=argparse.BooleanOptionalAction, default=True)
    token_probe = sub.add_parser("token-probe", help="make one bounded model call and persist TokenLedger usage")
    token_probe.add_argument("--run-id", required=True, help="独立的本次账本运行标识")
    usage = sub.add_parser("show-usage", help="show persisted TokenLedger usage for one run")
    usage.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "pyproject.toml").is_file():
        parser.error("run from project root or supply --root")
    try:
        if args.command == "bootstrap":
            added = bootstrap(root, args.source or root.parent / "deepseekAPI.txt")
            print("Imported variable names: " + ", ".join(added))
            return 0
        if args.command == "show-usage":
            from ai_status_report.token_budget.allocator import TokenLedger, TokenLedgerStateError

            path = run_directory(root, args.run_id) / "token_usage.json"
            if not path.is_file():
                print(json.dumps({"run_id": args.run_id, "status": "not_found"}, ensure_ascii=False))
                return 1
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                ledger = TokenLedger.from_snapshot(payload, recover_in_flight=False)
                if ledger.run_id != args.run_id:
                    raise TokenLedgerStateError("token_ledger_snapshot_invalid")
            except (OSError, UnicodeError, json.JSONDecodeError, TokenLedgerStateError):
                print(json.dumps({"run_id": args.run_id, "status": "invalid_usage_file"}, ensure_ascii=False))
                return 1
            print(json.dumps({
                "run_id": ledger.run_id,
                "status": "ok",
                "path": str(path),
                "estimated": ledger.estimated,
                "actual": ledger.actual,
                "unknown": ledger.unknown,
                "remaining": {
                    kind: ledger.remaining(kind) for kind in ledger.limits.capacities
                },
            }, ensure_ascii=False, indent=2))
            return 0
        if args.command == "token-probe":
            from ai_status_report.model.client import DeepSeekClient
            from ai_status_report.token_budget.config import task_budget
            from ai_status_report.token_budget.runtime import new_run_ledger

            settings = load_settings(root)
            probe_budget = task_budget(root, settings.generation.model, "token_probe")
            ledger = new_run_ledger(args.run_id, root=root)
            with DeepSeekClient(
                settings.generation,
                settings.timeout,
                ledger=ledger,
            ) as client:
                response = client.chat(
                    [{"role": "user", "content": "Reply with exactly: TOKEN_LEDGER_OK"}],
                    max_tokens=probe_budget.max_output_tokens,
                    temperature=0,
                )
            usage = response.get("usage") if isinstance(response, dict) else None
            print(json.dumps({
                "run_id": args.run_id,
                "status": "ok",
                "model_response_received": isinstance(response, dict),
                "provider_usage": usage if isinstance(usage, dict) else {},
                "usage_file": str(run_directory(root, args.run_id) / "token_usage.json"),
            }, ensure_ascii=False, indent=2))
            return 0
        if args.command == "export-pdf":
            from ai_status_report.documents.pdf_export import export_report_pdf
            source = args.input_markdown.resolve()
            if not source.is_file():
                parser.error("input Markdown file does not exist")
            try:
                relative = source.relative_to(root / "data" / "runs")
            except ValueError:
                parser.error("input Markdown must be data/runs/<run_id>/reports/report.md")
            if len(relative.parts) != 3 or relative.parts[1:] != ("reports", "report.md"):
                parser.error("input Markdown must be data/runs/<run_id>/reports/report.md")
            try:
                run_id = run_id_for_directory(root, relative.parts[0])
            except ValueError:
                parser.error("input Markdown must belong to a known run directory")
            content = source.read_bytes()
            artifact = ArtifactRef(
                artifact_id=f"artifact-report-{run_id}", run_id=run_id, version="1",
                kind="report_markdown", mime_type="text/markdown", size=len(content),
                hash=hashlib.sha256(content).hexdigest(), access_ref=str(source),
            )
            result = export_report_pdf(root, report_artifact=artifact, output_pdf=args.output_pdf)
            print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
            return 0
        if args.command == "run-report":
            register_run_directory(root, args.run_id, args.topic)
            result = _run_report_workflow(
                root,
                topic=args.topic,
                run_id=args.run_id,
                search_agent_url=args.search_agent_url,
                document_agent_url=args.document_agent_url,
                model_route=args.model_route,
            )
            summary = _report_summary(root, args.run_id, result)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0 if result.get("phase") == "report_pdf_exported" else 1
        if args.command == "teach":
            result = run_teaching_session(
                root,
                run_id=args.run_id,
                search_agent_url=args.search_agent_url,
                document_agent_url=args.document_agent_url,
                model_route=args.model_route,
                input_reader=input,
                output=print,
            )
            if result.get("status") == "cancelled":
                return 0
            return 0 if result.get("phase") == "report_pdf_exported" else 1
        settings = load_settings(root)
        if args.command == "check-config":
            print("Configuration valid; values withheld. No network requests.")
            return 0
        if args.command == "check-rag-store":
            from ai_status_report.rag.chroma import ChromaEvidenceIndex

            index = ChromaEvidenceIndex(
                root,
                dimensions=settings.dimensions,
                client_mode=settings.chroma_client_mode,
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "client_mode": settings.chroma_client_mode,
                        "collection": index.collection.name,
                        "index_version": index.index_version,
                        "count": index.healthcheck(),
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "reset-rag-store":
            from ai_status_report.rag.maintenance import reset_rag_store

            print(json.dumps(reset_rag_store(root), ensure_ascii=False, indent=2))
            return 0
        if args.command == "route":
            if not args.model_route:
                from ai_status_report.briefing.intents import classify

                print(classify(args.text).kind.value)
                print("source=deterministic; no network request")
                return 0
            if not args.run_id:
                parser.error("paid route requires --run-id for TokenLedger persistence")
            from ai_status_report.model.client import DeepSeekClient
            from ai_status_report.model.router import DeepSeekRouter, ResponseCache
            from ai_status_report.token_budget.config import task_budget
            from ai_status_report.token_budget.runtime import new_run_ledger

            with DeepSeekClient(
                settings.generation,
                settings.timeout,
                ledger=new_run_ledger(args.run_id, root=root),
            ) as client:
                result = DeepSeekRouter(
                    client,
                    ResponseCache(root / "data" / "cache" / "model.sqlite"),
                    max_output_tokens=task_budget(
                        root, settings.generation.model, "intent_routing"
                    ).max_output_tokens,
                ).route_intent(args.text, allow_model=True)
            print(result.model_dump_json())
            return 0
        if args.command in {"index-evidence", "retrieve-evidence", "retrieve-section", "evaluate-retrieval"}:
            if args.command == "index-evidence" and is_legacy_read_only(args.index_version):
                parser.error("legacy index versions are read-only")
            from ai_status_report.rag.bundle import build_evidence_bundle, persist_evidence_bundle
            from ai_status_report.rag.chroma import ChromaEvidenceIndex
            from ai_status_report.rag.evaluation import evaluate_retrieval, load_evaluation_set
            from ai_status_report.rag.indexing import index_run_evidence

            if args.command == "index-evidence":
                result = index_run_evidence(
                    root,
                    args.run_id,
                    settings,
                    index_version=args.index_version,
                )
                print(json.dumps(result.as_dict(), ensure_ascii=False))
                return 0

            from ai_status_report.rag.glm import GLMClient
            from ai_status_report.token_budget.runtime import new_run_ledger

            with GLMClient(
                settings.embedding,
                settings.timeout,
                ledger=new_run_ledger(args.run_id, root=root),
            ) as client:
                if args.command == "evaluate-retrieval":
                    evaluation_set = load_evaluation_set(args.eval_set)
                    index = ChromaEvidenceIndex(
                        root,
                        index_version=evaluation_set["index_version"],
                        dimensions=settings.dimensions,
                        client_mode=settings.chroma_client_mode,
                        host=settings.chroma_host,
                        port=settings.chroma_port,
                    )
                    result = evaluate_retrieval(
                        evaluation_set,
                        index,
                        client,
                        n_results=args.n_results,
                        distance_threshold=args.distance_threshold,
                    )
                    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                    destination = (
                        root
                        / "data"
                        / "evals"
                        / "results"
                        / evaluation_set["name"]
                        / f"result-{timestamp}.json"
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(
                        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                    )
                    print(json.dumps({"path": str(destination), "metrics": result["metrics"]}, ensure_ascii=False))
                    return 0
                index = ChromaEvidenceIndex(
                    root,
                    index_version=args.index_version,
                    dimensions=settings.dimensions,
                    client_mode=settings.chroma_client_mode,
                    host=settings.chroma_host,
                    port=settings.chroma_port,
                )
                question = args.query if args.command == "retrieve-section" else args.question
                matches = index.query(
                    question,
                    client,
                    n_results=args.n_results,
                    run_id=args.run_id,
                    verification_status=args.verification_status,
                    distance_threshold=args.distance_threshold,
                )
                if args.rerank:
                    from ai_status_report.rag.rerank import rerank_matches

                    matches = rerank_matches(question, matches, client)
                if args.command == "retrieve-section":
                    bundle = build_evidence_bundle(
                        run_id=args.run_id,
                        section_id=args.section_id,
                        queries=[args.query],
                        matches=matches,
                        budget_tokens=args.budget_tokens,
                        index_version=index.index_version,
                        root=root,
                    )
                    path = persist_evidence_bundle(root, bundle)
                    print(json.dumps({"bundle": bundle.model_dump(mode="json"), "path": str(path)}, ensure_ascii=False, indent=2))
                    return 0
                print(json.dumps([match.model_dump(mode="json") for match in matches], ensure_ascii=False, indent=2))
                return 0
        if not args.run_id:
            parser.error("paid smoke requires --run-id for TokenLedger persistence")
        from ai_status_report.token_budget.runtime import new_run_ledger

        report = run_checks(
            settings,
            args.service,
            ledger=new_run_ledger(args.run_id, root=root),
        )
        destination = run_directory(root, args.run_id) / "diagnostics"
        destination.mkdir(parents=True, exist_ok=True)
        name = datetime.now(UTC).strftime("smoke-%Y%m%dT%H%M%S%fZ.json")
        (destination / name).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("Sanitized report: " + str(destination / name))
        return int(any(c["status"] != "passed" for c in report["checks"].values()))
    except ConfigError as exc:
        print("Configuration error: " + str(exc))
        return 2
    except RagIndexError as exc:
        print("RAG storage unavailable: " + str(exc))
        return 2
    except RagStoreMaintenanceError as exc:
        print("RAG store maintenance unavailable: " + str(exc))
        return 2
    except LocalServiceError as exc:
        print("Local teaching service unavailable: " + str(exc))
        return 2
    except (OSError, UnicodeError):
        print("Local file operation failed; details withheld.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
