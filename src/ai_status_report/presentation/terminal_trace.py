"""Human-readable terminal presentation for persisted controller trace events."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

_SECTION_NAMES = {
    "background": "背景",
    "current-status": "现状",
    "trends": "趋势",
    "recommendations": "建议",
}
_RATE_LIMIT_PROGRESS = re.compile(r"^tavily_rate_limited_wait:([1-9]|[1-8][0-9]|90):2$")


class ChineseEventMapper:
    """Convert selected controller facts into safe, concise terminal messages.

    It deliberately ignores event summaries and A2A text: those fields can
    contain lengthy business payloads.  The terminal displays only program
    event types and short, controlled reference values.
    """

    def render(self, event: Mapping[str, Any]) -> str | None:
        event_type = str(event.get("event_type") or "")
        refs = event.get("refs")
        safe_refs = refs if isinstance(refs, Mapping) else {}
        section = self._section_name(safe_refs.get("section_id"))

        fixed = {
            "controller.request.classified": "主控已识别报告请求，开始准备固定四章任务。",
            "controller.search.awaiting_dispatch": "等待网络搜索 Agent 服务地址。",
            "controller.rag.index.started": "正在将检索证据写入向量索引。",
            "controller.rag.index.completed": "证据索引已完成，文档 Agent 可以进行按章检索。",
            "controller.rag.index.failed": "证据索引未完成，当前章节不会继续写作。",
            "controller.document.artifact_received": "主控已收到文档 Agent 的章节成果。",
            "controller.document.artifact_validated": "章节成果完整性校验通过。",
            "controller.document.revision_artifact_received": "主控已收到文档 Agent 的修订成果。",
            "controller.document.revision.skipped": "补证没有带来新增证据，跳过本轮修订。",
            "controller.supplement.completed": "定向补证已完成，正在判断是否进入修订。",
            "controller.report.assembled": "四章已组装为完整 Markdown 报告。",
            "controller.report.assembly_failed": "报告组装未完成。",
            "controller.report.pdf_exported": "PDF 已导出完成。",
            "controller.report.pdf_export_failed": "PDF 导出未完成。",
        }
        if event_type == "controller.search.artifact_received":
            if safe_refs.get("report_status") == "failed":
                if safe_refs.get("incomplete_reason") == "tavily_rate_limited":
                    return "网络搜索 Agent 被 Tavily 限流，当前章节未进入证据索引。"
                return "网络搜索 Agent 未能交付可用检索成果，当前章节未进入证据索引。"
            return "主控已接收网络搜索 Agent 的检索成果。"
        if event_type in fixed:
            return fixed[event_type]
        if event_type == "a2a.status_update" and event.get("agent_id") == "search_agent":
            match = _RATE_LIMIT_PROGRESS.fullmatch(str(safe_refs.get("progress_code") or ""))
            if match:
                return f"网络搜索 Agent 检测到 Tavily 限流，等待 {match.group(1)} 秒后重试（第 2/2 次）。"
        if event_type == "controller.a2a.dispatch":
            task_type = str(safe_refs.get("task_type") or "")
            if task_type == "research.search":
                return "主控已通过 A2A 向网络搜索 Agent 派发检索任务。"
            if task_type == "document.write_section":
                return f"主控已通过 A2A 向文档 Agent 派发{section}章节写作任务。"
            if task_type == "document.revise_section":
                return f"主控已通过 A2A 向文档 Agent 派发{section}章节修订任务。"
        if event_type == "controller.document.task_prepared":
            return f"主控已准备{section}章节的写作任务。"
        if event_type == "controller.document.revision_prepared":
            return f"主控已准备{section}章节的修订任务。"
        if event_type == "controller.review.started":
            return f"主控正在审核{section}章节。"
        if event_type == "controller.review.decided":
            return self._review_message(section, str(safe_refs.get("decision") or ""))
        if event_type == "a2a.artifact_update":
            agent = str(event.get("agent_id") or "")
            if agent == "search_agent":
                return "网络搜索 Agent 已通过 A2A 交付检索成果。"
            if agent == "document_agent":
                return "文档 Agent 已通过 A2A 交付章节成果。"
        return None

    @staticmethod
    def _section_name(value: object) -> str:
        section_id = str(value or "")
        return _SECTION_NAMES.get(section_id, "当前")

    @staticmethod
    def _review_message(section: str, decision: str) -> str:
        messages = {
            "approve": f"{section}章节审核通过，继续下一阶段。",
            "deliver_with_limits": f"{section}章节在公开资料边界内获准交付。",
            "need_evidence": f"{section}章节需要定向补证。",
            "need_revision": f"{section}章节需要按审核意见修订。",
        }
        return messages.get(decision, f"{section}章节审核已完成。")


class TerminalTraceFollower:
    """Follow persisted controller JSONL traces without modifying their writer.

    The follower is intentionally read-only.  Each emitted line has already
    passed trace redaction and is produced by the existing controller writer.
    """

    def __init__(
        self,
        trace_directory: Path,
        *,
        mapper: ChineseEventMapper | None = None,
        output: Callable[[str], None] = print,
        poll_interval_seconds: float = 0.2,
    ) -> None:
        self.trace_directory = trace_directory
        self.mapper = mapper or ChineseEventMapper()
        self.output = output
        self.poll_interval_seconds = poll_interval_seconds
        self._offsets: dict[Path, int] = {}
        self._buffers: dict[Path, str] = {}
        self._seen_event_ids: set[str] = set()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def poll_once(self) -> int:
        """Render all newly completed JSONL event lines and return their count."""

        rendered = 0
        for path in sorted(self.trace_directory.glob("*.jsonl")):
            if path.name.endswith(".a2a.jsonl"):
                continue
            rendered += self._read_new_events(path)
        return rendered

    def start(self) -> None:
        """Start a daemon follower suitable for the later interactive CLI."""

        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._follow, name="terminal-trace-follower", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the follower without touching the controller or trace files."""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.poll_interval_seconds * 2, 1.0))
            self._thread = None

    def _follow(self) -> None:
        while not self._stop_event.is_set():
            self.poll_once()
            self._stop_event.wait(self.poll_interval_seconds)

    def _read_new_events(self, path: Path) -> int:
        offset = self._offsets.get(path, 0)
        try:
            with path.open("r", encoding="utf-8") as handle:
                handle.seek(offset)
                content = handle.read()
                self._offsets[path] = handle.tell()
        except (OSError, UnicodeError):
            return 0
        text = self._buffers.get(path, "") + content
        lines = text.split("\n")
        self._buffers[path] = lines.pop()
        rendered = 0
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("event_id") or "")
            if not event_id or event_id in self._seen_event_ids:
                continue
            self._seen_event_ids.add(event_id)
            message = self.mapper.render(event)
            if message:
                self.output(message)
                rendered += 1
        return rendered
