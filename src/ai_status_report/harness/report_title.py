"""Generate and validate the reader-facing title of a complete report."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ai_status_report.model.client import DeepSeekClient, ProviderError
from ai_status_report.model.structured import StructuredOutputError, parse_structured, schema_hint
from ai_status_report.schemas.report_spec import ReportTitle
from ai_status_report.settings import ConfigError, load_settings
from ai_status_report.token_budget.config import task_budget
from ai_status_report.token_budget.runtime import new_run_ledger


def _fallback_title(user_input: str) -> str:
    topic = re.sub(r"^(请|帮我|请帮我)?\s*(分析|研究|撰写|生成|编写)?\s*", "", user_input.strip())
    topic = topic.rstrip("。！？!? ") or "人工智能"
    return f"{topic}：现状、趋势与建议"


def generate_report_title(
    root: Path, *, run_id: str, user_input: str, section_summaries: list[dict[str, object]]
) -> tuple[str, str]:
    """Use one bounded model call, with a deterministic safe fallback."""

    fallback = _fallback_title(user_input)
    try:
        settings = load_settings(root)
        title_budget = task_budget(root, settings.generation.model, "report_title")
        context = json.dumps(
            {"user_request": user_input, "approved_section_summaries": section_summaries},
            ensure_ascii=False,
        )[:12000]
        with DeepSeekClient(
            settings.generation,
            settings.timeout,
            ledger=new_run_ledger(run_id, root=root),
        ) as client:
            response = client.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是报告编辑。根据用户主题和四章已批准摘要，提炼一个准确、简洁、"
                            "不夸大结论的中文报告标题。只输出 JSON，不输出 Markdown。"
                            f"{schema_hint(ReportTitle)}"
                        ),
                    },
                    {"role": "user", "content": context},
                ],
                temperature=0.3,
                max_tokens=title_budget.max_output_tokens,
            )
        content = response["choices"][0]["message"]["content"]
        title = parse_structured(content, ReportTitle).title
        if title == "人工智能现状分析报告":
            raise StructuredOutputError("generic_title")
        return title, "model"
    except (ConfigError, ProviderError, StructuredOutputError, KeyError, IndexError, TypeError):
        return fallback, "deterministic_fallback"


def generate_report_title_node(state: dict, *, project_root: Path) -> dict:
    """Create the title after all approved chapter summaries are available."""

    title, source = generate_report_title(
        project_root,
        run_id=str(state.get("run_id", "title")),
        user_input=str(state.get("user_input", "")),
        section_summaries=list(state.get("approved_section_summaries", [])),
    )
    return {
        "report_title": title,
        "report_title_source": source,
        "events": [*state.get("events", []), "report.title.generated"],
    }
