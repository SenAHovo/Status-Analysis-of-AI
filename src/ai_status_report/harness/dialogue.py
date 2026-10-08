"""Small in-process dialogue controller for the teaching CLI.

The model may propose an intent, reply, questions and preference selections.
This module owns the session state, validation, defaults and the handoff into
the existing report graph; it never starts services or changes budgets.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ai_status_report.briefing.clock import SystemClock
from ai_status_report.briefing.defaults import defaults_for_root
from ai_status_report.briefing.intents import classify
from ai_status_report.briefing.normalize import normalize_brief
from ai_status_report.harness.dialogue_state import DialogueMessage, DialogueState
from ai_status_report.model.client import DeepSeekClient, ProviderError
from ai_status_report.model.structured import StructuredOutputError, parse_structured, schema_hint
from ai_status_report.schemas.dialogue import (
    DEFAULT_PREFERENCES,
    PREFERENCE_OPTION_LABELS,
    DialogueModelOutput,
    PreferenceField,
    PreferenceQuestion,
    PreferenceSelection,
)
from ai_status_report.schemas.intent import Intent
from ai_status_report.schemas.research import ResearchBrief

MAX_DIALOGUE_MESSAGES = 10
MAX_DIALOGUE_MESSAGE_CHARS = 4000
DIALOGUE_MAX_OUTPUT_TOKENS = 2048
STRUCTURED_RETRY_MAX_TOKENS = 512

# These phrases mean that the user accepts every program-owned default.  They
# are deliberately narrow: a phrase such as "其他默认" must not discard an
# explicit preference supplied in the same turn.
_DEFAULT_PREFERENCE_PATTERNS = (
    "默认即可",
    "偏好默认",
    "全部默认",
    "均使用默认",
    "均按默认",
    "按默认设置",
    "按默认生成",
    "使用默认偏好",
    "不做偏好要求",
    "无偏好要求",
)
_DEFAULT_PREFERENCE_NEGATIONS = ("不使用默认", "不要默认", "不按默认")


class DialogueModelError(RuntimeError):
    """Safe model failure for the interactive entry point."""


_CONVERSATION_FALLBACK = "我可以围绕人工智能现状进行资料检索、证据整理、四章报告写作和 PDF 导出。您也可以继续提问。"
_UNSUPPORTED_FALLBACK = "当前系统主要支持人工智能现状分析报告及相关能力咨询，请重新描述您的报告主题或问题。"


@dataclass(frozen=True)
class DialogueTurn:
    output: DialogueModelOutput
    status: str
    brief: ResearchBrief | None = None
    message: str = ""


def _question(field: PreferenceField, question: str) -> PreferenceQuestion:
    options = list(PREFERENCE_OPTION_LABELS[field])
    return PreferenceQuestion(
        field=field,
        question=question,
        options=options,
        default_value=DEFAULT_PREFERENCES[field],
    )


def default_questions() -> list[PreferenceQuestion]:
    """Build the one-time two-field clarification shown by the CLI."""

    return [
        _question(PreferenceField.WRITING_STYLE, "请告诉我您期望的报告写作风格，如严谨细致、通俗说明、公正客观等。"),
        _question(PreferenceField.USAGE_SCENARIO, "请告诉我报告主要使用场景，如教学演示、学生学习、学术研究或企业决策。"),
    ]


def _deterministic_output(text: str, *, first_turn: bool) -> DialogueModelOutput:
    decision = classify(text)
    conversational = text in {"你好", "您好", "嗨", "继续", "谢谢"} or any(
        phrase in text for phrase in ("你有什么能力", "你能做什么", "介绍下你自己", "你是谁")
    )
    if decision.kind is Intent.CONVERSATION and conversational:
        return DialogueModelOutput(
            intent=Intent.CONVERSATION,
            reply="我可以围绕人工智能现状进行资料检索、证据整理、四章报告写作和 PDF 导出。您也可以继续提问。",
            confidence=1.0,
            reason_code="deterministic_conversation",
        )
    if decision.kind is Intent.UNSUPPORTED:
        return DialogueModelOutput(
            intent=Intent.UNSUPPORTED,
            reply="当前系统主要支持人工智能现状分析报告及相关能力咨询，请重新描述您的报告主题或问题。",
            confidence=1.0,
            reason_code="deterministic_unsupported",
        )
    if first_turn:
        return DialogueModelOutput(
            intent=Intent.REPORT_REQUEST,
            report_subject=text[:200],
            needs_clarification=True,
            questions=default_questions(),
            confidence=1.0,
            reason_code="deterministic_report_preferences",
        )
    return DialogueModelOutput(
        intent=Intent.REPORT_REQUEST,
        report_subject=text[:200],
        preferences=[
            PreferenceSelection(field=field, value=value)
            for field, value in DEFAULT_PREFERENCES.items()
        ],
        confidence=1.0,
        reason_code="deterministic_default_preferences",
    )


def _bounded_messages(messages: list[DialogueMessage]) -> list[DialogueMessage]:
    bounded = [
        {"role": item["role"], "content": item["content"][:MAX_DIALOGUE_MESSAGE_CHARS]}
        for item in messages
    ]
    return bounded[-MAX_DIALOGUE_MESSAGES:]


def _requests_default_preferences(text: str) -> bool:
    """Return whether a turn explicitly accepts all four default preferences."""

    normalized = re.sub(r"[\s，,。；;：:]", "", text)
    if any(phrase in normalized for phrase in _DEFAULT_PREFERENCE_NEGATIONS):
        return False
    return normalized == "默认" or any(
        phrase in normalized for phrase in _DEFAULT_PREFERENCE_PATTERNS
    )


def _program_default_output(subject: str) -> DialogueModelOutput:
    """Create the program-owned report handoff for explicit default preferences."""

    return DialogueModelOutput(
        intent=Intent.REPORT_REQUEST,
        report_subject=subject,
        confidence=1.0,
        reason_code="program_default_preferences",
    )


class DialogueSession:
    """One current-process conversation with a model or deterministic adapter."""

    def __init__(
        self,
        root: Path,
        *,
        model: Callable[[list[DialogueMessage]], DialogueModelOutput] | None = None,
    ) -> None:
        self.root = root
        self.model = model
        self.state: DialogueState = {"messages": [], "session_status": "awaiting_input"}
        self._subject = ""
        self._pending_questions: list[PreferenceQuestion] = []
        self._selections: dict[PreferenceField, PreferenceSelection] = {}

    def _ready_for_report(self, cleaned: str, output: DialogueModelOutput) -> DialogueTurn:
        """Normalize the collected topic and preferences into a report brief."""

        brief_text = self._subject or cleaned
        if classify(brief_text).kind is not Intent.REPORT_REQUEST:
            brief_text = "分析" + brief_text
        try:
            defaults = defaults_for_root(self.root)
            plan = normalize_brief(
                brief_text,
                SystemClock(defaults.timezone),
                defaults,
                preferences=list(self._selections.values()),
            )
        except (TypeError, ValueError) as exc:
            raise DialogueModelError("dialogue_brief_invalid") from exc
        self.state["brief"] = plan.brief.model_dump(mode="json")
        self.state["session_status"] = "ready_for_report"
        return DialogueTurn(output, "ready_for_report", brief=plan.brief, message=output.reply)

    def _model_output(self, text: str) -> DialogueModelOutput:
        messages = _bounded_messages(self.state["messages"])
        if self.model is None:
            return _deterministic_output(text, first_turn=not bool(self.state.get("intent")))
        try:
            output = self.model(messages)
            if not isinstance(output, DialogueModelOutput):
                raise DialogueModelError("dialogue_model_failed")
            return output
        except DialogueModelError:
            raise
        except StructuredOutputError as exc:
            raise DialogueModelError(str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise DialogueModelError("dialogue_model_failed") from exc

    def handle(self, text: str) -> DialogueTurn:
        cleaned = " ".join(text.split())
        if not cleaned:
            raise ValueError("empty_dialogue_input")
        self.state["messages"] = _bounded_messages([
            *self.state["messages"],
            {"role": "user", "content": cleaned},
        ])
        if (
            self.state.get("session_status") == "awaiting_preferences"
            and _requests_default_preferences(cleaned)
        ):
            return self._ready_for_report(cleaned, _program_default_output(self._subject))
        output = self._model_output(cleaned)
        self.state["intent"] = output.intent.value
        self.state["confidence"] = output.confidence
        self.state["intent_reason"] = output.reason_code
        reply = output.reply
        if output.intent is Intent.CONVERSATION:
            reply = reply or _CONVERSATION_FALLBACK
            self.state["messages"] = _bounded_messages([
                *self.state["messages"], {"role": "assistant", "content": reply}
            ])
            self.state["session_status"] = "awaiting_input"
            return DialogueTurn(output, "conversation", message=reply)
        if output.intent is Intent.UNSUPPORTED:
            reply = reply or _UNSUPPORTED_FALLBACK
            self.state["messages"] = _bounded_messages([
                *self.state["messages"], {"role": "assistant", "content": reply}
            ])
            self.state["session_status"] = "awaiting_input"
            return DialogueTurn(output, "unsupported", message=reply)

        if reply:
            self.state["messages"] = _bounded_messages([
                *self.state["messages"], {"role": "assistant", "content": reply}
            ])

        if output.report_subject and not self._subject:
            self._subject = output.report_subject
        # The model remains responsible for deciding that this is a report
        # request. Once it does, an explicit all-default instruction is a
        # program control command: it must retain DefaultsFile provenance and
        # cannot be overturned by a model-generated clarification request.
        if output.intent is Intent.REPORT_REQUEST and _requests_default_preferences(cleaned):
            return self._ready_for_report(cleaned, _program_default_output(self._subject))
        if output.needs_clarification:
            self._pending_questions = output.questions or default_questions()
            self.state["pending_questions"] = [
                item.model_dump(mode="json") for item in self._pending_questions
            ]
            question_summary = "；".join(item.question for item in self._pending_questions)
            self.state["messages"] = _bounded_messages([
                *self.state["messages"],
                {"role": "assistant", "content": f"需要补充以下偏好：{question_summary}"},
            ])
        for selection in output.preferences:
            self._selections[selection.field] = selection
        self.state["selections"] = [item.model_dump(mode="json") for item in self._selections.values()]
        if (
            output.needs_clarification
            and self._pending_questions
            and len(self._selections) < len(DEFAULT_PREFERENCES)
        ):
            self.state["session_status"] = "awaiting_preferences"
            return DialogueTurn(output, "needs_preferences", message=output.reply)

        return self._ready_for_report(cleaned, output)


class StructuredDialogueModel:
    """DeepSeek JSON adapter; no service lifecycle or workflow decisions."""

    def __init__(self, client: DeepSeekClient) -> None:
        self.client = client

    def __call__(self, messages: list[DialogueMessage]) -> DialogueModelOutput:
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是人工智能现状分析系统的命令行对话主控。只输出 JSON，不输出 Markdown。"
                    "固定意图只能是 report_request、conversation、unsupported。"
                    "报告请求只需要一次性收集写作风格、使用场景两项偏好；"
                    "信息不足时只返回 needs_clarification=true，questions 必须为 []，"
                    "程序会生成固定两项追问；已有偏好必须放入 preferences。"
                    "用户明确表示“默认”“默认即可”“偏好默认”“全部默认”或“按默认生成”时，"
                    "这是接受全部默认偏好的报告请求：返回 needs_clarification=false，"
                    "questions 和 preferences 都必须为 []，由程序记录默认值来源。"
                    "闲聊继续对话；范围外要求说明能力边界。"
                    "每次必须输出全部字段。conversation 或 unsupported 时：report_subject 为空字符串，"
                    "needs_clarification 为 false，questions 和 preferences 必须是空数组 []，不能是 null 或 {}。"
                    "report_request 没有已选偏好时 preferences 也必须是 []。"
                    f"输出契约：{schema_hint(DialogueModelOutput)}"
                ),
            },
            *messages,
        ]
        try:
            response = self.client.chat(
                prompt,
                response_format={"type": "json_object"},
                thinking={"type": "disabled"},
                max_tokens=DIALOGUE_MAX_OUTPUT_TOKENS,
                temperature=0.2,
            )
            content = response["choices"][0]["message"]["content"]
            return parse_structured(content, DialogueModelOutput)
        except StructuredOutputError:
            # JSON mode can occasionally produce prose or a malformed optional
            # field. One low-temperature retry preserves the same conversation
            # and asks only for a contract-compliant restatement.
            retry_prompt = [
                *prompt,
                {
                    "role": "user",
                    "content": (
                        "上一条输出未满足 JSON 契约。请基于以上完整对话重新输出一次，"
                        "只输出一个符合全部字段和类型要求的 JSON 对象。"
                    ),
                },
            ]
            try:
                response = self.client.chat(
                    retry_prompt,
                    response_format={"type": "json_object"},
                    thinking={"type": "disabled"},
                    max_tokens=STRUCTURED_RETRY_MAX_TOKENS,
                    temperature=0,
                )
                content = response["choices"][0]["message"]["content"]
                return parse_structured(content, DialogueModelOutput)
            except StructuredOutputError as exc:
                raise DialogueModelError(str(exc)) from exc
            except (AttributeError, KeyError, IndexError, TypeError, ProviderError) as exc:
                raise DialogueModelError(str(exc)) from exc
        except ProviderError as exc:
            raise DialogueModelError(str(exc)) from exc
        except (AttributeError, KeyError, IndexError, TypeError) as exc:
            raise DialogueModelError("dialogue_model_failed") from exc
