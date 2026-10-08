"""Entry intent classification (deterministic first stage).

Three entry kinds follow the fixed design: ``report_request``,
``conversation`` and ``unsupported``. The LangGraph controller augments this
rule layer with a model decision where needed, while the boundaries stay the
same: report requests enter the research flow, small talk is answered briefly
and out-of-scope actions are declined.
"""

from __future__ import annotations

import re

from ai_status_report.schemas.intent import Intent, IntentDecision

__all__ = ["Intent", "IntentDecision", "classify"]

_WORD = re.compile(r"\bai\b", re.IGNORECASE)

_OUT_OF_SCOPE = (
    "天气", "邮件", "订票", "机票", "酒店", "闹钟", "提醒", "翻译",
    "播放", "音乐", "订餐", "点餐", "记账", "报销", "转账", "汇款",
    "打车", "导航", "计算", "日程", "日历", "股票下单", "订酒店",
)

_CONVERSATION = (
    "你好", "您好", "hi", "hello", "hey", "再见", "谢谢", "你是谁",
    "你能做什么", "可以做什么", "介绍一下你自己", "介绍你自己", "继续", "接着",
)

_ACTION_VERBS = (
    "分析", "研究", "调研", "调查", "综述", "梳理", "评估", "盘点",
    "追踪", "概览", "总结", "整理", "撰写", "生成一份", "出一份",
)

_REPORT_NOUNS = ("现状", "进展", "趋势", "报告", "综述", "市场", "前景", "技术发展")

_DOMAIN_HINTS = (
    "人工智能", "大模型", "机器学习", "深度学习", "智能体", "生成式",
    "ai", "llm", "算法", "大语言模型",
)


def _decision(kind: Intent, reason: str) -> IntentDecision:
    """Rule-layer decisions are certain and deterministic by construction."""

    return IntentDecision(kind=kind, reason=reason)


def _has_domain(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in _DOMAIN_HINTS) or bool(_WORD.search(lowered))


def classify(text: str) -> IntentDecision:
    """Classify a single user utterance.

    A sentence that mixes report words with out-of-scope vocabulary stays on
    the report side when it contains a report action and a subject. Pure
    out-of-scope actions and small talk without a report subject return their
    own kinds.
    """
    cleaned = " ".join(text.split())
    lowered = cleaned.lower()

    has_action = any(verb in cleaned for verb in _ACTION_VERBS)
    has_report_subject = any(noun in cleaned for noun in _REPORT_NOUNS) or _has_domain(cleaned)
    if has_action and has_report_subject:
        return _decision(Intent.REPORT_REQUEST, "report_intent_detected")
    if any(marker in lowered for marker in _OUT_OF_SCOPE):
        return _decision(Intent.UNSUPPORTED, "out_of_scope_action")
    if any(marker in lowered for marker in _CONVERSATION):
        return _decision(Intent.CONVERSATION, "small_talk_or_followup")

    has_noun = any(noun in cleaned for noun in _REPORT_NOUNS)
    if has_noun and _has_domain(cleaned):
        return _decision(Intent.REPORT_REQUEST, "report_intent_detected")
    return _decision(Intent.CONVERSATION, "no_report_subject")
