"""The fixed four-chapter report contract."""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from ai_status_report.schemas.common import SCHEMA_VERSION, ProjectModel

REPORT_SECTION_ORDER = ("background", "current-status", "trends", "recommendations")


class ReportTitle(ProjectModel):
    """Validated, reader-facing title for one complete report."""

    title: str = Field(min_length=4, max_length=80)

    @field_validator("title")
    @classmethod
    def _safe_title(cls, value: str) -> str:
        value = value.strip()
        if any(char in value for char in "\r\n#"):
            raise ValueError("report title contains unsupported formatting")
        return value


class ReportSectionSpec(ProjectModel):
    """One top-level chapter's stable role in the final report."""

    section_id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=80)
    purpose: str = Field(min_length=1, max_length=1000)
    core_questions: list[str] = Field(min_length=1, max_length=8)
    evidence_boundary: str = Field(min_length=1, max_length=1000)
    output_requirements: list[str] = Field(min_length=1, max_length=8)
    depends_on: list[str] = Field(default_factory=list, max_length=4)


class ReportSpec(ProjectModel):
    """Versioned, fixed top-level structure for a complete report."""

    schema_version: str = SCHEMA_VERSION
    spec_version: str = "1"
    sections: list[ReportSectionSpec] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def _fixed_order_and_ids(self) -> ReportSpec:
        ids = tuple(section.section_id for section in self.sections)
        if ids != REPORT_SECTION_ORDER:
            raise ValueError("report spec must contain the fixed four sections in order")
        return self

    def section(self, section_id: str) -> ReportSectionSpec:
        for section in self.sections:
            if section.section_id == section_id:
                return section
        raise KeyError(section_id)


def default_report_spec() -> ReportSpec:
    """Return the project's fixed background-current-trends-recommendations spec."""

    return ReportSpec(
        sections=[
            ReportSectionSpec(
                section_id="background",
                title="背景",
                purpose="交代人工智能发展的宏观背景、问题范围、研究对象和分析口径。",
                core_questions=["为什么需要分析这一主题？", "研究对象、时间范围和地域范围是什么？"],
                evidence_boundary="优先使用官方统计、政策文件、权威机构报告和可核验的历史资料；区分事实、预测和分析判断。",
                output_requirements=["明确研究范围和关键概念", "说明时间、地域和统计口径", "避免提前展开趋势判断和实施建议"],
            ),
            ReportSectionSpec(
                section_id="current-status",
                title="现状",
                purpose="说明当前技术、市场、应用、产业基础设施和治理状态。",
                core_questions=["当前发展到什么程度？", "主要参与者、应用和约束是什么？"],
                evidence_boundary="优先采用最新可获得的事实资料；不同统计口径不得拼接为连续序列，个案不得写成行业普遍规律。",
                output_requirements=["按主题组织现状事实", "区分已发生结果与机构预测", "明确公开资料的时间和口径限制"],
            ),
            ReportSectionSpec(
                section_id="trends",
                title="趋势",
                purpose="基于背景和现状证据，归纳未来一段时期内值得关注的变化方向。",
                core_questions=["哪些变化具有持续性？", "趋势判断由哪些证据和条件支持？"],
                evidence_boundary="趋势必须标明依据和不确定性；资料不足时降低判断强度，不把预测或主题性表述写成确定事实。",
                output_requirements=["每项趋势连接到前文证据", "说明驱动因素和限制条件", "区分短期变化与长期判断"],
                depends_on=["background", "current-status"],
            ),
            ReportSectionSpec(
                section_id="recommendations",
                title="建议",
                purpose="面向指定读者提出与前文问题、趋势和约束相匹配的行动建议。",
                core_questions=["管理者或实践者应关注什么？", "建议的适用条件、优先级和风险是什么？"],
                evidence_boundary="建议应能追溯到前文事实、趋势或限制；无法由资料直接证明的内容应明确作为分析建议，而非事实结论。",
                output_requirements=["给出分层、可执行的建议", "说明前提条件和主要风险", "避免脱离证据泛化或承诺确定收益"],
                depends_on=["current-status", "trends"],
            ),
        ]
    )
