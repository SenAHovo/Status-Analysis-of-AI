"""Deterministic ordering and context handoff for whole-report generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_status_report.schemas.document import RecentSectionSummary
from ai_status_report.schemas.report_spec import ReportSectionSpec, ReportSpec, default_report_spec


@dataclass
class ReportScheduler:
    """Walk the fixed report spec and retain only approved predecessor summaries."""

    spec: ReportSpec = field(default_factory=default_report_spec)
    _next_index: int = 0
    _summaries: list[RecentSectionSummary] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return self._next_index >= len(self.spec.sections)

    @property
    def completed_sections(self) -> tuple[str, ...]:
        return tuple(item.section_id for item in self._summaries)

    def next_section(self) -> ReportSectionSpec:
        if self.completed:
            raise StopIteration("report schedule completed")
        section = self.spec.sections[self._next_index]
        missing = [dependency for dependency in section.depends_on if dependency not in self.completed_sections]
        if missing:
            raise ValueError(f"section dependencies not completed: {', '.join(missing)}")
        return section

    def context_for_next(self) -> list[RecentSectionSummary]:
        """Return bounded predecessor context for the next document task."""

        return list(self._summaries[-5:])

    def complete_section(self, *, section_id: str, approved_version: str, summary: str, artifact_ref: str) -> None:
        section = self.next_section()
        if section.section_id != section_id:
            raise ValueError(f"expected section {section.section_id}, got {section_id}")
        self._summaries.append(
            RecentSectionSummary(
                section_id=section_id,
                approved_version=approved_version,
                summary=summary,
                artifact_ref=artifact_ref,
            )
        )
        self._next_index += 1
