import pytest

from ai_status_report.harness.report_scheduler import ReportScheduler


def test_scheduler_walks_four_chapters_and_hands_off_approved_context():
    scheduler = ReportScheduler()
    for index, section_id in enumerate(("background", "current-status", "trends", "recommendations"), 1):
        section = scheduler.next_section()
        assert section.section_id == section_id
        assert [item.section_id for item in scheduler.context_for_next()] == list(scheduler.completed_sections)
        scheduler.complete_section(
            section_id=section_id,
            approved_version=str(index),
            summary=f"summary-{section_id}",
            artifact_ref=f"artifact-{section_id}",
        )
    assert scheduler.completed
    with pytest.raises(StopIteration):
        scheduler.next_section()


def test_scheduler_rejects_out_of_order_completion():
    scheduler = ReportScheduler()
    with pytest.raises(ValueError, match="expected section background"):
        scheduler.complete_section(
            section_id="current-status",
            approved_version="1",
            summary="summary",
            artifact_ref="artifact",
        )
