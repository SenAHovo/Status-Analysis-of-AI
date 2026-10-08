"""Parse the fixed Markdown protocol emitted by the document writing model."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

from ai_status_report.schemas.document import EvidenceGap
from ai_status_report.schemas.evidence import EvidenceBundle, Excerpt
from ai_status_report.schemas.report import ArtifactRef
from ai_status_report.storage.search_results import canonicalize_url, run_directory


class MarkdownProtocolError(ValueError):
    """A stable reason why a draft does not meet the project Markdown protocol."""


_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_H2 = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_CITATION = re.compile(r"\[(\d{1,3})\]")
_LEGACY_INTERNAL_CITATION = re.compile(r"\[[^\[\]|\n]+\|[^\[\]|\n]+\]")
_GAP = re.compile(r"^[-*]\s+(?:\[(hard|soft)\]\s*)?(.+?)\s*$", re.IGNORECASE)
_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_READER_LIMITATION = re.compile(
    r"(?:需要说明|需要指出|尚缺少|缺少独立交叉验证|证据主要覆盖|证据不足)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedSectionMarkdown:
    title: str
    summary: str
    body: str
    evidence_gaps: tuple[EvidenceGap, ...]
    citation_numbers: tuple[int, ...]


@dataclass(frozen=True)
class CitationBinding:
    """One reader-visible numeric citation bound to one durable source."""

    number: int
    source_ids: tuple[str, ...]
    title: str
    url: str
    raw_refs: tuple[str, ...]
    content_refs: tuple[str, ...]
    supporting_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class CitationPlan:
    """Stable source-level numeric references from the active EvidenceBundle."""

    bindings: tuple[CitationBinding, ...]

    def selected(self, numbers: tuple[int, ...]) -> tuple[CitationBinding, ...]:
        bindings = {binding.number: binding for binding in self.bindings}
        try:
            return tuple(bindings[number] for number in numbers)
        except KeyError as exc:
            raise MarkdownProtocolError("markdown_unknown_citation_number") from exc

    def for_excerpt(self, excerpt: Excerpt) -> CitationBinding:
        """Return the source-level reader citation assigned to one excerpt."""

        source_id = str(excerpt.source_id)
        url = str(excerpt.url)
        key = _citation_source_key(source_id, url)
        for binding in self.bindings:
            if key == _citation_source_key(binding.source_ids[0], binding.url):
                return binding
        raise MarkdownProtocolError("citation_plan_excerpt_missing")


def _section(markdown: str, name: str) -> str:
    headings = list(_H2.finditer(markdown))
    matching = [
        (index, heading)
        for index, heading in enumerate(headings)
        if heading.group(1).strip() == name
    ]
    if not matching:
        raise MarkdownProtocolError(f"markdown_missing_{name}")
    if len(matching) > 1:
        raise MarkdownProtocolError(f"markdown_duplicate_{name}")
    index, heading = matching[0]
    end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
    content = markdown[heading.end() : end].strip()
    if not content:
        raise MarkdownProtocolError(f"markdown_empty_{name}")
    return content


def _parse_evidence_gaps(text: str) -> tuple[EvidenceGap, ...]:
    """Convert reader-facing Markdown gaps into controller-ready contracts."""

    gaps = []
    for line in text.splitlines():
        match = _GAP.match(line)
        if match is None:
            continue
        severity, description = match.groups()
        description = description.strip()
        if description in {"无", "暂无"}:
            continue
        digest = hashlib.sha256(description.encode("utf-8")).hexdigest()[:16]
        gaps.append(
            EvidenceGap(
                gap_id=f"gap-{digest}",
                severity=(severity or "soft").lower(),
                description=description,
            )
        )
    return tuple(gaps)


def parse_section_markdown(markdown: str) -> ParsedSectionMarkdown:
    """Extract required sections and explicit evidence citations from Markdown."""

    text = markdown.strip()
    if "```" in text:
        raise MarkdownProtocolError("markdown_code_fence_forbidden")
    title_match = _H1.match(text)
    if title_match is None:
        raise MarkdownProtocolError("markdown_missing_title")
    if any(heading.group(1).strip() == "参考来源" for heading in _H2.finditer(text)):
        raise MarkdownProtocolError("markdown_references_reserved_for_program")
    summary = _section(text, "本章摘要")
    body = _section(text, "正文")
    gaps_text = _section(text, "证据缺口")
    if _LEGACY_INTERNAL_CITATION.search(body):
        raise MarkdownProtocolError("markdown_internal_citation_forbidden")
    citation_numbers = tuple(dict.fromkeys(int(number) for number in _CITATION.findall(body)))
    return ParsedSectionMarkdown(
        title=title_match.group(1).strip(),
        summary=summary,
        body=body,
        evidence_gaps=_parse_evidence_gaps(gaps_text),
        citation_numbers=citation_numbers,
    )


def extract_reader_section_summary(markdown: str) -> str:
    """Read the approved summary from a reader-facing chapter Markdown artifact.

    Reader artifacts intentionally omit the controller-only ``证据缺口`` section,
    so they cannot be parsed with :func:`parse_section_markdown`.  The summary is
    still a structured section in the persisted artifact and is the only chapter
    context admitted to a later ReportSpec section.
    """

    text = markdown.strip()
    if _H1.match(text) is None:
        raise MarkdownProtocolError("markdown_missing_title")
    return _section(text, "本章摘要")


def _citation_source_key(source_id: str, url: str) -> str:
    """Prefer a normalized URL so one report receives one reader number."""

    return canonicalize_url(url) or f"source_id:{source_id}"


def build_citation_plan(bundle: EvidenceBundle) -> CitationPlan:
    """Assign one stable reader number to each source represented in the bundle."""

    grouped: dict[str, list[Excerpt]] = {}
    for excerpt in bundle.excerpts:
        grouped.setdefault(_citation_source_key(excerpt.source_id, excerpt.url), []).append(excerpt)
    return CitationPlan(
        bindings=tuple(
            CitationBinding(
                number=index,
                source_ids=tuple(dict.fromkeys(str(item.source_id) for item in excerpts)),
                title=str(excerpts[0].title),
                url=canonicalize_url(str(excerpts[0].url)),
                raw_refs=tuple(
                    dict.fromkeys(str(item.raw_ref) for item in excerpts if item.raw_ref)
                ),
                content_refs=tuple(
                    dict.fromkeys(str(item.content_ref) for item in excerpts if item.content_ref)
                ),
                supporting_chunk_ids=tuple(
                    dict.fromkeys(str(item.chunk_id) for item in excerpts)
                ),
            )
            for index, excerpts in enumerate(grouped.values(), start=1)
        )
    )


def validate_markdown_evidence(
    parsed: ParsedSectionMarkdown, bundle: EvidenceBundle
) -> tuple[CitationBinding, ...]:
    """Resolve numeric citations against the active EvidenceBundle only."""

    return build_citation_plan(bundle).selected(parsed.citation_numbers)


def normalize_reader_citations(
    parsed: ParsedSectionMarkdown, citations: tuple[CitationBinding, ...]
) -> tuple[ParsedSectionMarkdown, tuple[CitationBinding, ...]]:
    """Renumber only the citations actually used by a reader-facing chapter.

    The model receives provisional numbers for every source in the active
    EvidenceBundle.  A chapter commonly uses only a subset, so its visible
    references must be compacted to ``[1..N]`` after validation.  Keeping the
    normalization here makes the Markdown and its sidecar citation map agree.
    """

    number_map = {binding.number: index for index, binding in enumerate(citations, start=1)}
    normalized_body = _CITATION.sub(
        lambda match: f"[{number_map.get(int(match.group(1)), int(match.group(1)))}]",
        parsed.body,
    )
    normalized = tuple(
        replace(binding, number=number_map[binding.number]) for binding in citations
    )
    return (
        replace(
            parsed,
            body=normalized_body,
            citation_numbers=tuple(binding.number for binding in normalized),
        ),
        normalized,
    )


def render_reader_markdown(
    parsed: ParsedSectionMarkdown, citations: tuple[CitationBinding, ...]
) -> str:
    """Build the reader artifact without internal EvidenceGap coordination data."""

    reader_body = "\n\n".join(
        paragraph
        for paragraph in parsed.body.split("\n\n")
        if not _READER_LIMITATION.search(paragraph)
    ).strip()
    lines = [
        f"# {parsed.title}",
        "",
        "## 本章摘要",
        "",
        parsed.summary,
        "",
        "## 正文",
        "",
        reader_body,
        "",
        "## 参考来源",
        "",
    ]
    for citation in citations:
        display_title = citation.title or "未命名来源"
        lines.append(f"[{citation.number}] {display_title}" + (f"。{citation.url}" if citation.url else "。"))
    if not citations:
        lines.append("本章未使用正文引用。")
    return "\n".join(lines) + "\n"


def persist_section_markdown(
    root: Path,
    *,
    run_id: str,
    section_id: str,
    version: str,
    markdown: str,
    input_versions: dict[str, str],
) -> ArtifactRef:
    """Persist one immutable chapter version and return its controlled reference."""

    if not _SAFE_PATH_COMPONENT.fullmatch(section_id) or section_id in {".", ".."}:
        raise MarkdownProtocolError("invalid_section_id")
    directory = run_directory(root, run_id) / "chapters" / section_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"v{version}.md"
    content = markdown.encode("utf-8")
    content_hash = hashlib.sha256(content).hexdigest()
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != content_hash:
            raise MarkdownProtocolError("chapter_version_conflict")
    else:
        path.write_bytes(content)
    identity = hashlib.sha256(f"{run_id}\n{section_id}\n{version}".encode()).hexdigest()[:32]
    return ArtifactRef(
        artifact_id=f"artifact-chapter-{identity}",
        run_id=run_id,
        version=version,
        kind="chapter_markdown",
        mime_type="text/markdown",
        size=path.stat().st_size,
        hash=content_hash,
        access_ref=str(path),
        input_versions=input_versions,
    )


def validate_persisted_chapter_artifact(
    root: Path,
    *,
    artifact: ArtifactRef,
    section_id: str,
) -> tuple[Path, str]:
    """Resolve one chapter Artifact and verify its immutable content identity.

    A2A results are untrusted at the controller boundary.  The Document Agent
    produces these artifacts locally, but the controller still verifies that
    the declared path, size, and SHA-256 describe the exact chapter version
    inside the current run before relying on it for a revision decision.
    """

    if (
        not _SAFE_PATH_COMPONENT.fullmatch(section_id)
        or section_id in {".", ".."}
        or artifact.kind != "chapter_markdown"
        or artifact.mime_type != "text/markdown"
        or not _SHA256_HEX.fullmatch(artifact.hash)
    ):
        raise MarkdownProtocolError("chapter_artifact_invalid")
    run_id = artifact.run_id
    expected = (
        run_directory(root, run_id)
        / "chapters"
        / section_id
        / f"v{artifact.version}.md"
    ).resolve()
    candidate = Path(artifact.access_ref)
    actual_path = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if actual_path != expected or not actual_path.is_file():
        raise MarkdownProtocolError("chapter_artifact_unavailable")
    try:
        content = actual_path.read_bytes()
    except OSError as exc:
        raise MarkdownProtocolError("chapter_artifact_unavailable") from exc
    content_hash = hashlib.sha256(content).hexdigest()
    if artifact.size != len(content) or content_hash != artifact.hash:
        raise MarkdownProtocolError("chapter_artifact_integrity_failed")
    return actual_path, content_hash


def persist_citation_map(
    root: Path,
    *,
    artifact: ArtifactRef,
    citations: tuple[CitationBinding, ...],
    evidence_gaps: tuple[EvidenceGap, ...],
) -> Path:
    """Persist the machine-only evidence mapping beside its chapter version."""

    chapter_path = Path(artifact.access_ref)
    path = chapter_path.with_suffix(".citations.json")
    payload = {
        "schema_version": "1",
        "markdown_artifact": artifact.model_dump(mode="json"),
        "citations": [binding.__dict__ for binding in citations],
        "evidence_gaps": [gap.model_dump(mode="json") for gap in evidence_gaps],
    }
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise MarkdownProtocolError("citation_map_version_conflict")
    if not path.exists():
        path.write_text(content, encoding="utf-8")
    return path
