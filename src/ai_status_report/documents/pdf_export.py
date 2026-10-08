"""Deterministic Markdown-to-PDF export using the project-managed Chromium."""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import markdown2

from ai_status_report.schemas.report import ArtifactRef
from ai_status_report.storage.search_results import run_directory


def _css() -> str:
    return """
@page { size: A4; margin: 22mm 20mm 20mm 20mm; @bottom-center { content: counter(page); font-size: 9pt; color: #666; } }
body { font-family: 'Times New Roman', 'SimSun', '宋体', 'Noto Serif CJK SC', serif; font-size: 11pt; line-height: 1.65; color: #222; }
body > h1 { font-size: 22pt; margin: 0 0 18pt; page-break-before: always; }
body > h1:first-child {
  font-size: 20pt; line-height: 1.4; text-align: center; max-width: 92%;
  margin: 34pt auto 14pt; page-break-before: avoid; break-before: avoid;
  page-break-inside: avoid; break-inside: avoid;
  page-break-after: avoid; break-after: avoid;
}
h2 { font-size: 16pt; margin-top: 18pt; page-break-after: avoid; }
h3 { font-size: 13pt; page-break-after: avoid; }
p, li { orphans: 2; widows: 2; }
table { border-collapse: collapse; width: 100%; margin: 10pt 0; page-break-inside: avoid; }
th, td { border: 0.5pt solid #999; padding: 5pt 7pt; vertical-align: top; }
th { background: #f2f2f2; }
code, pre { font-family: 'Times New Roman', 'Consolas', monospace; }
pre { white-space: pre-wrap; }
a { color: #222; text-decoration: none; }
.toc { margin: 0 auto 18pt; max-width: 92%; font-size: 9.5pt; line-height: 1.25; page-break-inside: avoid; break-inside: avoid; }
.toc h2 { font-size: 15pt; text-align: center; margin: 0 0 6pt; }
.toc ol { margin: 0; padding-left: 24pt; list-style: none; }
.toc li { margin: 2pt 0; }
.references p { margin: 3pt 0; line-height: 1.45; overflow-wrap: anywhere; }
"""


def _pdf_markdown(markdown: str) -> str:
    """Keep each assembled reference entry as a separate Markdown block."""

    marker = "## 统一参考来源"
    if marker not in markdown:
        return markdown
    body, references = markdown.split(marker, 1)
    entries = [line for line in references.splitlines() if re.match(r"^\[\d+\]\s+", line)]
    if not entries:
        return markdown
    return f"{body.rstrip()}\n\n{marker}\n\n" + "\n\n".join(entries) + "\n"


def _promote_bold_subheadings(body: str) -> str:
    """Turn numbered, standalone bold paragraphs into searchable subheadings."""

    return re.sub(
        r"<p><strong>([一二三四五六七八九十]+、[^<]+)</strong></p>",
        r"<h3>\1</h3>",
        body,
    )


def _toc_and_anchors(body: str) -> str:
    """Add stable anchors and a three-level table of contents."""

    heading = re.compile(r"<h([1-3])>(.*?)</h\1>", re.DOTALL)
    nodes: list[tuple[int, str, str]] = []
    title_seen = False
    chapter_count = 0
    section_count = 0
    subsection_count = 0
    chinese_numbers = "一二三四五六七八九十"

    def add_anchor(match: re.Match[str]) -> str:
        nonlocal chapter_count, section_count, subsection_count, title_seen
        level, inner = match.group(1), match.group(2)
        label = html.unescape(re.sub(r"<[^>]+>", "", inner)).strip()
        numeric_level = int(level)
        if numeric_level == 1 and not title_seen:
            title_seen = True
            return f"<h1 id='report-title'>{inner}</h1>"
        if label == "统一参考来源":
            return match.group(0)
        if numeric_level == 1:
            chapter_count += 1
            section_count = 0
            subsection_count = 0
            prefix = f"{chinese_numbers[chapter_count - 1]}、" if chapter_count <= len(chinese_numbers) else f"{chapter_count}、"
            if not re.match(r"^[一二三四五六七八九十]+、", label):
                inner = f"{prefix}{inner}"
                label = f"{prefix}{label}"
        elif numeric_level == 2:
            section_count += 1
            subsection_count = 0
            if not re.match(r"^\d+、", label):
                inner = f"{section_count}、{inner}"
                label = f"{section_count}、{label}"
        elif numeric_level == 3:
            subsection_count += 1
            old_prefix = r"^(?:[一二三四五六七八九十]+、|\d+[、.]|（\d+）)\s*"
            inner = re.sub(old_prefix, "", inner)
            label = re.sub(old_prefix, "", label).strip()
            inner = f"（{subsection_count}）{inner}"
            label = f"（{subsection_count}）{label}"
        anchor = f"toc-{len(nodes) + 1}"
        toc_label = re.sub(r"^(?:[一二三四五六七八九十]+、|\d+、|（\d+）)", "", label).strip()
        nodes.append((numeric_level, anchor, toc_label))
        return f"<h{level} id='{anchor}'>{inner}</h{level}>"

    anchored = heading.sub(add_anchor, body)
    if not nodes:
        return anchored

    tree: list[dict[str, object]] = []
    stack: list[dict[str, object]] = []
    for level, anchor, label in nodes:
        node: dict[str, object] = {"level": level, "anchor": anchor, "label": label, "children": []}
        while stack and int(stack[-1]["level"]) >= level:
            stack.pop()
        parent = stack[-1]["children"] if stack else tree
        assert isinstance(parent, list)
        parent.append(node)
        stack.append(node)

    def render(items: list[dict[str, object]], level: int) -> str:
        chinese_numbers = "一二三四五六七八九十"
        rendered: list[str] = []
        for index, item in enumerate(items, 1):
            marker = (
                f"{chinese_numbers[index - 1]}、"
                if level == 1 and index <= len(chinese_numbers)
                else f"{index}、"
                if level == 2
                else f"（{index}）"
            )
            anchor = html.escape(str(item["anchor"]), quote=True)
            label = html.escape(str(item["label"]))
            children = item["children"]
            nested = render(children, level + 1) if isinstance(children, list) and children else ""
            rendered.append(f"<li><a href='#{anchor}'>{marker}{label}</a>{nested}</li>")
        return f"<ol class='toc-level-{level}'>{''.join(rendered)}</ol>"

    toc = f"<nav class='toc' aria-label='目录'><h2>目录</h2>{render(tree, 1)}</nav>"
    return anchored.replace("</h1>", "</h1>" + toc, 1)


def _html(markdown: str, base_url: Path) -> str:
    body = markdown2.markdown(_pdf_markdown(markdown), extras=["tables", "fenced-code-blocks", "strike"])
    body = _promote_bold_subheadings(body)
    body = re.sub(
        r"(<h2>统一参考来源</h2>)(.*)",
        r"\1<div class='references'>\2</div>",
        body,
        count=1,
        flags=re.DOTALL,
    )
    body = _toc_and_anchors(body)
    base_href = html.escape(base_url.resolve().as_uri() + "/", quote=True)
    return (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<base href='{base_href}'><style>{_css()}</style></head>"
        f"<body>{body}</body></html>"
    )


def _report_paths(root: Path, run_id: str) -> tuple[Path, Path]:
    """Return the only Markdown/PDF pair accepted for one report run."""

    run_root = run_directory(root.resolve(), run_id) / "reports"
    return run_root / "report.md", run_root / "report.pdf"


def _is_allowed_resource(url: str, *, report_directory: Path) -> bool:
    """Allow only in-document data and local assets below the report directory."""

    parsed = urlparse(url)
    if parsed.scheme in {"", "about", "data"}:
        return True
    if parsed.scheme != "file":
        return False
    try:
        resource = Path(unquote(parsed.path).lstrip("/")).resolve()
        return resource.is_relative_to(report_directory.resolve())
    except (OSError, ValueError):
        return False


def export_report_pdf(
    root: Path, *, report_artifact: ArtifactRef, output_pdf: Path | None = None
) -> ArtifactRef:
    """Convert one immutable report Markdown artifact to a PDF artifact."""

    if report_artifact.kind != "report_markdown" or report_artifact.mime_type != "text/markdown":
        raise ValueError("invalid_report_markdown_artifact")
    expected_source, expected_output = _report_paths(root, report_artifact.run_id)
    source = Path(report_artifact.access_ref).resolve()
    if source != expected_source.resolve() or not source.is_file():
        raise ValueError("report_markdown_unavailable")
    source_bytes = source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != report_artifact.hash:
        raise ValueError("report_markdown_integrity_failed")
    try:
        markdown = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("report_markdown_invalid_encoding") from exc

    output = (output_pdf or expected_output).resolve()
    if output != expected_output.resolve():
        raise ValueError("report_pdf_output_path_invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ValueError("pdf_export_runtime_missing") from exc
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = None
            try:
                context = browser.new_context(java_script_enabled=False)
                page = context.new_page()

                def block_untrusted_resources(route):
                    if _is_allowed_resource(route.request.url, report_directory=source.parent):
                        route.continue_()
                    else:
                        route.abort()

                page.route("**/*", block_untrusted_resources)
                page.set_content(_html(markdown, source.parent), wait_until="load")
                page.pdf(path=str(output), format="A4", print_background=True, prefer_css_page_size=True)
            finally:
                if context is not None:
                    context.close()
                browser.close()
    except (PlaywrightError, OSError) as exc:
        raise ValueError("pdf_export_failed") from exc

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = output.with_suffix(".pdf.json")
    manifest.write_text(
        json.dumps(
            {"schema_version": "1", "run_id": report_artifact.run_id,
             "input_hash": report_artifact.hash, "pdf_hash": digest,
             "size": output.stat().st_size},
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return ArtifactRef(
        artifact_id=f"artifact-pdf-{digest[:32]}", run_id=report_artifact.run_id,
        version="1", kind="report_pdf", mime_type="application/pdf",
        size=output.stat().st_size, hash=digest, access_ref=str(output),
        input_versions={"report_markdown": report_artifact.version},
    )
