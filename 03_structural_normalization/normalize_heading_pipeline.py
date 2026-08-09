#!/usr/bin/env python3
"""Layered Markdown heading refinement (Layer 1 detection → Layer 2 routing).

Default input folder:
- /home/user/Desktop/Improving Quality MarkDown/Testing Files

Default output folder:
- /home/user/Desktop/Improving Quality MarkDown/Output/1-Refinement 2-Enrichment/1-

Layer 1 — Heading Detection (Whole-Document Decision):
- Detect headings in the input and make one decision per file:
        - numbered (e.g., "1. Introduction", "2.1 Methodology")
        - non-numbered (e.g., "Abstract", "Conclusion", "References")
- Uses Ollama if available (env: OLLAMA_BASE_URL, OLLAMA_MODEL), otherwise falls back
    to a conservative heuristic classifier.
- When both LLM and heuristic are available, the script checks whether they align.
    If they disagree, the LLM gets one double-check pass; if it is still inconsistent,
    the heuristic wins.

Layer 2 — Tool-based Refinement:
- non-numbered → mdast-normalize-headings (Node/remark)
- numbered → marktripy (Python)

Writes one refined Markdown file per input file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, TypedDict

DEFAULT_INPUT_DIR = Path("/home/user/Desktop/Improving Quality MarkDown/Testing Files")
DEFAULT_OUTPUT_DIR = Path("/home/user/Desktop/Improving Quality MarkDown/Output/1-Refinement 2-Enrichment/1-")

# Layered pipeline (Layer 1 + Layer 2)
# ─────────────────────────────────────────────────────────────────────────────

HeadingCategory = Literal["numbered", "non-numbered"]


class Layer1Report(TypedDict):
    document_format: HeadingCategory
    detector: str
    heading_count: int
    numbered_count: int
    non_numbered_count: int
    llm_used: bool
    llm_double_checked: bool
    heuristic_numbered_ratio: float
    llm_numbered_ratio: float | None
    agreement_ratio: float | None


@dataclass(frozen=True)
class HeadingMatch:
    line_index: int
    level: int
    text: str


@dataclass
class MarkdownChunk:
    category: HeadingCategory
    text: str
    first_heading_level: int | None = None


ATX_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
SETEXT_UNDERLINE_RE = re.compile(r"^\s*(=+|-+)\s*$")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
NUMERIC_HEADING_RE = re.compile(
    r"^\s{0,3}(?P<num>\d+(?:\.\d+)*)(?P<sep>[.)]|[.:-])?\s+(?P<title>\S.*)$"
)


def _toggle_fence(in_fence: str | None, line: str) -> str | None:
    match = FENCE_RE.match(line)
    if not match:
        return in_fence
    fence = match.group(1)
    if in_fence is None:
        return fence
    if fence[0] == in_fence[0]:
        return None
    return in_fence


def extract_heading_matches(markdown: str) -> list[HeadingMatch]:
    """Extract heading lines conservatively (skip fenced code blocks)."""
    lines = markdown.splitlines()
    matches: list[HeadingMatch] = []

    in_fence: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        in_fence = _toggle_fence(in_fence, line)
        if in_fence is not None:
            i += 1
            continue

        atx = ATX_HEADING_RE.match(line)
        if atx:
            level = len(atx.group(1))
            text = atx.group(2).strip()
            matches.append(HeadingMatch(line_index=i, level=level, text=text))
            i += 1
            continue

        if i + 1 < len(lines) and lines[i].strip() and SETEXT_UNDERLINE_RE.match(lines[i + 1]):
            underline = lines[i + 1].lstrip()
            level = 1 if underline.startswith("=") else 2
            matches.append(HeadingMatch(line_index=i, level=level, text=lines[i].strip()))
            i += 2
            continue

        numeric = NUMERIC_HEADING_RE.match(line)
        if numeric:
            num = numeric.group("num")
            sep = (numeric.group("sep") or "").strip()

            dot_depth = num.count(".")
            is_decimal = dot_depth >= 1
            is_ambiguous_list_marker = (dot_depth == 0) and (sep in {".", ")"})

            prev_blank = i == 0 or not lines[i - 1].strip()
            next_blank = i + 1 >= len(lines) or not lines[i + 1].strip()

            if is_decimal or (not is_ambiguous_list_marker) or (prev_blank and next_blank):
                level = min(6, dot_depth + 1)
                matches.append(HeadingMatch(line_index=i, level=level, text=line.strip()))

        i += 1

    return matches


def heuristic_heading_category(heading_text: str) -> HeadingCategory:
    def _normalize_for_heuristic(t: str) -> str:
        s = t.strip()
        # remove simple HTML emphasis tags at the ends
        s = re.sub(r'(?i)^\s*<(?:strong|b|em|i)>\s*', '', s)
        s = re.sub(r'(?i)\s*</(?:strong|b|em|i)>\s*$', '', s)
        # unwrap common leading/trailing emphasis/quote/backtick characters
        s = re.sub(r'^[\*\_`\"\']+', '', s)
        s = re.sub(r'[\*\_`\"\']+$', '', s)
        # remove common leading punctuation/markers like '> ', '- ', '* '
        s = re.sub(r'^[\s>\-\*]+', '', s)
        return s.lstrip()

    norm = _normalize_for_heuristic(heading_text)
    return "numbered" if re.match(r"^\s*\d+(?:\.\d+)*\b", norm) else "non-numbered"


def _ollama_generate(base_url: str, model: str, prompt: str, timeout: int) -> str | None:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
        text = str(parsed.get("response", "")).strip()
        return text or None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def _extract_json_array(text: str) -> list[str] | None:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        value = json.loads(snippet)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return [item.strip() for item in value]


def ollama_classify_headings(
    headings: list[str],
    *,
    base_url: str,
    model: str,
    timeout: int = 1800000,
) -> list[HeadingCategory] | None:
    if not headings:
        return []

    prompt_lines = [


        "You are a markdown heading format classifier. Your only task is to determine whether the body section headings of the given markdown document follow a numbered or non-numbered convention.",
         
        "Definitions:numbered: The majority of body section headings (lines starting with #, ##, ###, or ####) begin with a numeric prefix immediately after the # symbols and any surrounding ** bold markers. The numeric prefix can take forms such as 1, 1., 1.1, 1.1.1, 2., 3.1, etc. (with or without trailing period, with or without bold **).",
        "Examples of numbered headings: ### **1. INTRODUCTION** #### **3 1 Terminology** # **1 Classics of Data-Driven BIM for Energy Efficient Design**",


        "non-numbered: The majority of body section headings begin directly with a descriptive word or title, containing no leading numeric prefix.",
        "Examples of non-numbered headings:## **Foreword** #### THE BUILDING AS A HABITAT ### Preface",

        "Instructions:",

        "Ignore front matter such as title pages, author lists, copyright notices, and table of contents entries (inside markdown tables).",
        "Focus exclusively on lines that begin with one or more # characters (markdown headings) appearing in the document body.",
        "Determine whether the majority of those heading lines carry a numeric prefix (numbers like 1, 2., 1.1, 3.2.1, etc.) right after the # symbols and any optional ** bold markers.",
        
        "Allowed values for each document are just: [\"numbered\", \"non-numbered\"].",
        
        "Pay attention that for each document, you have to just mention one value for each document according to their headings. Not writing for any heading in one document. You should decide the format of the whole document based on the majority of its headings and write only that single format value for the whole document.",
        
        "",
        "Document to classify:",
    ]
    
    for idx, heading in enumerate(headings, start=1):
        prompt_lines.append(f"{idx}. {heading}")
    prompt = "\n".join(prompt_lines)

    raw = _ollama_generate(base_url, model, prompt, timeout=timeout)
    if not raw:
        return None

    items = _extract_json_array(raw)
    if not items or len(items) != len(headings):
        return None

    normalized: list[HeadingCategory] = []
    for item in items:
        low = item.strip().lower()
        if low in {"numbered", "non-numbered"}:
            normalized.append("numbered" if low == "numbered" else "non-numbered")
        else:
            return None

    return normalized


def _agreement_ratio(a: list[HeadingCategory], b: list[HeadingCategory] | None) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    same = sum(1 for x, y in zip(a, b) if x == y)
    return same / len(a)


def _numbered_ratio(categories: list[HeadingCategory]) -> float:
    if not categories:
        return 0.0
    return sum(1 for category in categories if category == "numbered") / len(categories)


def _choose_document_format(categories: list[HeadingCategory]) -> HeadingCategory:
    numbered = sum(1 for category in categories if category == "numbered")
    non_numbered = len(categories) - numbered
    return "numbered" if numbered >= non_numbered else "non-numbered"


def ollama_double_check_headings(
    headings: list[str],
    *,
    heuristic: list[HeadingCategory],
    base_url: str,
    model: str,
    timeout: int = 1800000,
) -> list[HeadingCategory] | None:
    """Ask the LLM to reconcile with the heuristic once when there is disagreement."""
    if not headings:
        return []
    if len(headings) != len(heuristic):
        return None

    prompt_lines = [


        "You are a markdown heading format classifier. Your only task is to determine whether the body section headings of the given markdown document follow a numbered or non-numbered convention.",
         
        "Definitions:numbered: The majority of body section headings (lines starting with #, ##, ###, or ####) begin with a numeric prefix immediately after the # symbols and any surrounding ** bold markers. The numeric prefix can take forms such as 1, 1., 1.1, 1.1.1, 2., 3.1, etc. (with or without trailing period, with or without bold **).",
        "Examples of numbered headings: ### **1. INTRODUCTION** #### **3 1 Terminology** # **1 Classics of Data-Driven BIM for Energy Efficient Design**",


        "non-numbered: The majority of body section headings begin directly with a descriptive word or title, containing no leading numeric prefix.",
        "Examples of non-numbered headings:## **Foreword** #### THE BUILDING AS A HABITAT ### Preface",

        "Instructions:",

        "Ignore front matter such as title pages, author lists, copyright notices, and table of contents entries (inside markdown tables).",
        "Focus exclusively on lines that begin with one or more # characters (markdown headings) appearing in the document body.",
        "Determine whether the majority of those heading lines carry a numeric prefix (numbers like 1, 2., 1.1, 3.2.1, etc.) right after the # symbols and any optional ** bold markers.",
        
        "Allowed values: [\"numbered\", \"non-numbered\"].",
        "",
        "Input headings:",
    ]
    
    
    
    for idx, (heading, label) in enumerate(zip(headings, heuristic), start=1):
        prompt_lines.append(f"{idx}. {heading}  => heuristic: {label}")
    prompt = "\n".join(prompt_lines)

    raw = _ollama_generate(base_url, model, prompt, timeout=timeout)
    if not raw:
        return None

    items = _extract_json_array(raw)
    if not items or len(items) != len(headings):
        return None

    normalized: list[HeadingCategory] = []
    for item in items:
        low = item.strip().lower()
        if low == "numbered":
            normalized.append("numbered")
        elif low in {"non-numbered", "non numbered", "nonnumeric", "non-numeric"}:
            normalized.append("non-numbered")
        else:
            return None
    return normalized


def detect_document_heading_format(markdown: str) -> tuple[HeadingCategory, Layer1Report]:
    """Layer 1: detect headings once and choose one document-level format."""
    heading_matches = extract_heading_matches(markdown)
    if not heading_matches:
        report: Layer1Report = {
            "document_format": "non-numbered",
            "detector": "none",
            "heading_count": 0,
            "numbered_count": 0,
            "non_numbered_count": 0,
            "llm_used": False,
            "llm_double_checked": False,
            "heuristic_numbered_ratio": 0.0,
            "llm_numbered_ratio": None,
            "agreement_ratio": None,
        }
        return "non-numbered", report

    headings = [match.text for match in heading_matches]
    heuristic = [heuristic_heading_category(text) for text in headings]

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3.1:70b")
    llm = ollama_classify_headings(headings, base_url=base_url, model=model)
    llm_used = llm is not None
    llm_double_checked = False

    agreement = _agreement_ratio(heuristic, llm)
    if llm is not None and agreement is not None and agreement < 0.85:
        checked = ollama_double_check_headings(
            headings,
            heuristic=heuristic,
            base_url=base_url,
            model=model,
        )
        if checked is not None:
            llm = checked
            llm_double_checked = True
            agreement = _agreement_ratio(heuristic, llm)

    categories = llm if llm is not None and agreement is not None and agreement >= 0.65 else heuristic
    detector = "ollama" if categories is llm else "heuristic"
    document_format = _choose_document_format(categories)

    report: Layer1Report = {
        "document_format": document_format,
        "detector": detector,
        "heading_count": len(categories),
        "numbered_count": sum(1 for category in categories if category == "numbered"),
        "non_numbered_count": sum(1 for category in categories if category == "non-numbered"),
        "llm_used": llm_used,
        "llm_double_checked": llm_double_checked,
        "heuristic_numbered_ratio": _numbered_ratio(heuristic),
        "llm_numbered_ratio": _numbered_ratio(llm) if llm is not None else None,
        "agreement_ratio": agreement,
    }
    return document_format, report


def _first_atx_heading_level(markdown: str) -> int | None:
    in_fence: str | None = None
    for line in markdown.splitlines():
        in_fence = _toggle_fence(in_fence, line)
        if in_fence is not None:
            continue
        match = ATX_HEADING_RE.match(line)
        if match:
            return len(match.group(1))
    return None


def _shift_atx_heading_levels(markdown: str, delta: int) -> str:
    if delta == 0:
        return markdown

    out_lines: list[str] = []
    in_fence: str | None = None
    for line in markdown.splitlines(keepends=False):
        in_fence = _toggle_fence(in_fence, line)
        if in_fence is None:
            match = ATX_HEADING_RE.match(line)
            if match:
                level = len(match.group(1))
                new_level = max(1, min(6, level + delta))
                prefix = line[: match.start(1)]
                rest = line[match.end(1) :]
                line = prefix + ("#" * new_level) + rest
        out_lines.append(line)
    return "\n".join(out_lines)


def _merge_chunks(chunks: list[MarkdownChunk]) -> list[MarkdownChunk]:
    merged: list[MarkdownChunk] = []
    for chunk in chunks:
        if not chunk.text:
            continue
        if not merged or merged[-1].category != chunk.category:
            merged.append(chunk)
            continue
        merged[-1].text += chunk.text
        if merged[-1].first_heading_level is None and chunk.first_heading_level is not None:
            merged[-1].first_heading_level = chunk.first_heading_level
    return merged


def build_layered_chunks(markdown: str) -> tuple[list[MarkdownChunk], str]:
    """Layer 1: detect headings and categorize them as numbered vs non-numbered."""
    heading_matches = extract_heading_matches(markdown)
    if not heading_matches:
        return [MarkdownChunk(category="non-numbered", text=markdown, first_heading_level=None)], "detector=none"

    heading_texts = [m.text for m in heading_matches]

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3.1:70b")
    categories = ollama_classify_headings(heading_texts, base_url=base_url, model=model)
    detector_backend = "ollama" if categories is not None else "heuristic"
    if categories is None:
        categories = [heuristic_heading_category(text) for text in heading_texts]

    lines_ke = markdown.splitlines(keepends=True)
    indices = [m.line_index for m in heading_matches]

    chunks: list[MarkdownChunk] = []
    if indices[0] > 0:
        chunks.append(MarkdownChunk(category="non-numbered", text="".join(lines_ke[: indices[0]])))

    for idx, (match, category) in enumerate(zip(heading_matches, categories)):
        start = match.line_index
        end = heading_matches[idx + 1].line_index if idx + 1 < len(heading_matches) else len(lines_ke)
        chunks.append(
            MarkdownChunk(
                category=category,
                text="".join(lines_ke[start:end]),
                first_heading_level=match.level,
            )
        )

    return _merge_chunks(chunks), f"detector={detector_backend}"


def layered_mdast_marktripy(markdown: str, work_dir: Path) -> tuple[str, str]:
    """Layer 2: route chunks to mdast (non-numbered) or marktripy (numbered)."""
    chunks, detector_note = build_layered_chunks(markdown)

    numbered_backends: set[str] = set()
    non_numbered_backends: set[str] = set()
    out_parts: list[str] = []

    for chunk in chunks:
        if chunk.category == "non-numbered":
            processed, backend = mdast_util_normalize_headings(
                chunk.text,
                work_dir,
                fallback_fn=lambda s: remove_formula_context_labels(normalize_heading_hierarchy(s)),
            )
            non_numbered_backends.add(backend)
        else:
            processed, backend = marktripy_normalize(
                chunk.text,
                work_dir,
                fallback_fn=lambda s: remove_formula_context_labels(normalize_heading_hierarchy(promote_numbered_headings(s))),
                postprocess_fn=remove_formula_context_labels,
            )
            numbered_backends.add(backend)

        if chunk.first_heading_level is not None:
            new_level = _first_atx_heading_level(processed)
            if new_level is not None and new_level != chunk.first_heading_level:
                processed = _shift_atx_heading_levels(processed, chunk.first_heading_level - new_level)

        out_parts.append(processed)

    def summarize(values: set[str]) -> str:
        if not values:
            return "n/a"
        if len(values) == 1:
            return next(iter(values))
        return "mixed"

    backend_summary = (
        f"layered_mdast_marktripy({detector_note}, "
        f"non_numbered={summarize(non_numbered_backends)}, "
        f"numbered={summarize(numbered_backends)})"
    )
    return "".join(out_parts), backend_summary
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def normalize_heading_hierarchy(content: str) -> str:
    lines = content.splitlines()
    levels_used: set[int] = set()
    for line in lines:
        match = re.match(r"^(#{1,6}) ", line)
        if match:
            levels_used.add(len(match.group(1)))

    if 1 not in levels_used or 2 in levels_used:
        return content

    levels_below_h1 = sorted(level for level in levels_used if level > 1)
    if not levels_below_h1:
        return content

    gap = levels_below_h1[0] - 2
    if gap <= 0:
        return content

    result = []
    for line in lines:
        match = re.match(r"^(#{2,6})( .*)", line)
        if match:
            current_level = len(match.group(1))
            new_level = max(2, current_level - gap)
            line = "#" * new_level + match.group(2)
        result.append(line)
    return "\n".join(result)


def promote_numbered_headings(content: str) -> str:
    lines = content.splitlines()
    out_lines: list[str] = []
    pattern = re.compile(
        r"^\s*(?P<prefix>(?:#{1,6}\s*)?(?:[-*+]\s+)?(?:>\s+)?)"
        r"(?P<bold>\*{0,2})?"
        r"(?P<num>\d+(?:\.\d+)*)"
        r"(?P<sep>[\s.:\-)]+)?"
        r"(?P<title>.*)$"
    )

    for line in lines:
        match = pattern.match(line)
        if not match:
            out_lines.append(line)
            continue

        sep = match.group("sep") or ""
        if ")" in sep:
            out_lines.append(line)
            continue

        num = match.group("num")
        level = min(6, num.count(".") + 1)
        title = match.group("title").strip()
        title = re.sub(r"^\*{1,2}", "", title).strip()
        title = re.sub(r"\*{1,2}$", "", title).strip()
        out_lines.append(f"{'#' * level} {num} {title}".rstrip())

    return "\n".join(out_lines)


def remove_formula_context_labels(content: str) -> str:
    return re.sub(
        r"(?mi)^\s*(?:Formula\s*context\s*for\s*RAG|RAG\s*enrichment\s*block):?\s*$\n?",
        "",
        content,
    )



def run_command(
    command: list[str],
    cwd: Path | None = None,
    timeout: int | None = None,
) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        return completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        timeout_note = f"timeout after {timeout}s"
        combined = "\n".join(part for part in [stderr.strip(), timeout_note] if part)
        return 124, stdout, combined


def node_available() -> bool:
    return shutil.which("node") is not None or shutil.which("npx") is not None


def ensure_node_project(project_dir: Path, packages: list[str]) -> None:
    """Create a cached npm project with the requested packages installed."""
    package_json = project_dir / "package.json"
    if not package_json.exists():
        project_dir.mkdir(parents=True, exist_ok=True)
        run_command(["npm", "init", "-y"], cwd=project_dir, timeout=120000)
        run_command(["npm", "install", *packages], cwd=project_dir, timeout=60000)


def run_node_markdown_tool(
    content: str,
    work_dir: Path,
    method_key: str,
    packages: list[str],
    script_source: str,
    timeout: int,
    *,
    fallback_fn: Callable[[str], str] = lambda s: s,
) -> tuple[str, str]:
    if not node_available():
        return fallback_fn(content), "fallback: node/npm not available"

    project_dir = work_dir / "node_projects" / method_key
    ensure_node_project(project_dir, packages)

    input_path = project_dir / "input.md"
    output_path = project_dir / "output.md"
    script_path = project_dir / f"{method_key}.mjs"
    write_text(input_path, content)
    script_path.write_text(script_source, encoding="utf-8")

    code, _, stderr = run_command(
        ["node", str(script_path), str(input_path), str(output_path)],
        cwd=project_dir,
        timeout=timeout,
    )
    if code != 0 or not output_path.exists():
        detail = stderr.strip() or f"{method_key} failed"
        return fallback_fn(content), f"fallback: {detail}"
    return read_text(output_path), method_key


def mdast_util_normalize_headings(
    content: str,
    work_dir: Path,
    *,
    fallback_fn: Callable[[str], str] = lambda s: s,
) -> tuple[str, str]:
    return run_node_markdown_tool(
        content,
        work_dir,
        "mdast_util_normalize_headings",
        ["remark", "remark-parse", "remark-stringify", "mdast-normalize-headings"],
        """
import fs from 'fs';
import {remark} from 'remark';
import remarkParse from 'remark-parse';
import remarkStringify from 'remark-stringify';
import * as normalizeHeadingsMod from 'mdast-normalize-headings';

const normalizeHeadings =
    normalizeHeadingsMod.default ||
    normalizeHeadingsMod.normalizeHeadings ||
    normalizeHeadingsMod;

async function main() {
    const inputPath = process.argv[2];
    const outputPath = process.argv[3];
    const source = await fs.promises.readFile(inputPath, 'utf8');
    const processor = remark().use(remarkParse).use(remarkStringify);
    const tree = processor.parse(source);
    normalizeHeadings(tree);
    const rendered = processor.stringify(tree);
    await fs.promises.writeFile(outputPath, String(rendered), 'utf8');
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
""".strip()
        + "\n",
        timeout=1800000,
        fallback_fn=fallback_fn,
    )


def marktripy_normalize(
    content: str,
    work_dir: Path,
    *,
    fallback_fn: Callable[[str], str] = lambda s: s,
    postprocess_fn: Callable[[str], str] = lambda s: s,
) -> tuple[str, str]:
    try:
        from marktripy.parsers.markdown_it import MarkdownItParser
        from marktripy.renderers.markdown import MarkdownRenderer
        from marktripy.transformers.heading import normalize_headings
    except Exception as exc:
        return fallback_fn(content), f"fallback: {exc}"

    try:
        parser = MarkdownItParser({"preset": "commonmark"})
        ast = parser.parse(content)
        ast = normalize_headings(ast)
        renderer = MarkdownRenderer()
        rendered = renderer.render(ast)
        rendered = postprocess_fn(rendered)
        return rendered, "marktripy"
    except Exception as exc:
        return fallback_fn(content), f"fallback: {exc}"


def refine_markdown(markdown: str, work_dir: Path) -> tuple[str, str]:
    """Layer 2: route the whole document to one backend based on Layer 1."""
    document_format, report = detect_document_heading_format(markdown)

    if document_format == "numbered":
        processed, backend = marktripy_normalize(
            markdown,
            work_dir,
            fallback_fn=lambda s: remove_formula_context_labels(normalize_heading_hierarchy(s)),
            postprocess_fn=remove_formula_context_labels,
        )
    else:
        processed, backend = mdast_util_normalize_headings(
            markdown,
            work_dir,
            fallback_fn=lambda s: remove_formula_context_labels(normalize_heading_hierarchy(s)),
        )

    processed = remove_formula_context_labels(normalize_heading_hierarchy(processed))

    backend_summary = (
        f"refine_markdown(format={report['document_format']}, detector={report['detector']}, "
        f"headings={report['heading_count']}, numbered={report['numbered_count']}, "
        f"non_numbered={report['non_numbered_count']}, agreement={report['agreement_ratio']}, "
        f"llm_used={report['llm_used']}, double_checked={report['llm_double_checked']}, "
        f"backend={backend})"
    )
    return processed, backend_summary


def collect_markdown_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".md"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refine Markdown headings with a layered mdast/marktripy router.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Folder containing input Markdown files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder where refined Markdown files will be written.",
    )
    parser.add_argument(
        "--section",
        choices=["heuristic", "llm", "marktripy", "mdast"],
        help="Run a single pipeline section for quick checks.",
    )
    parser.add_argument(
        "--file",
        help="Path to a single Markdown file to run the selected section on.",
    )
    parser.add_argument(
        "--output-file",
        help="Optional path to write the section output (for marktripy/mdast).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    sources = collect_markdown_files(input_dir)

    def run_section(section: str, file_path: Path | None, output_file: str | None) -> int:
        work_dir = output_dir / "_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        if file_path is None:
            if section not in {"heuristic", "llm"}:
                print("A single --file is required for this section.")
                return 2

            files = collect_markdown_files(input_dir)
            if not files:
                print(f"No .md files found in {input_dir}")
                return 0

            results = 0
            for path in files:
                md = read_text(path)
                if section == "heuristic":
                    matches = extract_heading_matches(md)
                    headings = [m.text for m in matches]
                    labels = [heuristic_heading_category(h) for h in headings]
                    out = {
                        "file": str(path),
                        "headings": headings,
                        "heuristic_labels": labels,
                        "numbered_count": sum(1 for l in labels if l == "numbered"),
                        "non_numbered_count": sum(1 for l in labels if l == "non-numbered"),
                    }
                    print(json.dumps(out, indent=2))
                    results += 1
                    continue

                base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                model = os.environ.get("OLLAMA_MODEL", "llama3.1:70b")
                matches = extract_heading_matches(md)
                headings = [m.text for m in matches]
                llm = ollama_classify_headings(headings, base_url=base_url, model=model)
                out = {"file": str(path), "headings": headings, "llm_labels": llm}
                print(json.dumps(out, indent=2))
                results += 1

            return 0 if results >= 0 else 2

        if not file_path.exists():
            print(f"File not found: {file_path}")
            return 2

        md = read_text(file_path)

        if section == "heuristic":
            matches = extract_heading_matches(md)
            headings = [m.text for m in matches]
            labels = [heuristic_heading_category(h) for h in headings]
            out = {
                "file": str(file_path),
                "headings": headings,
                "heuristic_labels": labels,
                "numbered_count": sum(1 for l in labels if l == "numbered"),
                "non_numbered_count": sum(1 for l in labels if l == "non-numbered"),
            }
            print(json.dumps(out, indent=2))
            return 0

        if section == "llm":
            matches = extract_heading_matches(md)
            headings = [m.text for m in matches]
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            model = os.environ.get("OLLAMA_MODEL", "llama3.1:70b")
            llm = ollama_classify_headings(headings, base_url=base_url, model=model)
            out = {"file": str(file_path), "headings": headings, "llm_labels": llm}
            print(json.dumps(out, indent=2))
            return 0

        if section == "marktripy":
            processed, backend = marktripy_normalize(md, work_dir)
            if output_file:
                Path(output_file).write_text(processed, encoding="utf-8")
                print(f"Wrote processed output to {output_file}")
            else:
                print(processed)
            return 0

        if section == "mdast":
            processed, backend = mdast_util_normalize_headings(md, work_dir)
            if output_file:
                Path(output_file).write_text(processed, encoding="utf-8")
                print(f"Wrote processed output to {output_file}")
            else:
                print(processed)
            return 0

        print(f"Unknown section: {section}")
        return 3
    if args.section:
        file_arg = Path(args.file) if args.file else (sources[0] if sources else None)
        return run_section(args.section, file_arg, args.output_file)

    if not sources:
        print(f"No .md files found in {input_dir}")
        return 0

    wrote = 0
    for source in sources:
        original = read_text(source)
        refined, backend = refine_markdown(original, work_dir)
        out_path = output_dir / source.name
        write_text(out_path, refined)
        wrote += 1
        print(f"- {source.name} -> {out_path} [{backend}]")

    print(f"Wrote {wrote} refined Markdown files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
