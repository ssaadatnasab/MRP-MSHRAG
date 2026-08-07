#!/usr/bin/env python3
"""Clean page-marker artifacts from Markdown files in a folder tree."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION
# Defaults below are relative to wherever you run this script from. Override
# either with the matching CLI flag instead of editing this file.
# ──────────────────────────────────────────────────────────────────────────
DEFAULT_INPUT_DIR = Path("./data/02_markdown_extraction")
DEFAULT_OUTPUT_DIR = Path("./data/03_structural_normalization/01_cleaned")
# ──────────────────────────────────────────────────────────────────────────


def remove_page_span_tags(content: str) -> tuple[str, int]:
    """Remove common span-based page markers.

    Removes:
    - empty span tags with an id attribute: `<span id="section-1"></span>`
    - page-id spans with a page label inside: `<span id="page-31-0">Page 31</span>`

    Keeps other non-empty spans (non-page ids) intact.

    This handles:
    - id values quoted with single/double quotes or unquoted
    - arbitrary other attributes and attribute order
    - case-insensitive tag/attribute names
    - emptiness checks that treat common HTML space entities as whitespace

    Returns a tuple of (new_content, removed_count).
    """


    removed = 0

    # Replace common non-breaking entities with a space for emptiness checks
    def _repl(match: re.Match) -> str:
        nonlocal removed
        val = match.group("val")
        # strip surrounding quotes if present
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val_unq = val[1:-1]
        else:
            val_unq = val

        inner = match.group("inner") or ""
        # Treat common HTML space entities as whitespace
        inner_norm = re.sub(r"(&nbsp;|\u00A0|\u200B)", " ", inner, flags=re.IGNORECASE)
        inner_compact = re.sub(r"\s+", " ", inner_norm).strip()

        id_val = val_unq.strip()

        # Remove empty spans (any id).
        if re.fullmatch(r"[\s\u00A0\u200B]*", inner_norm):
            removed += 1
            return ""

        # Remove page-marker spans even if they contain only a page label.
        is_page_id = re.match(r"^page[\\\-\d_]*$", id_val, flags=re.IGNORECASE) is not None
        if is_page_id and re.match(
            r"^(?:page\s*)?\d{1,6}$|^page\s*\d{1,6}(?:\s*[-–]\s*\d{1,6})?$",
            inner_compact,
            flags=re.IGNORECASE,
        ):
            removed += 1
            return ""
        return match.group(0)

    # find span tags with an id=... attribute (quoted or unquoted). Capture the id value and inner.
    pattern = re.compile(
        r"""(?is)<span\b[^>]*\bid\s*=\s*(?P<val>"[^"]*"|'[^']*'|[^>\s]+)[^>]*>(?P<inner>.*?)</span>"""
    )

    new_content = pattern.sub(_repl, content)
    return new_content, removed


def unwrap_markdown_page_anchor_links(content: str) -> tuple[str, int]:
    """Unwrap Markdown links that point to numeric #page anchors.

    Examples:
    - `## **[Foreword](#page-6-0)**` -> `## **Foreword**`
    - `hazard[,1](#page-68-0)` -> `hazard,1`

    Returns a tuple of (new_content, unwrapped_count).
    """

    link_pattern = re.compile(
        r"""\[(?P<label>[^\]]*?)\]\(\s*(?P<target>#page(?:[-_]?\d+){1,3})\s*\)""",
        flags=re.IGNORECASE,
    )

    # If a line ends with a page number immediately after a #page link (often a TOC entry),
    # drop that trailing page number token (roman or digits), keeping any closing emphasis.
    toc_line_ends_with_page_no = re.compile(
        r"""\)\s*\*{0,3}\s+(?P<num>(?:[ivxlcdm]{1,12}|\d{1,4}))\s*\*{0,3}\s*$""",
        flags=re.IGNORECASE,
    )
    strip_trailing_page_no = re.compile(
        r"""\s+(?P<num>(?:[ivxlcdm]{1,12}|\d{1,4}))(?P<em>\*{0,3})\s*$""",
        flags=re.IGNORECASE,
    )

    def _repl(match: re.Match) -> str:
        label = (match.group("label") or "").strip()
        return label

    keep_trailing_newline = content.endswith("\n")
    unwrapped_total = 0
    out_lines: list[str] = []
    for line in content.splitlines():
        new_line, n = link_pattern.subn(_repl, line)
        unwrapped_total += n

        if n and toc_line_ends_with_page_no.search(line):
            new_line = strip_trailing_page_no.sub(lambda m: m.group("em") or "", new_line).rstrip()

        out_lines.append(new_line)

    new_content = "\n".join(out_lines)
    if keep_trailing_newline:
        new_content += "\n"
    return new_content, unwrapped_total


def clean_markdown_file(
    input_file: Path,
    output_file: Path,
) -> dict[str, int]:
    text = input_file.read_text(encoding="utf-8", errors="replace")

    removed_spans = 0
    removed_links = 0

    cleaned, removed_spans = remove_page_span_tags(text)
    cleaned, removed_links = unwrap_markdown_page_anchor_links(cleaned)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(cleaned, encoding="utf-8")
    return {
        "spans": removed_spans,
        "page_links": removed_links,
    }


def iter_markdown_files(root_dir: Path):
    for file_path in sorted(root_dir.rglob("*.md")):
        if file_path.is_file():
            yield file_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Remove common page-marker artifacts from Markdown files (span markers and #page links)."
        )
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Source folder to clean.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Destination folder for cleaned files.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")

    processed_count = 0
    totals = {"spans": 0, "page_links": 0}
    for input_file in iter_markdown_files(input_dir):
        relative_path = input_file.relative_to(input_dir)
        output_file = output_dir / relative_path
        removed = clean_markdown_file(
            input_file,
            output_file,
        )
        processed_count += 1
        for k in totals:
            totals[k] += int(removed.get(k, 0))

    print(f"Cleaned {processed_count} Markdown file(s) into: {output_dir}")
    print(f"Removed {totals['spans']} span marker tag(s)")
    print(f"Unwrapped {totals['page_links']} #page anchor link(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
