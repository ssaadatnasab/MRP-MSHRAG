"""
Rule-Based Enrichment Processor
================================
Drop-in replacement for the LLM-based enrichment pipeline.
Instead of returning a *prompt*, every function returns the *processed output*
directly — no API call, no latency, fully deterministic.

Supported tasks (same names as the original):
  math          – LaTeX brace-balance, \\frac repair, RAG skeleton + symbol table
  table         – Column-width alignment, separator row, numeric range extraction
  table_repair  – Extended pre-clean + reconstruction heuristics + RAG block
  code          – Language detection via pattern scoring, GCD-based indent rescaling
  list          – Bullet / numbered-list normalisation
  section       – Heading preservation, first-use **bold** for technical terms
  default       – Spacing, blank-line, inline-code cleanup

CLI usage:
    python rule_based_enrichment.py <input.md> <output.md> [task]

Public API:
    from rule_based_enrichment import process
    result = process(raw_markdown, "table_repair")
"""

from __future__ import annotations

import re
import pathlib
import sys
from math import gcd
from functools import reduce
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _pre_clean(text: str) -> str:
    """
    Normalize markdown table widths only.

    Non-table text is preserved exactly as-is. Table blocks are re-rendered with
    fixed-width columns derived from the widest visible cell in each column.
    """
    def _split_table_row(row: str) -> list[str]:
        stripped = row.strip()
        if stripped.startswith("|") and stripped.endswith("|") and len(stripped) >= 2:
            inner = stripped[1:-1]
        else:
            inner = stripped.strip("|")
        cells = [cell.strip() for cell in inner.split("|")]
        return cells

    def _is_table_separator_row(cells: list[str]) -> bool:
        if not cells:
            return False
        return all(bool(re.fullmatch(r"[:\-\s]*", cell)) for cell in cells)

    def _render_table_block(lines: list[str]) -> list[str]:
        parsed = [_split_table_row(line) for line in lines]
        col_count = max((len(row) for row in parsed), default=0)
        for row in parsed:
            if len(row) < col_count:
                row.extend([""] * (col_count - len(row)))

        data_rows = [row for row in parsed if not _is_table_separator_row(row)]
        if not data_rows:
            return lines

        widths = [max(3, max((len(row[idx]) for row in data_rows), default=0)) for idx in range(col_count)]

        rendered: list[str] = []
        header_row = data_rows[0]
        rendered.append("| " + " | ".join(header_row[idx].ljust(widths[idx]) for idx in range(col_count)) + " |")
        rendered.append("|" + "|".join("-" * (widths[idx] + 2) for idx in range(col_count)) + "|")

        for row in data_rows[1:]:
            rendered.append("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(col_count)) + " |")

        return rendered

    # Detect markdown table blocks and normalize widths only.
    out_lines: list[str] = []
    src_lines = text.splitlines()
    i = 0
    while i < len(src_lines):
        line = src_lines[i]
        if line.strip().startswith("|"):
            tbl_lines: list[str] = []
            while i < len(src_lines) and src_lines[i].strip().startswith("|"):
                tbl_lines.append(src_lines[i])
                i += 1

            out_lines.extend(_render_table_block(tbl_lines))
            continue
        else:
            out_lines.append(line)
            i += 1

    text = "\n".join(out_lines)

    return text.strip()


# Minimal public API for compatibility: perform a pre-cleaning pass and return markdown-safe output
def process(content: str, task: str = "default") -> str:
    """
    Minimal processing entrypoint. Currently performs the rule-based pre-clean and
    returns markdown-formatted text. The `task` argument is accepted for compatibility
    but only `default` and `table_repair` are effectively supported here.
    """
    # For now, always apply _pre_clean which is a safe markdown-preserving pass
    cleaned = _pre_clean(content)
    return cleaned


def _process_path_file(in_path: pathlib.Path, out_path: pathlib.Path) -> None:
    raw = in_path.read_text(encoding="utf-8", errors="replace")
    result = process(raw, "default")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result, encoding="utf-8")


def main() -> None:
    """CLI: accept either file->file or dir->dir. Enforce .md inputs and .md outputs."""
    if len(sys.argv) == 2 and sys.argv[1] == "--test":
        print("Running built-in test: applying pre-clean to sample input")
        print(process("A<br> B  &amp; C", "default"))
        return

    if len(sys.argv) < 3:
        print("Usage:\n  python Hard_Line_Breaks.py <input.md|input-dir> <output.md|output-dir>")
        sys.exit(1)

    input_path = pathlib.Path(sys.argv[1])
    output_path = pathlib.Path(sys.argv[2])

    # Directory mode
    if input_path.is_dir():
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        if not output_path.is_dir():
            print("When input is a directory, output must be a directory", file=sys.stderr)
            sys.exit(1)

        md_files = sorted([p for p in input_path.rglob("*.md") if p.is_file()])
        if not md_files:
            print(f"No .md files found in {input_path}", file=sys.stderr)
            return

        for p in md_files:
            rel_path = p.relative_to(input_path)
            out_file = output_path / rel_path
            if out_file.suffix.lower() != ".md":
                out_file = out_file.with_suffix(".md")
            try:
                out_file.parent.mkdir(parents=True, exist_ok=True)
                _process_path_file(p, out_file)
                print(f"WROTE: {out_file}")
            except Exception as e:
                print(f"ERROR processing {p}: {e}", file=sys.stderr)
        return

    # Single file mode
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if input_path.suffix.lower() != ".md":
        print(f"Input must be a markdown (.md) file: {input_path}", file=sys.stderr)
        sys.exit(1)

    if output_path.exists() and output_path.is_dir():
        print("Output path is a directory; provide an output file path when input is a file", file=sys.stderr)
        sys.exit(1)
    if output_path.suffix.lower() != ".md":
        output_path = output_path.with_suffix(".md")

    _process_path_file(input_path, output_path)
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
