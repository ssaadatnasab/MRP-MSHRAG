#!/usr/bin/env python3
"""
Markdown Chunker — Strategy 2
=============================

Strategy 2: Content-type aware chunking
    Separate pipelines for text, equations, and tables.
    Each chunk tagged with content_type metadata.
    Outputs: strategy2_text.jsonl, strategy2_equations.jsonl,
                     strategy2_tables.jsonl, strategy2_all.jsonl

Usage:
        python "Content-type aware.py"
        python "Content-type aware.py" --input /path/to/md
        python "Content-type aware.py" --output /path/to/output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as exc:  # pragma: no cover - user-facing guidance
    raise SystemExit(
        "Missing dependency: install it with `pip install langchain-text-splitters`."
    ) from exc

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_INPUT = Path(os.getenv("MARKDOWN_INPUT_DIR", "./input_markdown"))
DEFAULT_OUTPUT = Path(os.getenv("MARKDOWN_OUTPUT_DIR", "./output/content_type_aware"))

# ─────────────────────────────────────────────────────────────────────────────
# Token estimation (no network needed)
# ─────────────────────────────────────────────────────────────────────────────

CHARS_PER_TOKEN = 4   # standard English prose approximation

def chars(tokens: int) -> int:
    """Convert target token count to character count."""
    return tokens * CHARS_PER_TOKEN

def tokens(text: str) -> int:
    """Estimate token count from character count."""
    return max(1, len(text) // CHARS_PER_TOKEN)

# ─────────────────────────────────────────────────────────────────────────────
# Chunk dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id:        str
    text:            str
    content_type:    str            # text | equation | table
    source_file:     str
    has_variables:   bool = False


def make_id(text: str, source: str) -> str:
    h = hashlib.md5(f"{source}::{text[:120]}".encode()).hexdigest()[:12]
    return f"chunk_{h}"


# ─────────────────────────────────────────────────────────────────────────────
# Markdown structure parsers
# ─────────────────────────────────────────────────────────────────────────────

FENCE_RE      = re.compile(r"^\s*(`{3,}|~{3,})")
HEADING_RE    = re.compile(r"^(#{1,6})\s+(.*)")
TABLE_ROW_RE  = re.compile(r"^\s*\|")
TABLE_SEP_RE  = re.compile(r"^\s*\|[\s|:-]+\|\s*$")
DISPLAY_EQ_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
INLINE_EQ_RE  = re.compile(r"\$([^$\n]{2,200})\$")
RAG_META_RE   = re.compile(r"<!--\s*RAG-META.*?-->", re.DOTALL)

TABLE_META_LABEL_RE = re.compile(
    r"^\s*\*\*(Table\s+subject|Property\s+type|Columns\s+and\s+units|Domain\s+keywords)\s*:\*\*",
    re.IGNORECASE,
)


def _toggle_fence(in_fence: str | None, line: str) -> str | None:
    m = FENCE_RE.match(line)
    if not m:
        return in_fence
    fence = m.group(1)
    return None if in_fence and fence[0] == in_fence[0] else fence


def _current_heading(lines: list[str], up_to: int) -> str:
    """Return the deepest heading that appears before line `up_to`."""
    best = ""
    for i in range(up_to):
        m = HEADING_RE.match(lines[i])
        if m:
            best = m.group(2).strip()
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Equation extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_equations(markdown: str, source_file: str) -> list[Chunk]:
    """
    Extract every display equation ($$...$$) together with:
      - surrounding variable definitions ("where:" clause)
      - section heading context
      - RAG-META block if present (from Layer-3 enrichment)

    Each equation becomes exactly ONE chunk. Two equations are never merged.
    """
    lines = markdown.splitlines()
    chunks: list[Chunk] = []
    in_fence: str | None = None
    i = 0

    while i < len(lines):
        in_fence = _toggle_fence(in_fence, lines[i])
        if in_fence:
            i += 1
            continue

        # ── detect $$...$$  block ────────────────────────────────────────────
        if lines[i].strip().startswith("$$"):
            eq_start = i
            eq_lines = [lines[i]]
            i += 1
            while i < len(lines) and "$$" not in lines[i]:
                eq_lines.append(lines[i])
                i += 1
            if i < len(lines):
                eq_lines.append(lines[i])
            i += 1

            latex_raw = "\n".join(eq_lines)
            latex_inner = re.sub(r"\$\$", "", latex_raw).strip()

            # look ahead for "where:" variable definitions
            var_lines: list[str] = []
            j = i
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and re.match(r"^\s*where\b", lines[j], re.I):
                j += 1
                while j < len(lines) and lines[j].strip() and not HEADING_RE.match(lines[j]):
                    var_lines.append(lines[j])
                    j += 1
                i = j   # advance past variable block

            # look for RAG-META block immediately before or after
            rag_meta = ""
            window_before = "\n".join(lines[max(0, eq_start - 15): eq_start])
            window_after  = "\n".join(lines[i: i + 15])
            for m in RAG_META_RE.finditer(window_before + "\n" + window_after):
                rag_meta = m.group(0)
                break

            # assemble chunk text
            parts = [latex_raw]
            if var_lines:
                parts.append("where:")
                parts.extend(var_lines)
            if rag_meta:
                parts.append(rag_meta)

            text = "\n".join(parts).strip()
            if not text:
                continue

            cid = make_id(text, source_file)
            chunks.append(Chunk(
                chunk_id=cid,
                text=text,
                content_type="equation",
                source_file=source_file,
                has_variables=bool(var_lines),
            ))
        else:
            i += 1

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Table extractor
# ─────────────────────────────────────────────────────────────────────────────

def _table_to_json_rows(table_lines: list[str]) -> list[dict]:
    """Convert a markdown pipe table to a list of row dicts."""
    rows = [l for l in table_lines if TABLE_ROW_RE.match(l) and not TABLE_SEP_RE.match(l)]
    if not rows:
        return []

    def split_row(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    header = split_row(rows[0])
    result = []
    for data_row in rows[1:]:
        cells = split_row(data_row)
        # pad or trim to header length
        cells = (cells + [""] * len(header))[: len(header)]
        result.append(dict(zip(header, cells)))
    return result


def _is_metadata_table_block(table_lines: list[str]) -> bool:
    """Heuristic: detect short schema/index-metadata tables that should be merged."""
    table_text = "\n".join(table_lines).lower()
    data_rows = [l for l in table_lines if TABLE_ROW_RE.match(l) and not TABLE_SEP_RE.match(l)]
    markers = [
        "column header",
        "unit",
        "category/prerequisite/credit",
        "description",
        "path/subcategory",
        "page numbers",
    ]
    score = sum(1 for m in markers if m in table_text)
    return score >= 2 and len(data_rows) <= 10


def _collect_pre_table_metadata(lines: list[str], tbl_start: int) -> list[str]:
    """Collect labeled metadata lines immediately above a table block."""
    meta_lines: list[str] = []
    j = tbl_start - 1

    while j >= 0 and not lines[j].strip():
        j -= 1

    while j >= 0:
        line = lines[j].strip()
        if not line:
            break
        if HEADING_RE.match(line) or TABLE_ROW_RE.match(line):
            break
        if TABLE_META_LABEL_RE.match(line):
            meta_lines.append(lines[j])
            j -= 1
            continue
        # Allow continuation lines for long metadata values.
        if meta_lines and not TABLE_META_LABEL_RE.match(line):
            meta_lines.append(lines[j])
            j -= 1
            continue
        break

    return list(reversed(meta_lines))


def extract_tables(markdown: str, source_file: str) -> list[Chunk]:
    """
    Extract every markdown table as a single atomic chunk.
    Stores both the markdown text and a JSON row array.
    Captures the table title (heading immediately before) and
    any notes paragraph that immediately follows.
    """
    lines = markdown.splitlines()
    chunks: list[Chunk] = []
    in_fence: str | None = None
    i = 0

    while i < len(lines):
        in_fence = _toggle_fence(in_fence, lines[i])
        if in_fence:
            i += 1
            continue

        if TABLE_ROW_RE.match(lines[i]):
            # ── collect table lines ───────────────────────────────────────────
            tbl_start = i
            tbl_lines: list[str] = []
            while i < len(lines) and (TABLE_ROW_RE.match(lines[i]) or not lines[i].strip()):
                if lines[i].strip():
                    tbl_lines.append(lines[i])
                i += 1

            # Merge immediately adjacent table blocks when one block looks like
            # table metadata/schema and the other is the actual table body.
            while True:
                probe = i
                while probe < len(lines) and not lines[probe].strip():
                    probe += 1
                if probe >= len(lines) or not TABLE_ROW_RE.match(lines[probe]):
                    break

                next_tbl_lines: list[str] = []
                cursor = probe
                while cursor < len(lines) and (TABLE_ROW_RE.match(lines[cursor]) or not lines[cursor].strip()):
                    if lines[cursor].strip():
                        next_tbl_lines.append(lines[cursor])
                    cursor += 1

                if not (_is_metadata_table_block(tbl_lines) or _is_metadata_table_block(next_tbl_lines)):
                    break

                tbl_lines.extend([""] + next_tbl_lines)
                i = cursor

            if not tbl_lines:
                continue

            pre_table_meta = _collect_pre_table_metadata(lines, tbl_start)

            # ── find title: nearest heading above the table ───────────────────
            title = _current_heading(lines, tbl_start)

            # ── also grab the RAG-META block if it precedes the table ─────────
            rag_meta = ""
            window = "\n".join(lines[max(0, tbl_start - 10): tbl_start])
            for m in RAG_META_RE.finditer(window):
                rag_meta = m.group(0)

            # ── capture notes paragraph immediately after table ───────────────
            notes = ""
            j = i
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and not HEADING_RE.match(lines[j]) and not TABLE_ROW_RE.match(lines[j]):
                notes_lines = []
                while j < len(lines) and lines[j].strip() and not HEADING_RE.match(lines[j]):
                    notes_lines.append(lines[j])
                    j += 1
                notes = " ".join(notes_lines).strip()

            # ── build chunk text ──────────────────────────────────────────────
            md_table = "\n".join(tbl_lines)

            parts = []
            if rag_meta:
                parts.append(rag_meta)
            if pre_table_meta:
                parts.append("\n".join(pre_table_meta))
            if title:
                parts.append(f"**Table: {title}**")
            parts.append(md_table)
            if notes:
                parts.append(f"Notes: {notes}")

            text = "\n".join(parts).strip()
            cid = make_id(text, source_file)

            chunks.append(Chunk(
                chunk_id=cid,
                text=text,
                content_type="table",
                source_file=source_file,
            ))
        else:
            i += 1

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Text extractor (Strategy 2 text pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_chunks(markdown: str, source_file: str) -> list[Chunk]:
    """
    Split prose sections using LangChain RecursiveCharacterTextSplitter.
    Equations and tables are first masked so they are not split.
    Target: 300–500 tokens (1200–2000 chars). Overlap: ~1–2 sentences (~150 chars).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chars(512),      # 2048 chars ≈ 512 tokens
        chunk_overlap=300,          # ~50 tokens
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    # ── mask equations and tables so they are skipped ────────────────────────
    masked = markdown
    placeholders: dict[str, str] = {}

    def _placeholder(content: str) -> str:
        key = f"__MASKED_{uuid.uuid4().hex[:8]}__"
        placeholders[key] = content
        return key

    # mask display equations
    masked = DISPLAY_EQ_RE.sub(lambda m: _placeholder(m.group(0)), masked)
    # mask RAG-META blocks
    masked = RAG_META_RE.sub(lambda m: _placeholder(m.group(0)), masked)
    # mask full table blocks
    masked = re.sub(
        r"(?:^\|.*\n)+",
        lambda m: _placeholder(m.group(0)),
        masked,
        flags=re.MULTILINE,
    )
    # mask table metadata prose blocks so they stay with table chunks
    masked = re.sub(
        r"(?:^\s*\*\*(?:Table\s+subject|Property\s+type|Columns\s+and\s+units|Domain\s+keywords)\s*:\*\*.*\n?){2,}",
        lambda m: _placeholder(m.group(0)),
        masked,
        flags=re.MULTILINE,
    )

    # ── split masked text ─────────────────────────────────────────────────────
    raw_splits = splitter.split_text(masked)

    chunks: list[Chunk] = []
    lines = markdown.splitlines()

    for split in raw_splits:
        # skip if the split is only a placeholder
        stripped = split.strip()
        if not stripped:
            continue
        if stripped in placeholders or re.fullmatch(r"__MASKED_[0-9a-f]+__", stripped):
            continue

        # remove any embedded placeholders left after splitting
        for key in placeholders:
            split = split.replace(key, "")
        split = re.sub(r"\n{3,}", "\n\n", split).strip()
        if not split:
            continue

        # approximate section heading for this split
        # find where this text appears in the original
        approx_pos = masked.find(split[:60].strip())
        line_no = masked[:max(0, approx_pos)].count("\n") if approx_pos >= 0 else 0
        cid = make_id(split, source_file)
        chunks.append(Chunk(
            chunk_id=cid,
            text=split,
            content_type="text",
            source_file=source_file,
        ))

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Content-type aware chunking
# ─────────────────────────────────────────────────────────────────────────────

def run_strategy2(md_files: list[Path], output_dir: Path) -> None:
    """
    Run three separate pipelines (text / equation / table) and write:
      strategy2_text.jsonl
      strategy2_equations.jsonl
      strategy2_tables.jsonl
            strategy2_all.jsonl      ← merged, used for the vector index
    """
    print("\n" + "═" * 60)
    print("STRATEGY 2 — Content-type aware chunking")
    print("═" * 60)

    all_text:  list[Chunk] = []
    all_eq:    list[Chunk] = []
    all_table: list[Chunk] = []

    for md_file in md_files:
        print(f"  Processing: {md_file.name}", flush=True)
        markdown = md_file.read_text(encoding="utf-8", errors="replace")

        text_chunks  = extract_text_chunks(markdown, md_file.name)
        eq_chunks    = extract_equations(markdown, md_file.name)
        table_chunks = extract_tables(markdown, md_file.name)

        all_text.extend(text_chunks)
        all_eq.extend(eq_chunks)
        all_table.extend(table_chunks)

        print(f"    text={len(text_chunks)}, equations={len(eq_chunks)}, tables={len(table_chunks)}")

    all_chunks = all_text + all_eq + all_table

    # ── deduplicate by chunk_id ───────────────────────────────────────────────
    seen: set[str] = set()
    deduped: list[Chunk] = []
    for c in all_chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            deduped.append(c)

    # ── write outputs ─────────────────────────────────────────────────────────
    _write_jsonl(output_dir / "strategy2_text.jsonl",      all_text)
    _write_jsonl(output_dir / "strategy2_equations.jsonl", all_eq)
    _write_jsonl(output_dir / "strategy2_tables.jsonl",    all_table)
    _write_jsonl(output_dir / "strategy2_all.jsonl",       deduped)

    print(f"\n  ✅ Strategy 2 complete:")
    print(f"     text chunks    : {len(all_text)}")
    print(f"     equation chunks: {len(all_eq)}")
    print(f"     table chunks   : {len(all_table)}")
    print(f"     total (deduped): {len(deduped)}")
    print(f"     output dir     : {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_to_dict(c: Chunk) -> dict:
    d = asdict(c)
    # remove child_ids list from JSONL to keep it flat (keep in JSON only)
    return d


def _write_jsonl(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(_chunk_to_dict(c), ensure_ascii=False) + "\n")
    print(f"  → wrote {len(chunks):,} chunks to {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chunk markdown files into RAG-ready JSONL/JSON using Strategy 2.",
    )
    parser.add_argument(
        "--input", "-i", default=str(DEFAULT_INPUT),
        help="Folder containing input .md files.",
    )
    parser.add_argument(
        "--output", "-o", default=str(DEFAULT_OUTPUT),
        help="Folder where chunked output files are written.",
    )
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: input folder not found: {input_dir}")
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(input_dir.rglob("*.md"))
    if not md_files:
        print(f"No .md files found in {input_dir}")
        return 1

    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    print(f"Files  : {len(md_files)} markdown files found")

    run_strategy2(md_files, output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())