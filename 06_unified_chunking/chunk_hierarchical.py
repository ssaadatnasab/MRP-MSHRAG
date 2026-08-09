#!/usr/bin/env python3
"""
chunk_strategy1_langchain.py
Strategy 1 — Hierarchical Section Chunking
Tool: LangChain MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter

What this script does
─────────────────────
1. For every .md file in INPUT_DIR it:
   a. Protects block equations ($$…$$) + their variable-definition paragraphs
   b. Protects markdown tables + their title lines
      → Both become single-line placeholders so LangChain splitters never
        split inside them  (equations and tables are always atomic)
   c. Runs MarkdownHeaderTextSplitter on H1–H4 boundaries
      → all ancestor headings auto-injected as metadata on every chunk
   d. Runs RecursiveCharacterTextSplitter on sections > MAX_TOKENS
      → protected placeholders are never broken apart
   e. Restores all placeholders; classifies and annotates each chunk

2. Writes per-run:
   OUTPUT_DIR/strategy1_langchain/
       chunks.jsonl   — one JSON object per line  (for streaming / large loads)
       summary.json   — run statistics

Output schema (per chunk)
──────────────────────────
{
  "chunk_id":       "ashrae_62_1_2022__0042",
  "source_file":    "ASHRAE_62.1-2022.md",
  "text":           "…",
  "heading_path":   "6 Requirements > 6.2 Ventilation Rate Procedure > …",
  "section":        "6.2.1.1",
  "parent_section": "6.2.1",
  "headings":       {"h1": "…", "h2": "…", "h3": "…", "h4": "…"},
  "content_type":   "text" | "equation" | "table" | "mixed",
  "has_equation":   false,
  "eq_ids":         [],
  "has_table":      false,
  "table_ids":      [],
  "token_count":    287,
  "char_count":     1456,
  "chunk_index":    42
}

Install
───────
  pip install langchain-text-splitters tiktoken

Usage
─────
  python chunk_strategy1_langchain.py
  python chunk_strategy1_langchain.py --input-dir /path/md --output-dir /path/out
  python chunk_strategy1_langchain.py --max-tokens 400 --overlap 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ── optional tiktoken (falls back to char/4 estimate) ────────────────────────
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)

try:
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )
except ImportError as exc:  # pragma: no cover - user-facing guidance
    raise SystemExit(
        "Missing dependency: install it with `pip install langchain-text-splitters` and `pip install tiktoken`."
    ) from exc

# ─────────────────────────────────────────────────────────────────────────────
# Defaults  (override via CLI or edit here)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_INPUT = os.getenv("MARKDOWN_INPUT_DIR", "./input_markdown")
DEFAULT_OUTPUT = os.getenv("MARKDOWN_OUTPUT_DIR", "./output/hierarchical_langchain")
MAX_TOKENS     = 512   # hard ceiling per final chunk
OVERLAP_TOKENS = 80    # token overlap when a section is split further

# ─────────────────────────────────────────────────────────────────────────────
# Compiled patterns
# ─────────────────────────────────────────────────────────────────────────────
SECTION_NUM_RE  = re.compile(r"^(\d+(?:\.\d+)*)\s*")          # "6.2.1.1 Title"
EQ_ID_RE        = re.compile(                                  # \tag{6-1} or (6-1) near $$
    r"\\tag\{([^}]+)\}|"
    r"\((\d+[\-\.]\d+)\)"
)
TABLE_ID_RE     = re.compile(                                  # "Table 6-1" or "Table A-1"
    r"[Tt]able\s+([\w\-]+)",
    re.IGNORECASE,
)
# Placeholder tokens — use characters unlikely to appear in engineering text
_PH_TABLE = "__PROTECTED_TABLE_{idx}__"
_PH_EQ    = "__PROTECTED_EQ_{idx}__"
_PH_RE    = re.compile(r"__PROTECTED_(TABLE|EQ)_(\d+)__")

# ─────────────────────────────────────────────────────────────────────────────
# Block protector — makes equations and tables atomic before splitting
# ─────────────────────────────────────────────────────────────────────────────

class BlockProtector:
    """Replace tables and block-equations with single-line placeholders.

    This ensures LangChain's splitters never produce a chunk boundary
    inside a table row or inside an equation/variable-definition block.

    After splitting, call ``restore(text)`` to put original content back.
    """

    def __init__(self, text: str) -> None:
        self._blocks: dict[str, str] = {}   # placeholder → original content
        self._protected: str = self._protect(text)

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def text(self) -> str:
        """Return the text with all protected blocks replaced by placeholders."""
        return self._protected

    def restore(self, text: str) -> str:
        """Substitute every placeholder back with its original content."""
        def _repl(m: re.Match) -> str:
            key = m.group(0)
            return self._blocks.get(key, key)
        return _PH_RE.sub(_repl, text)

    # ── internal ──────────────────────────────────────────────────────────────

    def _protect(self, text: str) -> str:
        lines   = text.split("\n")
        t_idx   = 0   # table counter
        eq_idx  = 0   # equation counter
        result  : list[str] = []
        skip_to : int = -1   # skip lines already consumed

        i = 0
        while i < len(lines):
            if i <= skip_to:
                i += 1
                continue

            line = lines[i]

            # ── block equation: line is exactly "$$" or "$$expr$$" on one line ─
            stripped = line.strip()
            if stripped == "$$" or stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
                start, end = self._scan_equation(lines, i)
                block = "\n".join(lines[start : end + 1])
                ph = _PH_EQ.format(idx=eq_idx)
                self._blocks[ph] = block
                result.append(ph)
                skip_to = end
                eq_idx += 1
                i = end + 1
                continue

            # ── markdown table: line starts with "|" ──────────────────────────
            if stripped.startswith("|"):
                title_start, end = self._scan_table(lines, i)
                block = "\n".join(lines[title_start : end + 1])
                ph = _PH_TABLE.format(idx=t_idx)
                self._blocks[ph] = block
                result.append(ph)
                skip_to = end
                t_idx += 1
                i = end + 1
                continue

            result.append(line)
            i += 1

        return "\n".join(result)

    @staticmethod
    def _scan_equation(lines: list[str], start: int) -> tuple[int, int]:
        """Return (start, end) of block equation + following variable defs."""
        i = start
        stripped = lines[i].strip()

        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            end = i   # single-line $$expr$$
        else:
            # multi-line: scan for closing $$
            i += 1
            while i < len(lines) and "$$" not in lines[i]:
                i += 1
            end = i  # line containing closing $$

        # Grab variable definitions: "where …" block immediately following
        j = end + 1
        # skip exactly one blank line
        if j < len(lines) and not lines[j].strip():
            j += 1
        # collect until blank line or next heading / equation
        if j < len(lines):
            first = lines[j].strip().lower()
            if first.startswith("where") or first.startswith("-") or first.startswith("*"):
                while j < len(lines):
                    l = lines[j].strip()
                    if not l or l.startswith("#") or l.startswith("$$"):
                        break
                    j += 1
                end = j - 1

        return start, end

    @staticmethod
    def _scan_table(lines: list[str], start: int) -> tuple[int, int]:
        """Return (title_start, end) for a table block including its title."""
        # Look back one line for a table title
        title_start = start
        if start > 0:
            prev = lines[start - 1].strip()
            if prev and not prev.startswith("|") and (
                re.search(r"\b[Tt]able\b", prev) or
                prev.startswith("**") or
                prev.startswith("_")
            ):
                title_start = start - 1

        # Scan forward: collect all consecutive table rows
        # (allow one blank line between rows — some converters insert them)
        i = start
        while i < len(lines):
            l = lines[i].strip()
            if l.startswith("|"):
                i += 1
            elif not l and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
                i += 1  # blank line between rows — keep going
            else:
                break

        return title_start, i - 1


# ─────────────────────────────────────────────────────────────────────────────
# Metadata helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_section_number(heading_text: str) -> Optional[str]:
    """Extract dotted section number, e.g. '6.2.1.1' from '6.2.1.1 Breathing Zone'."""
    if not heading_text:
        return None
    m = SECTION_NUM_RE.match(heading_text.strip().lstrip("*").strip())
    return m.group(1) if m else None


def get_parent_section(section: Optional[str]) -> Optional[str]:
    """'6.2.1.1' → '6.2.1'.  Returns None for top-level sections."""
    if not section or "." not in section:
        return None
    return section.rsplit(".", 1)[0]


def extract_eq_ids(text: str) -> list[str]:
    """Extract equation IDs like '6-1' from \\tag{6-1} or (6-1) near $$."""
    ids = []
    for m in EQ_ID_RE.finditer(text):
        val = m.group(1) or m.group(2)
        if val:
            ids.append(val)
    return list(dict.fromkeys(ids))   # deduplicate, preserve order


def extract_table_ids(text: str) -> list[str]:
    """Extract table IDs like '6-1' from 'Table 6-1'."""
    return list(dict.fromkeys(
        m.group(1) for m in TABLE_ID_RE.finditer(text)
    ))


def classify_content_type(text: str) -> str:
    """Classify chunk as text / equation / table / mixed."""
    has_eq    = "$$" in text or bool(re.search(r"\$[^$\n]+\$", text))
    has_table = bool(re.search(r"^\|", text, re.MULTILINE))
    if has_eq and has_table:
        return "mixed"
    if has_eq:
        return "equation"
    if has_table:
        return "table"
    return "text"


def has_variables(text: str) -> bool:
    has_eq = "$$" in text or bool(re.search(r"\$[^$\n]+\$", text))
    if not has_eq:
        return False
    if re.search(r"^\s*where\b", text, re.IGNORECASE | re.MULTILINE):
        return True
    return bool(re.search(r"^\s*[-*]?\s*[A-Za-z][A-Za-z0-9_]*\s*(=|:)\s*\S+", text, re.MULTILINE))


def build_headings_dict(lc_metadata: dict) -> dict[str, str]:
    """Convert LangChain {'Header 1': '…', 'Header 2': '…'} to {'h1': '…', 'h2': '…'}."""
    mapping = {"Header 1": "h1", "Header 2": "h2", "Header 3": "h3", "Header 4": "h4"}
    return {mapping[k]: v for k, v in lc_metadata.items() if k in mapping}


def build_heading_path(headings: dict) -> str:
    """Build a human-readable breadcrumb: 'H1 > H2 > H3 > H4'."""
    parts = [headings.get(f"h{i}") for i in range(1, 5)]
    return " > ".join(p for p in parts if p)


def _base36(num: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if num == 0:
        return "0"
    out: list[str] = []
    while num > 0:
        num, rem = divmod(num, 36)
        out.append(digits[rem])
    return "".join(reversed(out))


def make_chunk_id(source_file: str, chunk_index: int) -> str:
    file_tag = hashlib.md5(source_file.encode("utf-8")).hexdigest()[:4]
    return f"c{file_tag}{_base36(chunk_index)}"


def make_chunk(
    text: str,
    lc_metadata: dict,
    source_file: str,
    chunk_index: int,
    stem: str,
) -> dict:
    """Assemble the final chunk dict from restored text + LangChain metadata."""
    headings    = build_headings_dict(lc_metadata)
    path        = build_heading_path(headings)

    # Section number: use deepest heading that has one
    section = None
    for level in ("h4", "h3", "h2", "h1"):
        h = headings.get(level, "")
        section = extract_section_number(h)
        if section:
            break

    return {
        "chunk_id"      : make_chunk_id(source_file, chunk_index),
        "source_file"   : source_file,
        "text"          : text.strip(),
        "heading_path"  : path,
        "section"       : section,
        "parent_section": get_parent_section(section),
        "headings"      : headings,
        "has_variables" : has_variables(text),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LangChain splitter configuration
# ─────────────────────────────────────────────────────────────────────────────

HEADERS_TO_SPLIT = [
    ("#",    "Header 1"),
    ("##",   "Header 2"),
    ("###",  "Header 3"),
    ("####", "Header 4"),
]


def build_splitters(max_tokens: int, overlap: int):
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,   # keep heading text inside the chunk
    )
    # Approximate chars from tokens (cl100k averages ~4 chars/token)
    overflow_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens * 4,
        chunk_overlap=overlap * 4,
        length_function=count_tokens,   # use actual token counter
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return header_splitter, overflow_splitter


# ─────────────────────────────────────────────────────────────────────────────
# Per-file chunking pipeline
# ─────────────────────────────────────────────────────────────────────────────

def chunk_markdown_file(
    md_path: Path,
    max_tokens: int,
    overlap: int,
) -> list[dict]:
    """Run the full pipeline on one markdown file and return chunk dicts."""
    text = md_path.read_text(encoding="utf-8", errors="replace")
    stem = re.sub(r"[^a-z0-9]+", "_", md_path.stem.lower()).strip("_")

    # ── Step 1: protect atomic blocks ────────────────────────────────────────
    protector = BlockProtector(text)
    protected_text = protector.text

    # ── Step 2: split at heading boundaries ──────────────────────────────────
    header_splitter, overflow_splitter = build_splitters(max_tokens, overlap)
    header_docs = header_splitter.split_text(protected_text)

    # ── Step 3: overflow-split oversized sections ─────────────────────────────
    raw_chunks: list[tuple[str, dict]] = []   # (content, lc_metadata)

    for doc in header_docs:
        content  = doc.page_content
        metadata = doc.metadata

        if count_tokens(content) > max_tokens:
            # Further split — but only if no protected block sits inside
            # (protected placeholders are one line, so the splitter won't
            # break them apart — they will land in exactly one sub-chunk)
            sub_docs = overflow_splitter.split_text(content)
            for sd in sub_docs:
                if sd.strip():
                    raw_chunks.append((sd, metadata))
        else:
            if content.strip():
                raw_chunks.append((content, metadata))

    # ── Step 4: restore protected content + build metadata ───────────────────
    chunks: list[dict] = []
    for idx, (content, lc_meta) in enumerate(raw_chunks):
        restored = protector.restore(content)
        if not restored.strip():
            continue
        chunk = make_chunk(
            text=restored,
            lc_metadata=lc_meta,
            source_file=md_path.name,
            chunk_index=idx,
            stem=stem,
        )
        chunks.append(chunk)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def write_outputs(chunks: list[dict], output_dir: Path) -> None:
    out = output_dir / "strategy1_langchain"
    out.mkdir(parents=True, exist_ok=True)

    # JSONL — one chunk per line
    jsonl_path = out / "chunks.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Summary
    content_types   = {}
    files_seen      : set[str] = set()
    total_tokens    = 0
    for c in chunks:
        ct = classify_content_type(c["text"])
        content_types[ct] = content_types.get(ct, 0) + 1
        files_seen.add(c["source_file"])
        total_tokens += count_tokens(c["text"])

    summary = {
        "total_chunks"          : len(chunks),
        "source_files"          : len(files_seen),
        "chunks_by_content_type": content_types,
        "total_tokens"          : total_tokens,
        "avg_tokens_per_chunk"  : round(total_tokens / max(1, len(chunks)), 1),
        "output_jsonl"          : str(jsonl_path),
    }
    summary_path = out / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n{'=' * 55}")
    print(f"Strategy 1 — LangChain — output written to: {out}")
    print(f"  Total chunks : {summary['total_chunks']}")
    print(f"  Source files : {summary['source_files']}")
    print(f"  Content types: {content_types}")
    print(f"  Avg tokens   : {summary['avg_tokens_per_chunk']}")
    print(f"  JSONL        : {jsonl_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strategy 1 — Hierarchical Section Chunking (LangChain)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input-dir",  default=DEFAULT_INPUT,
                        help="Folder containing .md files (defaults to MARKDOWN_INPUT_DIR or ./input_markdown).")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                        help="Folder to write chunk outputs (defaults to MARKDOWN_OUTPUT_DIR or ./output/hierarchical_langchain).")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help=f"Token ceiling per chunk (default {MAX_TOKENS}).")
    parser.add_argument("--overlap",    type=int, default=OVERLAP_TOKENS,
                        help=f"Token overlap for oversized splits (default {OVERLAP_TOKENS}).")
    parser.add_argument("--pattern",    default="*.md",
                        help="Glob pattern for input files (default: *.md).")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: input-dir not found: {input_dir}", file=sys.stderr)
        return 2

    md_files = sorted(input_dir.glob(args.pattern))
    if not md_files:
        md_files = sorted(input_dir.rglob("*.md"))
    if not md_files:
        print(f"ERROR: no .md files found under {input_dir}", file=sys.stderr)
        return 2

    print(f"Found {len(md_files)} markdown file(s) in {input_dir}")
    print(f"Settings: max_tokens={args.max_tokens}, overlap={args.overlap}")

    all_chunks: list[dict] = []
    for i, path in enumerate(md_files, 1):
        print(f"  [{i}/{len(md_files)}] {path.name} ...", end=" ", flush=True)
        chunks = chunk_markdown_file(path, args.max_tokens, args.overlap)
        print(f"{len(chunks)} chunks")
        all_chunks.extend(chunks)

    write_outputs(all_chunks, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())