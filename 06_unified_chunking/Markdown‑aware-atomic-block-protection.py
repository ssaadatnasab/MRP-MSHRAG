#!/usr/bin/env python3
"""

====================================

Loads .md files directly while protecting:
  - Block equations  ($$...$$)  → never split mid-formula
  - Markdown tables  (|...|)    → never split mid-table, header repeated on continuation

Pipeline:
    1. DirectoryLoader / markdown file reader — loads markdown documents
    2. MarkdownAwareTextSplitter             — respects headings, equations, tables
    3. LangChain Documents                   — ready to hand off to your RAG pipeline

Usage:
        from markdown_direct_loader import load_markdown_corpus

        docs = load_markdown_corpus(
        corpus_path="/path/to/markdown/folder",
    )
        # pass docs into your existing RAG pipeline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as exc:  # pragma: no cover - user-facing guidance
    raise SystemExit(
        "Missing dependency: install it with `pip install langchain-text-splitters`."
    ) from exc

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CHARS_PER_TOKEN  = 4
PARENT_TOKENS    = 512          # 2048 chars — fills all-MiniLM-L6-v2's 512-token window
PARENT_CHARS     = PARENT_TOKENS * CHARS_PER_TOKEN   # 2048
OVERLAP_TOKENS   = 100          # 400 chars — 20% overlap
OVERLAP_CHARS    = OVERLAP_TOKENS * CHARS_PER_TOKEN  # 400

# Regex patterns
FENCE_RE         = re.compile(r"^\s*(`{3,}|~{3,})")
HEADING_RE       = re.compile(r"^(#{1,6})\s+(.*)")
TABLE_ROW_RE     = re.compile(r"^\s*\|")
TABLE_SEP_RE     = re.compile(r"^\s*\|[\s|:-]+\|\s*$")
DISPLAY_EQ_RE    = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
RAG_META_RE      = re.compile(r"<!--\s*RAG-META.*?-->", re.DOTALL)

# Placeholder tokens used during splitting
_PH_PREFIX       = "__ATOMIC_"
_PH_RE           = re.compile(r"__ATOMIC_[A-Z]+_[0-9a-f]{8}__")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Atomic Block Protector
#
# Replaces equations, tables, and RAG-META blocks with single-line
# placeholder tokens so the text splitter never cuts inside them.
# After splitting, call restore() to put the original content back.
# ─────────────────────────────────────────────────────────────────────────────

class AtomicBlockProtector:
    """
    Scans markdown line by line and replaces atomic blocks with placeholders.

    Rules:
      - Block equation: lines between $$ ... $$ delimiters (inclusive),
        plus any immediately following "where:" variable-definition paragraph.
      - Markdown table: all consecutive pipe-rows, plus optional title line
        immediately above and optional notes paragraph immediately below.

        Both become a single __ATOMIC_TYPE_xxxxxxxx__ token.
    The splitter sees them as one indivisible word.
    """

    def __init__(self, text: str) -> None:
        self._store: dict[str, str] = {}   # placeholder → original content
        self.protected = self._run(text)

    def restore(self, text: str) -> str:
        """Replace every placeholder back with its original content."""
        return _PH_RE.sub(lambda m: self._store.get(m.group(0), m.group(0)), text)

    def _ph(self, kind: str, content: str) -> str:
        key = f"{_PH_PREFIX}{kind}_{uuid.uuid4().hex[:8]}__"
        self._store[key] = content
        return key

    def _run(self, text: str) -> str:
        lines  = text.splitlines()
        result : list[str] = []
        i      = 0
        in_fence: Optional[str] = None

        while i < len(lines):
            line     = lines[i]
            stripped = line.strip()

            # ── track fenced code blocks — never touch content inside them ──
            fm = FENCE_RE.match(line)
            if fm:
                fence_char = fm.group(1)[0]
                if in_fence is None:
                    in_fence = fence_char
                elif fence_char == in_fence:
                    in_fence = None
                result.append(line)
                i += 1
                continue

            if in_fence:
                result.append(line)
                i += 1
                continue

            # ── RAG-META HTML comment ────────────────────────────────────────
            if stripped.startswith("<!--") and "RAG-META" in stripped:
                block_lines = [line]
                i += 1
                while i < len(lines) and "-->" not in lines[i - 1]:
                    block_lines.append(lines[i])
                    i += 1
                result.append(self._ph("META", "\n".join(block_lines)))
                continue

            # ── Block equation: $$ ... $$ ─────────────────────────────────
            if stripped.startswith("$$"):
                eq_lines = [line]
                # single-line $$expr$$
                if stripped.endswith("$$") and len(stripped) > 4:
                    i += 1
                else:
                    i += 1
                    while i < len(lines) and "$$" not in lines[i]:
                        eq_lines.append(lines[i])
                        i += 1
                    if i < len(lines):
                        eq_lines.append(lines[i])
                        i += 1

                # grab "where:" variable-definition paragraph immediately after
                j = i
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                if j < len(lines) and re.match(r"^\s*where\b", lines[j], re.I):
                    eq_lines.append("")   # blank separator
                    j += 1
                    while j < len(lines) and lines[j].strip() and not HEADING_RE.match(lines[j]):
                        eq_lines.append(lines[j])
                        j += 1
                    i = j

                result.append(self._ph("EQ", "\n".join(eq_lines)))
                continue

            # ── Markdown table ────────────────────────────────────────────
            if TABLE_ROW_RE.match(line):
                tbl_lines = []

                # include optional title line immediately above
                if result and result[-1].strip() and not result[-1].strip().startswith("|"):
                    prev = result.pop()
                    tbl_lines.append(prev)

                # collect all pipe rows (allow one blank line between rows)
                while i < len(lines):
                    if TABLE_ROW_RE.match(lines[i]):
                        tbl_lines.append(lines[i])
                        i += 1
                    elif (not lines[i].strip()
                          and i + 1 < len(lines)
                          and TABLE_ROW_RE.match(lines[i + 1])):
                        i += 1   # skip single blank between rows
                    else:
                        break

                # include optional notes paragraph immediately after
                j = i
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                if (j < len(lines)
                        and not HEADING_RE.match(lines[j])
                        and not TABLE_ROW_RE.match(lines[j])
                        and not lines[j].strip().startswith("$$")):
                    notes: list[str] = []
                    while j < len(lines) and lines[j].strip() and not HEADING_RE.match(lines[j]):
                        notes.append(lines[j])
                        j += 1
                    tbl_lines.append("")
                    tbl_lines.extend(notes)
                    i = j

                result.append(self._ph("TABLE", "\n".join(tbl_lines)))
                continue

            result.append(line)
            i += 1

        return "\n".join(result)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Markdown-Aware Text Splitter
#
# Uses LangChain's RecursiveCharacterTextSplitter on the *protected* text,
# then restores atomic blocks.  Because placeholders are single tokens with no
# whitespace, the splitter never breaks them.
# ─────────────────────────────────────────────────────────────────────────────

def _split_markdown_file(
    md_path: Path,
    chunk_size: int  = PARENT_CHARS,
    overlap: int     = OVERLAP_CHARS,
) -> list[dict]:
    """
    Protect atomic blocks, split, restore, return chunk dicts.

    Returns list of dicts with keys:
        text, source_file, section_heading, content_type, token_estimate
    """
    text = md_path.read_text(encoding="utf-8", errors="replace")

    # ── protect atomic blocks ────────────────────────────────────────────────
    protector = AtomicBlockProtector(text)
    protected = protector.protected

    # ── split on paragraph / sentence / word boundaries ──────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    raw_splits = splitter.split_text(protected)

    # ── restore and annotate ─────────────────────────────────────────────────
    lines = text.splitlines()
    chunks: list[dict] = []

    for split in raw_splits:
        restored = protector.restore(split).strip()
        if not restored:
            continue

        # skip chunks that are only a placeholder (equation/table with no prose)
        # — these will be captured as their own restored chunk naturally
        if _PH_RE.fullmatch(restored.strip()):
            continue

        # approximate section heading
        approx_pos = protected.find(split[:60].strip())
        line_no    = protected[:max(0, approx_pos)].count("\n") if approx_pos >= 0 else 0
        heading    = _heading_at(lines, line_no)

        content_type = _classify(restored)
        token_est    = max(1, len(restored) // CHARS_PER_TOKEN)

        chunks.append({
            "text"           : restored,
            "source_file"    : md_path.name,
            "section_heading": heading,
            "content_type"   : content_type,
            "has_variables"  : _has_variables(restored),
            "token_estimate" : token_est,
        })

    return chunks


def _split_markdown_text(
    text: str,
    source_name: str,
    chunk_size: int = PARENT_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> list[dict]:
    """Split a markdown string into protected chunks."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    protector = AtomicBlockProtector(text)
    protected = protector.protected

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )
    raw_splits = splitter.split_text(protected)

    lines = text.splitlines()
    chunks: list[dict] = []
    for split in raw_splits:
        restored = protector.restore(split).strip()
        if not restored or _PH_RE.fullmatch(restored.strip()):
            continue

        approx_pos = protected.find(split[:60].strip())
        line_no = protected[:max(0, approx_pos)].count("\n") if approx_pos >= 0 else 0
        heading = _heading_at(lines, line_no)

        chunks.append({
            "text": restored,
            "source_file": source_name,
            "section_heading": heading,
            "content_type": _classify(restored),
            "has_variables": _has_variables(restored),
            "token_estimate": max(1, len(restored) // CHARS_PER_TOKEN),
        })

    return chunks


def _heading_at(lines: list[str], up_to: int) -> str:
    best = ""
    for i in range(min(up_to, len(lines))):
        m = HEADING_RE.match(lines[i])
        if m:
            best = m.group(2).strip()
    return best


def _classify(text: str) -> str:
    has_eq    = "$$" in text or bool(re.search(r"\$[^$\n]{2,200}\$", text))
    has_table = bool(re.search(r"^\|", text, re.MULTILINE))
    if has_eq and has_table:
        return "mixed"
    if has_eq:
        return "equation"
    if has_table:
        return "table"
    return "text"


def _has_variables(text: str) -> bool:
    if not ("$$" in text or bool(re.search(r"\$[^$\n]{2,200}\$", text))):
        return False
    if re.search(r"^\s*where\b", text, re.IGNORECASE | re.MULTILINE):
        return True
    return bool(re.search(r"^\s*[-*]?\s*[A-Za-z][A-Za-z0-9_]*\s*(=|:)\s*\S+", text, re.MULTILINE))


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Load all .md files from a folder
# ─────────────────────────────────────────────────────────────────────────────

def load_markdown_corpus(
    corpus_path: str,
    chunk_size : int = PARENT_CHARS,
    overlap    : int = OVERLAP_CHARS,
) -> list:
    """
    Walk corpus_path for *.md files, split each with atomic-block protection,
    and return a list of LangChain Document objects ready for embedding.
    """
    from langchain_core.documents import Document

    corpus_dir = Path(corpus_path)
    if not corpus_dir.exists():
        raise ValueError(f"Corpus folder not found: {corpus_path}")

    docs: list[Document] = []

    try:
        from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader

        loader = DirectoryLoader(
            str(corpus_dir),
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            show_progress=True,
            use_multithreading=True,
        )
        source_docs = loader.load()
        print(f"📂 Loaded {len(source_docs)} markdown documents with DirectoryLoader", flush=True)
    except Exception:
        md_files = sorted(corpus_dir.rglob("*.md"))
        if not md_files:
            raise ValueError(f"No .md files found in {corpus_path}")
        print(f"📂 Found {len(md_files)} markdown files in {corpus_path}", flush=True)
        source_docs = []
        for md_path in md_files:
            source_docs.append(
                Document(
                    page_content=md_path.read_text(encoding="utf-8", errors="replace"),
                    metadata={"source": str(md_path)},
                )
            )

    for source_doc in source_docs:
        source_name = Path(str(source_doc.metadata.get("source", "unknown"))).name
        chunks = _split_markdown_text(source_doc.page_content, source_name, chunk_size, overlap)
        for c in chunks:
            docs.append(Document(
                page_content=c["text"],
                metadata={
                    "source"          : c["source_file"],
                    "section_heading" : c["section_heading"],
                    "content_type"    : c["content_type"],
                    "has_variables"   : c["has_variables"],
                    "token_estimate"  : c["token_estimate"],
                },
            ))
        print(f"  {source_name}: {len(chunks)} chunks", flush=True)

    print(f"✅ Total chunks loaded: {len(docs)}", flush=True)
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Output writer / smoke-test
# ─────────────────────────────────────────────────────────────────────────────

def write_chunks_jsonl(docs, output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "direct_markdown_chunks.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps({
                "text": doc.page_content,
                **doc.metadata,
            }, ensure_ascii=False) + "\n")
    return jsonl_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct markdown loader smoke-test")
    parser.add_argument(
        "--input",
        default=os.getenv("MARKDOWN_INPUT_DIR", "./input_markdown"),
        help="Folder containing .md files (defaults to MARKDOWN_INPUT_DIR or ./input_markdown)",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("MARKDOWN_OUTPUT_DIR", "./output/markdown_aware"),
        help="Folder to write JSONL chunks (defaults to MARKDOWN_OUTPUT_DIR or ./output/markdown_aware)",
    )
    args = parser.parse_args()

    docs = load_markdown_corpus(args.input)
    jsonl_path = write_chunks_jsonl(docs, args.output)
    print(f"\n{'='*60}")
    print(f"Loaded {len(docs)} chunks from markdown corpus")
    print(f"Wrote JSONL to: {jsonl_path}")
    for i, doc in enumerate(docs[:5], 1):
        print(f"\n[{i}] {doc.metadata.get('source')} | section={doc.metadata.get('section_heading')}")
        print(doc.page_content[:300])