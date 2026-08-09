#!/usr/bin/env python3
"""
Markdown Chunker — Strategy 3
=============================

Strategy 3: Small-to-large (parent-child) chunking
  Small child chunks for retrieval, large parent chunks returned to LLM.
  Outputs: strategy3_children.jsonl, strategy3_parents.jsonl,
		   strategy3_children.jsonl and strategy3_parents.jsonl

Usage:
	python "Small-to-large (parent-child).py"
	python "Small-to-large (parent-child).py" --input /path/to/md
	python "Small-to-large (parent-child).py" --output /path/to/output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as exc:  # pragma: no cover - user-facing guidance
    raise SystemExit(
        "Missing dependency: install it with `pip install langchain-text-splitters`."
    ) from exc


DEFAULT_INPUT = Path(os.getenv("MARKDOWN_INPUT_DIR", "./input_markdown"))
DEFAULT_OUTPUT = Path(os.getenv("MARKDOWN_OUTPUT_DIR", "./output/parent_child_chunks"))

CHARS_PER_TOKEN = 4


def chars(token_count: int) -> int:
	return token_count * CHARS_PER_TOKEN


def tokens(text: str) -> int:
	return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class Chunk:
	chunk_id: str
	text: str
	source_file: str
	section_heading: str
	has_variables: bool = False
	parent_id: str | None = None
	chunk_role: str | None = None
	child_ids: list[str] = field(default_factory=list)


def make_id(text: str, source: str) -> str:
	h = hashlib.md5(f"{source}::{text[:120]}".encode()).hexdigest()[:12]
	return f"chunk_{h}"


FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
TABLE_ROW_RE = re.compile(r"^\s*\|")
TABLE_SEP_RE = re.compile(r"^\s*\|[\s|:-]+\|\s*$")
DISPLAY_EQ_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
RAG_META_RE = re.compile(r"<!--\s*RAG-META.*?-->", re.DOTALL)


def _toggle_fence(in_fence: str | None, line: str) -> str | None:
	m = FENCE_RE.match(line)
	if not m:
		return in_fence
	fence = m.group(1)
	return None if in_fence and fence[0] == in_fence[0] else fence


def _current_heading(lines: list[str], up_to: int) -> str:
	best = ""
	for i in range(up_to):
		m = HEADING_RE.match(lines[i])
		if m:
			best = m.group(2).strip()
	return best


def extract_equations(markdown: str, source_file: str) -> list[Chunk]:
	lines = markdown.splitlines()
	chunks: list[Chunk] = []
	in_fence: str | None = None
	i = 0

	while i < len(lines):
		in_fence = _toggle_fence(in_fence, lines[i])
		if in_fence:
			i += 1
			continue

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

			var_lines: list[str] = []
			j = i
			while j < len(lines) and lines[j].strip() == "":
				j += 1
			if j < len(lines) and re.match(r"^\s*where\b", lines[j], re.I):
				j += 1
				while j < len(lines) and lines[j].strip() and not HEADING_RE.match(lines[j]):
					var_lines.append(lines[j])
					j += 1
				i = j

			rag_meta = ""
			window_before = "\n".join(lines[max(0, eq_start - 15): eq_start])
			window_after = "\n".join(lines[i: i + 15])
			for m in RAG_META_RE.finditer(window_before + "\n" + window_after):
				rag_meta = m.group(0)
				break

			heading = _current_heading(lines, eq_start)

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
			chunks.append(
				Chunk(
					chunk_id=cid,
					text=text,
					source_file=source_file,
					section_heading=heading,
					has_variables=bool(var_lines),
				)
			)
		else:
			i += 1

	return chunks


def _table_to_json_rows(table_lines: list[str]) -> list[dict]:
	rows = [l for l in table_lines if TABLE_ROW_RE.match(l) and not TABLE_SEP_RE.match(l)]
	if not rows:
		return []

	def split_row(line: str) -> list[str]:
		return [c.strip() for c in line.strip().strip("|").split("|")]

	header = split_row(rows[0])
	result = []
	for data_row in rows[1:]:
		cells = split_row(data_row)
		cells = (cells + [""] * len(header))[: len(header)]
		result.append(dict(zip(header, cells)))
	return result


def extract_tables(markdown: str, source_file: str) -> list[Chunk]:
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
			tbl_start = i
			tbl_lines: list[str] = []
			while i < len(lines) and (TABLE_ROW_RE.match(lines[i]) or not lines[i].strip()):
				if lines[i].strip():
					tbl_lines.append(lines[i])
				i += 1

			if not tbl_lines:
				continue

			title = _current_heading(lines, tbl_start)

			rag_meta = ""
			window = "\n".join(lines[max(0, tbl_start - 10): tbl_start])
			for m in RAG_META_RE.finditer(window):
				rag_meta = m.group(0)

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

			md_table = "\n".join(tbl_lines)
			json_rows = _table_to_json_rows(tbl_lines)
			json_str = json.dumps(json_rows, ensure_ascii=False)

			parts = []
			if rag_meta:
				parts.append(rag_meta)
			if title:
				parts.append(f"**Table: {title}**")
			parts.append(md_table)
			if notes:
				parts.append(f"Notes: {notes}")

			text = "\n".join(parts).strip()
			cid = make_id(text, source_file)

			chunks.append(
				Chunk(
					chunk_id=cid,
					text=text,
					source_file=source_file,
					section_heading=title,
				)
			)
		else:
			i += 1

	return chunks


def extract_text_chunks(markdown: str, source_file: str) -> list[Chunk]:
	splitter = RecursiveCharacterTextSplitter(
		chunk_size=chars(512),
		chunk_overlap=300,
		separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
	)

	masked = markdown
	placeholders: dict[str, str] = {}

	def _placeholder(content: str) -> str:
		key = f"__MASKED_{uuid.uuid4().hex[:8]}__"
		placeholders[key] = content
		return key

	masked = DISPLAY_EQ_RE.sub(lambda m: _placeholder(m.group(0)), masked)
	masked = RAG_META_RE.sub(lambda m: _placeholder(m.group(0)), masked)
	masked = re.sub(
		r"(?:^\|.*\n)+",
		lambda m: _placeholder(m.group(0)),
		masked,
		flags=re.MULTILINE,
	)

	raw_splits = splitter.split_text(masked)

	chunks: list[Chunk] = []
	lines = markdown.splitlines()

	for split in raw_splits:
		stripped = split.strip()
		if not stripped or all(k in stripped for k in placeholders if stripped == k):
			continue
		if re.fullmatch(r"__MASKED_[0-9a-f]+__", stripped):
			continue

		approx_pos = masked.find(split[:60].strip())
		line_no = masked[:max(0, approx_pos)].count("\n") if approx_pos >= 0 else 0
		heading = _current_heading(lines, line_no)

		cid = make_id(split, source_file)
		chunks.append(
			Chunk(
				chunk_id=cid,
				text=split,
				source_file=source_file,
				section_heading=heading,
			)
		)

	return chunks


_CHILD_SPLITTER = RecursiveCharacterTextSplitter(
	chunk_size=chars(200),
	chunk_overlap=100,
	separators=["\n\n", "\n", ". ", " ", ""],
)


def _make_children(parent: Chunk) -> list[Chunk]:
	child_texts = _CHILD_SPLITTER.split_text(parent.text)
	children: list[Chunk] = []
	for ct in child_texts:
		ct = ct.strip()
		if not ct:
			continue
		cid = make_id(ct, parent.source_file + "_child")
		child = Chunk(
			chunk_id=cid,
			text=ct,
			source_file=parent.source_file,
			section_heading=parent.section_heading,
			parent_id=parent.chunk_id,
			chunk_role="child",
		)
		children.append(child)
	return children


def _equation_parent_child(eq_chunk: Chunk) -> tuple[Chunk, list[Chunk]]:
	parent = Chunk(
		**{k: v for k, v in asdict(eq_chunk).items() if k not in ("parent_id", "chunk_role", "child_ids")},
		parent_id=None,
		chunk_role="parent",
		child_ids=[],
	)

	lines = eq_chunk.text.splitlines()
	summary_lines = [
		l.strip()
		for l in lines
		if l.strip() and not l.strip().startswith("$$") and not l.strip().startswith("<!--")
	][:3]
	child_text = " ".join(summary_lines) or eq_chunk.text[:120]

	cid = make_id(child_text, eq_chunk.source_file + "_eq_child")
	child = Chunk(
		chunk_id=cid,
		text=child_text,
		source_file=eq_chunk.source_file,
		section_heading=eq_chunk.section_heading,
		has_variables=eq_chunk.has_variables,
		parent_id=parent.chunk_id,
		chunk_role="child",
	)
	parent.child_ids = [child.chunk_id]
	return parent, [child]



def _extract_table_rows_from_text(text: str) -> list[dict]:
	table_lines = [line for line in text.splitlines() if TABLE_ROW_RE.match(line)]
	return _table_to_json_rows(table_lines)


def _table_parent_child(tbl_chunk: Chunk) -> tuple[Chunk, list[Chunk]]:
	parent = Chunk(
		**{k: v for k, v in asdict(tbl_chunk).items() if k not in ("parent_id", "chunk_role", "child_ids")},
		parent_id=None,
		chunk_role="parent",
		child_ids=[],
	)

	json_rows: list[dict] = _extract_table_rows_from_text(tbl_chunk.text)
	children: list[Chunk] = []

	if not json_rows:
		child_text = tbl_chunk.section_heading or tbl_chunk.text[:200]
		cid = make_id(child_text, tbl_chunk.source_file + "_tbl_child")
		child = Chunk(
			chunk_id=cid,
			text=child_text,
			source_file=tbl_chunk.source_file,
			section_heading=tbl_chunk.section_heading,
			parent_id=parent.chunk_id,
			chunk_role="child",
		)
		children.append(child)
	else:
		group_size = 3 if len(json_rows) > 10 else 1
		for batch_start in range(0, len(json_rows), group_size):
			batch = json_rows[batch_start: batch_start + group_size]
			if batch:
				headers = list(batch[0].keys())
				header_row = "| " + " | ".join(headers) + " |"
				sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
				data_rows = [
					"| " + " | ".join(str(row.get(h, "")) for h in headers) + " |"
					for row in batch
				]
				child_text = "\n".join([header_row, sep_row] + data_rows)
				if tbl_chunk.section_heading:
					child_text = f"From table: {tbl_chunk.section_heading}\n" + child_text
			else:
				continue

			cid = make_id(child_text, tbl_chunk.source_file + f"_tbl_child_{batch_start}")
			child = Chunk(
				chunk_id=cid,
				text=child_text,
				source_file=tbl_chunk.source_file,
				section_heading=tbl_chunk.section_heading,
				parent_id=parent.chunk_id,
				chunk_role="child",
			)
			children.append(child)

	parent.child_ids = [c.chunk_id for c in children]
	return parent, children


def run_strategy3(md_files: list[Path], output_dir: Path) -> None:
	print("\n" + "=" * 60)
	print("STRATEGY 3 - Small-to-large (parent-child) chunking")
	print("=" * 60)

	all_parents: list[Chunk] = []
	all_children: list[Chunk] = []

	for md_file in md_files:
		print(f"  Processing: {md_file.name}", flush=True)
		markdown = md_file.read_text(encoding="utf-8", errors="replace")

		eq_chunks = extract_equations(markdown, md_file.name)
		for eq in eq_chunks:
			parent, children = _equation_parent_child(eq)
			all_parents.append(parent)
			all_children.extend(children)

		tbl_chunks = extract_tables(markdown, md_file.name)
		for tbl in tbl_chunks:
			parent, children = _table_parent_child(tbl)
			all_parents.append(parent)
			all_children.extend(children)

		text_chunks = extract_text_chunks(markdown, md_file.name)
		for txt in text_chunks:
			parent = Chunk(
				**{k: v for k, v in asdict(txt).items() if k not in ("parent_id", "chunk_role", "child_ids")},
				parent_id=None,
				chunk_role="parent",
				child_ids=[],
			)
			children = _make_children(parent)
			parent.child_ids = [c.chunk_id for c in children]
			all_parents.append(parent)
			all_children.extend(children)

		print(f"    parents={len(all_parents)}, children={len(all_children)}", flush=True)

	def dedup(lst: list[Chunk]) -> list[Chunk]:
		seen: set[str] = set()
		out: list[Chunk] = []
		for c in lst:
			if c.chunk_id not in seen:
				seen.add(c.chunk_id)
				out.append(c)
		return out

	all_parents = dedup(all_parents)
	all_children = dedup(all_children)

	_write_jsonl(output_dir / "strategy3_parents.jsonl", all_parents)
	_write_jsonl(output_dir / "strategy3_children.jsonl", all_children)

	print("\n  Strategy 3 complete:")
	print(f"     parent chunks : {len(all_parents)}")
	print(f"     child chunks  : {len(all_children)}")
	print(f"     output dir    : {output_dir}")
	print("\n  Note: index strategy3_children.jsonl in FAISS.")
	print("        At query time, look up parent_id -> return parent chunk to LLM.")


def _chunk_to_dict(c: Chunk) -> dict:
	return asdict(c)


def _write_jsonl(path: Path, chunks: list[Chunk]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		for c in chunks:
			f.write(json.dumps(_chunk_to_dict(c), ensure_ascii=False) + "\n")
	print(f"  -> wrote {len(chunks):,} chunks to {path.name}")


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Chunk markdown files into RAG-ready JSONL/JSON using Strategy 3.",
	)
	parser.add_argument(
		"--input",
		"-i",
		default=str(DEFAULT_INPUT),
		help="Folder containing input .md files (defaults to MARKDOWN_INPUT_DIR or ./input_markdown).",
	)
	parser.add_argument(
		"--output",
		"-o",
		default=str(DEFAULT_OUTPUT),
		help="Folder where chunked output files are written (defaults to MARKDOWN_OUTPUT_DIR or ./output/parent_child_chunks).",
	)
	args = parser.parse_args()

	input_dir = Path(args.input)
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

	run_strategy3(md_files, output_dir)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
