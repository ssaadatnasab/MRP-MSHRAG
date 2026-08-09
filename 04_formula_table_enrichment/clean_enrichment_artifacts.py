#!/usr/bin/env python3
"""Post-LLM cleanup: remove accidental heading artifacts.

Problem this script fixes
------------------------
Some LLM enrichment runs occasionally echo prompt “STEP …” instructions and/or
emit enrichment metadata labels as Markdown headings (prefixed with `#`, `##`,
`###`, etc.). That makes these lines get treated as document headings and
pollutes downstream heading-based tooling.

This cleaner performs two conservative, line-based transforms:

1) Remove standalone LLM prompt step lines entirely, e.g.:
   - "### STEP 2 — Add RAG enrichment block"
   - "STEP 1 — Fix the LaTeX (if needed):"
   - "**Step 2: RAG Enrichment Block**"

2) If certain enrichment labels are emitted as headings, strip only the leading
   `#` markers (keep the text), e.g.:
   - "### Formula name: Energy Efficiency" -> "Formula name: Energy Efficiency"
   - "#### **Domain keywords:** foo"       -> "**Domain keywords:** foo"

The script intentionally does NOT try to remove the enrichment blocks
themselves; it only prevents them from becoming headings.

CLI
---
Directory mode (recommended):
	python After_E_Deleting_MetaData#.py \
	  --input-path "/path/to/enriched" \
	  --output-path "/path/to/enriched_cleaned"

Single file mode:
	python After_E_Deleting_MetaData#.py \
	  --input-path "in.md" \
	  --output-path "out.md"
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT_PATH = Path("./input_markdown")
DEFAULT_OUTPUT_PATH = Path("./output_markdown")


FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")

# Remove stray <sup>r</sup> tokens and isolated <sup> or </sup> artifacts
SUP_R_RE = re.compile(r"<sup>\s*r\s*</sup>", flags=re.IGNORECASE)
ISOLATED_SUP_RE = re.compile(r"(^|\s)</?sup>(\s|$)", flags=re.IGNORECASE | re.MULTILINE)


@dataclass
class CleanupStats:
	step_lines_removed: int = 0
	heading_markers_stripped: int = 0
	sup_tags_removed: int = 0

	def add(self, other: "CleanupStats") -> None:
		self.step_lines_removed += int(other.step_lines_removed)
		self.heading_markers_stripped += int(other.heading_markers_stripped)
		self.sup_tags_removed += int(other.sup_tags_removed)


def _toggle_fence(in_fence: str | None, line: str) -> str | None:
	"""Toggle fenced-code-block state based on a line."""
	match = FENCE_RE.match(line)
	if not match:
		return in_fence
	fence = match.group(1)
	if in_fence is None:
		return fence
	# Only close on a fence of the same character family.
	if fence[0] == in_fence[0]:
		return None
	return in_fence


STEP_LINE_RE = re.compile(
	r"""^
	\s*(?:[-*+]\s*)?          # optional list marker
	(?:\#{1,6}\s*)?             # optional ATX heading markers
	(?:\*\*|__)?\s*            # optional bold/underline opener
	step\s*\d+(?:\.\d+)?\s*   # STEP 1 / Step 1.5 / Step 2.1
	(?:[:.\-–—])\s*            # separator between number and title
	(?P<title>.*?)              # title
	\s*(?:\*\*|__)?\s*         # optional bold/underline closer
	$
	""",
	flags=re.IGNORECASE | re.VERBOSE,
)


def _norm_text(text: str) -> str:
	# Normalization for robust matching across punctuation/casing variants.
	text = text.replace("—", "-").replace("–", "-")
	text = re.sub(r"\s+", " ", text).strip().lower()
	text = re.sub(r"[\s:;,.!?\-]+$", "", text)  # trim trailing punctuation
	return text


STEP_TITLE_KEYWORDS = (
	"rewrite the header row",
	"keep the data rows exactly as-is",
	"keep the data rows exactly as is",
	"fix the latex",
	"corrected latex",
	"equation verbalization",
	"verbalization of formula",
	"verbalization of the formula",
	"rag enrichment block",
	"RAG Enrichment Block",
	"fix the table formatting",
 	"corrected latex formula",
)


def _is_llm_step_line(line: str) -> bool:
	"""Return True if the line looks like an echoed LLM prompt step title."""
	match = STEP_LINE_RE.match(line)
	if not match:
		return False
	title = match.group("title") or ""
	title_norm = _norm_text(title)
	if not title_norm:
		return False
	return any(k in title_norm for k in STEP_TITLE_KEYWORDS)


# Labels where we should strip heading markers if they appear as headings.
# Keep this list narrow to avoid touching real document headings.
HEADING_STRIP_LABEL_RE = re.compile(
	r"""^
	(?:\*\*|__)?\s*  # optional bold opener
	(?:
		corrected\s+latex(?:\s+formula)?|
		rag\s+enrichment\s+block|
		formula\s+name|
		also\s+known\s+as|
		use\s+case|
		domain\s+keywords|
		table\s+subject|
		property\s+type|
		columns\s+and\s+units
	)\b
	""",
	flags=re.IGNORECASE | re.VERBOSE,
)


HEADING_PREFIX_RE = re.compile(r"^(?P<indent>\s*)#{1,6}\s*(?P<rest>\S.*)$")


def _strip_heading_marker_if_label(line: str) -> tuple[str, bool]:
	"""Strip leading `#` markers ONLY for known enrichment label lines."""
	match = HEADING_PREFIX_RE.match(line)
	if not match:
		return line, False

	indent = match.group("indent") or ""
	rest = match.group("rest") or ""
	if not HEADING_STRIP_LABEL_RE.match(rest):
		return line, False

	return indent + rest.lstrip(), True


def clean_llm_enrichment_artifacts(markdown: str) -> tuple[str, CleanupStats]:
	"""Clean a markdown string and return (cleaned_markdown, stats)."""
	keep_trailing_newline = markdown.endswith("\n")
	lines = markdown.splitlines()

	stats = CleanupStats()
	out_lines: list[str] = []

	in_fence: str | None = None
	for line in lines:
		in_fence = _toggle_fence(in_fence, line)
		if in_fence is not None:
			if _is_llm_step_line(line):
				stats.step_lines_removed += 1
				continue
			out_lines.append(line)
			continue

		if _is_llm_step_line(line):
			stats.step_lines_removed += 1
			continue

		new_line, stripped = _strip_heading_marker_if_label(line)
		if stripped:
			stats.heading_markers_stripped += 1
		out_lines.append(new_line)

	cleaned = "\n".join(out_lines)
	if keep_trailing_newline:
		cleaned += "\n"

	# Remove stray <sup> artifacts such as "<sup>r</sup>" or isolated <sup> / </sup> markers
	removed = 0
	cleaned, n = SUP_R_RE.subn("", cleaned)
	removed += n
	# Replace isolated lone tags with a single space to avoid word-joining; count removals.
	cleaned, n = ISOLATED_SUP_RE.subn(" ", cleaned)
	removed += n
	stats.sup_tags_removed += int(removed)
	return cleaned, stats


def _iter_markdown_files(root_dir: Path, exclude_dir: Path | None = None) -> Iterable[Path]:
	for file_path in sorted(root_dir.rglob("*.md")):
		if not file_path.is_file():
			continue
		if exclude_dir is not None:
			try:
				file_path.relative_to(exclude_dir)
			except ValueError:
				pass
			else:
				continue
		yield file_path


def clean_markdown_file(input_file: Path, output_file: Path) -> CleanupStats:
	raw = input_file.read_text(encoding="utf-8", errors="replace")
	cleaned, stats = clean_llm_enrichment_artifacts(raw)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	output_file.write_text(cleaned, encoding="utf-8")
	return stats


def main() -> int:
	parser = argparse.ArgumentParser(
		description=(
			"Remove echoed LLM STEP instruction lines and prevent enrichment labels from being Markdown headings."
		)
	)
	parser.add_argument(
		"--input-path",
		default=str(DEFAULT_INPUT_PATH),
		help="Input markdown file or directory.",
	)
	parser.add_argument(
		"--output-path",
		default=str(DEFAULT_OUTPUT_PATH),
		help="Output markdown file or directory.",
	)
	args = parser.parse_args()

	input_path = Path(args.input_path)
	output_path = Path(args.output_path)

	if not input_path.exists():
		raise SystemExit(f"Input not found: {input_path}")

	# Directory mode
	if input_path.is_dir():
		output_path.mkdir(parents=True, exist_ok=True)
		if not output_path.is_dir():
			raise SystemExit("When input is a directory, output must be a directory")

		totals = CleanupStats()
		processed = 0

		# Avoid re-processing outputs if output is a subdirectory of input.
		# If output == input, allow in-place processing.
		exclude_dir = None
		try:
			rel = output_path.resolve().relative_to(input_path.resolve())
		except Exception:
			exclude_dir = None
		else:
			# When output is exactly the same directory as input, do not exclude it.
			from pathlib import Path as _Path
			if rel == _Path('.'):  # same directory
				exclude_dir = None
			else:
				exclude_dir = output_path

		for md_file in _iter_markdown_files(input_path, exclude_dir=exclude_dir):
			rel = md_file.relative_to(input_path)
			out_file = output_path / rel
			file_stats = clean_markdown_file(md_file, out_file)
			totals.add(file_stats)
			processed += 1

		print(f"Cleaned {processed} file(s) into: {output_path}")
		print(f"Removed STEP lines: {totals.step_lines_removed}")
		print(f"Stripped heading markers on labels: {totals.heading_markers_stripped}")
		print(f"Removed stray <sup> artifacts: {totals.sup_tags_removed}")
		return 0

	# Single-file mode
	if output_path.exists() and output_path.is_dir():
		raise SystemExit("When input is a file, output must be a file path (not a directory)")
	if output_path.suffix.lower() != ".md":
		output_path = output_path.with_suffix(".md")

	stats = clean_markdown_file(input_path, output_path)
	print(f"Wrote: {output_path}")
	print(f"Removed STEP lines: {stats.step_lines_removed}")
	print(f"Stripped heading markers on labels: {stats.heading_markers_stripped}")
	print(f"Removed stray <sup> artifacts: {stats.sup_tags_removed}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

