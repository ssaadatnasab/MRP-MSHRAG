#!/usr/bin/env python3
"""Run the md_reheader normalization method on a directory of Markdown files.

This script mirrors the `md_reheader` branch from the compare pipeline, but it
runs only that method.

Input default:
  /home/user/Desktop/Improving Quality MarkDown/Testing Files

Output default:
  /home/user/Desktop/Improving Quality MarkDown/Output/1-Refinement 2-Enrichment/1-

Output layout:
  <output_root>/<source_stem>/<source_name>

Behavior:
  - Prefer the `rehead` CLI if available.
  - Otherwise use `reheader`.
  - If neither CLI is available, fall back to the local rule-based normalizer.
  - Write a `summary.json` file with the backend used for each file.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT_DIR = Path("/home/user/Desktop/Improving Quality MarkDown/Testing Files")
DEFAULT_OUTPUT_ROOT = Path("/home/user/Desktop/Improving Quality MarkDown/Output/1-Refinement 2-Enrichment/1-")
LOCAL_REHEADER = Path("/home/user/Desktop/.tools/reheader/node_modules/.bin/reheader")


@dataclass
class Outcome:
	source: str
	output_path: str
	backend: str
	status: str
	detail: str = ""


def read_text(path: Path) -> str:
	return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, content: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(content, encoding="utf-8")


def run_command(command: list[str], cwd: Path | None = None, timeout: int | None = None) -> tuple[int, str, str]:
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

	result: list[str] = []
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


def demote_non_numeric_front_matter_headings(content: str) -> str:
	lines = content.splitlines()
	out_lines: list[str] = []
	seen_numbered_section = False
	numbered_pattern = re.compile(r"^(?:#{1,6}\s*)?(?:[-*+]\s+)?(?:\*{0,2})?\d+(?:\.\d+)*")

	for line in lines:
		stripped = line.lstrip()
		if numbered_pattern.match(stripped):
			seen_numbered_section = True
			out_lines.append(line)
			continue
		if not seen_numbered_section and re.match(r"^#{1,6}\s+", stripped):
			plain = re.sub(r"^#{1,6}\s+", "", stripped)
			plain = plain.strip().strip("*").strip()
			out_lines.append(plain)
			continue
		out_lines.append(line)

	return "\n".join(out_lines)


def strip_non_numeric_headings(content: str) -> str:
	lines = content.splitlines()
	out_lines: list[str] = []

	for line in lines:
		stripped = line.lstrip()
		match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
		if match:
			title = match.group(2).strip()
			if not re.match(r"^\d+(?:\.\d+)*\s", title):
				out_lines.append(title.strip("*").strip())
				continue
		out_lines.append(line)

	return "\n".join(out_lines)


def remove_formula_context_labels(content: str) -> str:
	return re.sub(
		r"(?mi)^\s*(?:Formula\s*context\s*for\s*RAG|RAG\s*enrichment\s*block):?\s*$\n?",
		"",
		content,
	)


def local_rule_based_normalize(content: str) -> str:
	content = demote_non_numeric_front_matter_headings(content)
	content = normalize_heading_hierarchy(content)
	content = promote_numbered_headings(content)
	content = strip_non_numeric_headings(content)
	content = remove_formula_context_labels(content)
	return content


def md_reheader_normalize(content: str, work_dir: Path) -> tuple[str, str]:
	rehead = shutil.which("rehead")
	reheader = shutil.which("reheader")

	if rehead is None and reheader is None and not LOCAL_REHEADER.exists():
		return local_rule_based_normalize(content), "fallback: rehead/reheader CLI not available"

	with tempfile.TemporaryDirectory(dir=work_dir) as tmpdir:
		tmpdir_path = Path(tmpdir)
		input_path = tmpdir_path / "input.md"
		output_path = tmpdir_path / "output.md"
		write_text(input_path, content)

		if rehead is not None:
			command = [rehead, "-i", str(input_path), "-o", str(output_path), "--cpu", "--force"]
			backend = "md-reheader"
			read_from = output_path
		else:
			reheader_cmd = reheader if reheader is not None else str(LOCAL_REHEADER)
			command = [reheader_cmd, str(input_path)]
			backend = "reheader"
			read_from = input_path

		code, _, stderr = run_command(command, cwd=tmpdir_path, timeout=300)
		if code != 0 or not read_from.exists():
			return local_rule_based_normalize(content), f"fallback: {stderr.strip() or 'rehead/reheader failed'}"
		return read_text(read_from), backend


def process_all(input_dir: Path, output_root: Path) -> list[Outcome]:
	output_root.mkdir(parents=True, exist_ok=True)
	outcomes: list[Outcome] = []
	files = sorted(input_dir.rglob("*.md"))
	work_dir = output_root / "_work"
	work_dir.mkdir(parents=True, exist_ok=True)

	for source in files:
		content = read_text(source)
		normalized, backend = md_reheader_normalize(content, work_dir)
		target_path = output_root / source.relative_to(input_dir)
		write_text(target_path, normalized)
		outcomes.append(
			Outcome(
				source=str(source),
				output_path=str(target_path),
				backend=backend,
				status="written",
			)
		)

	summary_path = output_root / "summary.json"
	write_text(
		summary_path,
		json.dumps(
			[
				{
					"source": item.source,
					"output_path": item.output_path,
					"backend": item.backend,
					"status": item.status,
					"detail": item.detail,
				}
				for item in outcomes
			],
			indent=2,
		),
	)
	return outcomes


def main() -> int:
	parser = argparse.ArgumentParser(description="Run md_reheader normalization on a folder of markdown files.")
	parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Input directory containing .md files")
	parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Folder where outputs will be written")
	args = parser.parse_args()

	input_dir = Path(args.input_dir)
	output_root = Path(args.output_root)
	if not input_dir.exists():
		print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
		return 2

	outcomes = process_all(input_dir, output_root)
	print(f"Wrote {len(outcomes)} outputs to {output_root}")
	for item in outcomes:
		print(f"- {item.source} -> {item.output_path} [{item.backend}]")
	print(f"Summary: {output_root / 'summary.json'}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
