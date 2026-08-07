"""Run the `mdast-util-normalize-headings` pipeline on Markdown files.

Intended for non-numbered documents (see detect_heading_format.py). Enforces
a consistent heading hierarchy via the Node.js `mdast-normalize-headings`
package.

Output layout:
  <output_root>/<source_stem>/<source_name>

If Node/npm or the package is unavailable, the tool falls back to a local
rule-based normalizer and records that in `summary.json`.
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
from typing import Callable


# ──────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION
# Defaults below are relative to wherever you run this script from. Override
# either with the matching CLI flag instead of editing this file.
# ──────────────────────────────────────────────────────────────────────────
DEFAULT_INPUT_DIR = Path("./data/03_structural_normalization/01_cleaned/non_numbered")
DEFAULT_OUTPUT_ROOT = Path("./data/03_structural_normalization/02_normalized/non_numbered")
# ──────────────────────────────────────────────────────────────────────────


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



def local_rule_based_normalize(content: str) -> str:
    content = demote_non_numeric_front_matter_headings(content)
    content = normalize_heading_hierarchy(content)
    content = promote_numbered_headings(content)
    content = strip_non_numeric_headings(content)
    return content


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


def node_available() -> bool:
    return shutil.which("node") is not None or shutil.which("npx") is not None


def ensure_node_project(project_dir: Path, packages: list[str]) -> None:
    package_json = project_dir / "package.json"
    if not package_json.exists():
        project_dir.mkdir(parents=True, exist_ok=True)
        run_command(["npm", "init", "-y"], cwd=project_dir, timeout=120)
        run_command(["npm", "install", *packages], cwd=project_dir, timeout=600)


def run_node_markdown_tool(
    content: str,
    work_dir: Path,
    method_key: str,
    packages: list[str],
    script_source: str,
    timeout: int,
) -> tuple[str, str]:
    if not node_available():
        return local_rule_based_normalize(content), "fallback: node/npm not available"

    project_dir = work_dir / "node_projects" / method_key
    ensure_node_project(project_dir, packages)

    input_path = project_dir / "input.md"
    output_path = project_dir / "output.md"
    script_path = project_dir / f"{method_key}.mjs"
    write_text(input_path, content)
    script_path.write_text(script_source, encoding="utf-8")

    # Run the script using a relative filename while keeping cwd=project_dir.
    # Passing an absolute path here previously led to module resolution
    # duplicating the project directory in some environments.
    code, _, stderr = run_command(
        ["node", script_path.name, str(input_path), str(output_path)],
        cwd=project_dir,
        timeout=timeout,
    )
    if code != 0 or not output_path.exists():
        detail = stderr.strip() or f"{method_key} failed"
        return local_rule_based_normalize(content), f"fallback: {detail}"
    return read_text(output_path), method_key


def mdast_util_normalize_headings(
    content: str,
    work_dir: Path,
    *,
    fallback_fn: Callable[[str], str] = lambda s: s,
) -> tuple[str, str]:
    try:
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
        )
    except Exception:
        return fallback_fn(content), "fallback: unexpected error"


def process_all(input_dir: Path, output_root: Path) -> list[Outcome]:
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir = output_root / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    outcomes: list[Outcome] = []
    files = sorted(input_dir.rglob("*.md"))
    for source in files:
        content = read_text(source)
        normalized, backend = mdast_util_normalize_headings(content, work_dir, fallback_fn=local_rule_based_normalize)
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
    parser = argparse.ArgumentParser(description="Run mdast-util-normalize-headings on a folder of markdown files.")
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
