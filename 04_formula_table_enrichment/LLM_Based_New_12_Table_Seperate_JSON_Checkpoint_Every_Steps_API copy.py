# Solving the problem that Formulas are not included in MD file

#!/usr/bin/env python3
"""Parallel markdown enhancement using LLM API with asyncio.

This optimized version processes multiple markdown files concurrently
using asyncio, dramatically speeding up LLM enhancement.

A structural pre-processing pipeline runs BEFORE the LLM to fix
common PDF-to-markdown conversion artifacts without any LLM calls:
  - Remove <br> / <BR> / <br/> tags from table cells (replaced with a space)
  - Remove duplicate H1 headings (cover/title page artifacts)
  - Remove HTML span anchors (<span id="page-N-N"></span>)
  - Remove running header/footer tables (| Chapter | ix |)
  - Fix broken headings with literal \\n inside them
  - Normalize heading hierarchy (H1 → H3 gap → promotes to H1 → H2)

After pre-processing, the LLM pipeline handles:
  - Table headers with collapsed <br> artifacts  → task: "table_header"
  - Math / formula blocks                        → task: "math"
  - General table formatting                     → task: "table"

For every processed markdown file two additional JSON files are written
alongside the enhanced .md in the output tree:

  Formulas_<stem>.json   – one entry per math element sent to the LLM:
                            {"input": "...", "output": "..." | null}
  Tables_<stem>.json     – one entry per table sent to the LLM
                            (both table_header and table passes):
                            {"task": "table_header"|"table",
                             "input": "...", "output": "..." | null}

  Empty arrays are written when no elements of that type were found.
  Entries where the LLM result failed validation are recorded with
  "output": null (the original text was kept in the .md).

Example:
    python improve_markdown_qualityLLM_parallel.py \\
        --markdown-root "/home/user/Desktop/marker_pdf_output" \\
        --out-root "/home/user/Desktop/marker_pdf_output_improved" \\
        --model llama-3.1-70b \\
        --workers 8

    # Fast mode: process with concurrency=16
    python improve_markdown_qualityLLM_parallel.py \\
        --markdown-root "/home/user/Desktop/marker_pdf_output" \\
        --out-root "/home/user/Desktop/marker_pdf_output_improved" \\
        --workers 16 \\
        --batch-enhancement True
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Callable, Awaitable
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from collections import defaultdict
import time
from typing import Set


YUNWU_BASE_URL = os.getenv("YUNWU_API_BASE_URL", "https://api.wlai.vip/v1")
DEFAULT_MODELS = (
    os.getenv("YUNWU_MODEL", "llama-3.1-70b"),
)


def _get_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    if explicit_key:
        return explicit_key
    for env_name in ("YUNWU_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(env_name)
        if value:
            return value
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnhancementStats:
    """Track statistics during processing."""
    enhanced: int = 0
    skipped: int = 0
    failed: int = 0
    total_time: float = 0.0

    def summary(self) -> str:
        avg_time = self.total_time / max(1, self.enhanced) if self.enhanced > 0 else 0
        return (
            f"Enhanced: {self.enhanced} | Skipped: {self.skipped} | "
            f"Failed: {self.failed} | Avg time: {avg_time:.2f}s"
        )


def _checkpoint_path_for(out_root: Path) -> Path:
    return out_root / ".enhancement_checkpoint.json"


def _load_checkpoint(checkpoint_file: Path) -> Set[str]:
    if not checkpoint_file.exists():
        return set()
    try:
        data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        completed = data.get("completed_files", [])
        return {str(item) for item in completed}
    except Exception:
        return set()


def _write_checkpoint(checkpoint_file: Path, completed_files: Set[str]) -> None:
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_files": sorted(completed_files),
        "updated_at": time.time(),
    }
    checkpoint_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# LLM pair logging
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMPairLog:
    """Accumulates (input, output) pairs for a single markdown file.

    math_pairs   – every math element submitted to the LLM.
    table_pairs  – every table submitted to the LLM (header + general passes).

    Each entry is a plain dict so it serialises to JSON directly:
        math  : {"input": str, "output": str | None}
        tables: {"task": str, "input": str, "output": str | None}

    "output" is None when the LLM response failed validation and the
    original text was kept in the final markdown.
    """
    math_pairs:  list[dict] = field(default_factory=list)
    table_pairs: list[dict] = field(default_factory=list)


@dataclass
class FileProgressReporter:
    """Emit compact, monotonic progress updates for one file."""

    file_label: str
    total_steps: int
    completed_steps: int = 0
    started_at: float = field(default_factory=time.time)

    def update(self, stage: str, step_increment: int = 1) -> None:
        self.completed_steps = min(self.total_steps, self.completed_steps + step_increment)
        percent = (self.completed_steps / self.total_steps * 100.0) if self.total_steps else 100.0
        elapsed = time.time() - self.started_at
        print(
            f"[{percent:6.2f}%] {self.file_label} | {stage} | "
            f"{self.completed_steps}/{self.total_steps} steps | {elapsed:.1f}s elapsed",
            flush=True,
        )

    def finish(self, stage: str = "done") -> None:
        self.completed_steps = self.total_steps
        elapsed = time.time() - self.started_at
        print(
            f"[100.00%] {self.file_label} | {stage} | "
            f"{self.completed_steps}/{self.total_steps} steps | {elapsed:.1f}s elapsed",
            flush=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(content: str) -> str:
    """Structural pre-processing: remove common PDF-to-markdown artifacts.

    Runs entirely without any LLM call. Fixes applied:
      1. Strip <br> variants from table cells (replace with a space)
      2. Remove HTML span page-anchor tags
      3. Collapse duplicate blank lines inside tables
    """
    # 1. Remove HTML <br> variants inside pipe-table cells.
    #    Handles: <br>, <BR>, <br/>, <BR/>, <br />, <BR />
    #    Replacing with a single space avoids merging words together.
    content = re.sub(r"<br\s*/?>", " ", content, flags=re.IGNORECASE)

    # 2. Remove HTML span page-anchor tags produced by some PDF converters.
    #    e.g.  <span id="page-3-1"></span>
    content = re.sub(r'<span\s+id="page-[\d-]+"[^>]*>\s*</span>', "", content)

    # 3. Collapse runs of multiple spaces inside table cells that may have
    #    been introduced by the <br> → space substitution above.
    def _collapse_cell_spaces(line: str) -> str:
        if "|" not in line:
            return line
        cells = line.split("|")
        cells = [re.sub(r"  +", " ", c) for c in cells]
        return "|".join(cells)

    content = "\n".join(_collapse_cell_spaces(l) for l in content.splitlines())

    return content


# ─────────────────────────────────────────────────────────────────────────────
# API client
# ─────────────────────────────────────────────────────────────────────────────

class AsyncAPIClient:
    def __init__(
        self,
        base_url: str = YUNWU_BASE_URL,
        model: str = DEFAULT_MODELS[0],
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = _get_api_key(api_key)
        self.session_ready = False
        self._verify_connection()

    def _verify_connection(self) -> bool:
        if not self.api_key:
            print(
                "ERROR: No API key provided. Set --api-key or one of YUNWU_API_KEY/OPENAI_API_KEY.",
                file=sys.stderr,
            )
            return False
        try:
            url = f"{self.base_url}/models"
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                json.loads(response.read())
                print(f"✓ API ready: {self.base_url} model={self.model}", flush=True)
                self.session_ready = True
                return True
        except Exception as e:
            print(f"ERROR: Cannot connect to API at {self.base_url}: {e}", file=sys.stderr)
            return False

    async def enhance_async(self, content: str, task: str) -> Optional[str]:
        prompt = self._build_prompt(content, task)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._enhance_blocking, prompt)

    def _enhance_blocking(self, prompt: str) -> Optional[str]:
        try:
            url = f"{self.base_url}/chat/completions"
            data = json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=1800000) as response:
                result = json.loads(response.read())
                enhanced = result["choices"][0]["message"]["content"].strip()
                return enhanced if enhanced else None
        except Exception as e:
            print(f"WARNING: API request failed: {e}", file=sys.stderr, flush=True)
            return None

    @staticmethod
    def _build_prompt(content: str, task: str) -> str:
        """Build a task-specific enhancement prompt."""
        domain = (
            "You are a technical editor of engineering textbooks covering "
            "building energy science, HVAC systems, thermodynamics, and thermal comfort. "
        )

        # ── TABLE HEADER ──────────────────────────────────────────────────────
        if task == "table_header":
            return (
                f"{domain}"
                "You are cleaning up a markdown table that was extracted from a PDF. "
                "The table has multi-line or broken header cells caused by <br> tags "
                "or line wraps that were collapsed into spaces.\n\n"

                "STEP 1 — Rewrite the header row:\n"
                "Produce a single clean header row. Combine broken column labels into "
                "short, readable names "
                "(e.g. 'O.D. of Pipe / Insulation / Conduit', 'End of Valve Stem or Body'). "
                "Use Title Case. Keep every column — do NOT merge or drop any.\n\n"

                "STEP 2 — Keep the data rows exactly as-is:\n"
                "Do NOT change any numeric values, units, or data cells.\n\n"

                "INPUT TABLE:\n"
                f"{content}\n\n"

                "Output the caption line first, then the corrected table. "
                "No commentary after the table."
            )

        # ── MATH ──────────────────────────────────────────────────────────────
        elif task == "math":
            return (
                f"{domain}"
                "You are enriching a formula chunk so that a RAG retrieval system "
                "can find it accurately when a user asks a calculation question.\n\n"

                "STEP 1 — Fix the LaTeX (if needed):\n"
                "Review the rule-based formula extraction. Fix only clear errors: missing braces, "
                "broken subscripts/superscripts, missing \\text{} for units, unbalanced delimiters. "
                "Do NOT rewrite correct math. Preserve all variable names exactly.\n\n"

                "STEP 1.5 — EQUATION VERBALIZATION:\n"
                "Write one plain-English sentence that fully describes what the formula "
                "calculates, using the variable names from the formula. No LaTeX.\n\n"

                "STEP 2 — Add RAG enrichment block:\n"
                "After the corrected formula, add a markdown block with ALL of the following:\n\n"
                "**Formula name:** <the standard name of this formula, "
                "e.g. 'Fourier's Law of Heat Conduction'>\n"
                "**Also known as:** <list 2–4 synonyms or alternate names used in ASHRAE, "
                "building codes, or textbooks>\n"
                "**Use case:** <one sentence: when does an engineer use this formula? "
                "what does it calculate?>\n"
                "**Domain keywords:** <comma-separated list of 6–12 technical terms a student "
                "would use to search for this formula, e.g.: conduction, thermal resistance, "
                "U-value, heat flux, building envelope, ASHRAE 90.1, "
                "steady-state heat transfer>\n\n"

                "INPUT FORMULA:\n"
                f"{content}\n\n"

                "Output the corrected LaTeX first, then the complete RAG enrichment block. "
                "Do not add unrelated commentary. "
                "Keep variable names exactly as they appear in the formula."
            )

        # ── TABLE (general formatting) ────────────────────────────────────────
        elif task == "table":
            return (
                f"{domain}"
                "You are enriching a data table chunk so that a RAG retrieval system "
                "can find it accurately when a user asks for specific property values "
                "(e.g. steam enthalpy at 42 bar, R-134a saturation pressure at 40°C, "
                "U-value of double-pane glass).\n\n"

                "STEP 1 — Fix the table formatting (if needed):\n"
                "Ensure proper pipe | formatting, a header row, a separator row (|---|---|), "
                "and consistent column alignment. "
                "Do NOT add, remove, or change any data values or units.\n\n"

                "STEP 2 — Add RAG enrichment block BEFORE the table:\n"
                "Insert the following metadata block immediately before the table:\n\n"
                "**Table subject:** <one sentence describing what this table contains, "
                "e.g. 'Thermodynamic properties of superheated steam at pressures 20–60 bar "
                "and temperatures 200–500°C'>\n"
                "**Property type:** <what kind of values: e.g. 'thermodynamic properties', "
                "'thermal conductivity values', 'heat transfer coefficients', "
                "'psychrometric properties', 'refrigerant saturation properties'>\n"
                "**Columns and units:** <list every column with its unit, "
                "e.g. 'Pressure (bar) | Temperature (°C) | h enthalpy (kJ/kg) | "
                "s entropy (kJ/kg·K) | v specific volume (m³/kg)'>\n"
                "**Domain keywords:** <8–15 comma-separated terms a student would use to find "
                "this table, e.g.: steam tables, superheated steam, enthalpy, entropy, "
                "specific volume, Rankine cycle, turbine calculation, ASHRAE, boiler, "
                "thermodynamics>\n\n"

                "INPUT TABLE:\n"
                f"{content}\n\n"

                "Output the complete RAG enrichment block first, then the corrected table. "
                "Do not summarize or paraphrase any numeric values. "
                "Do not add any commentary after the table."
            )

        # ── FALLBACK ──────────────────────────────────────────────────────────
        else:
            return (
                f"{domain}"
                "Improve the markdown below: fix structure, clarity, consistency, "
                "and readability. Keep ALL content."
                "\n\n"
                f"{content}"
                "\n\n"
                "Output ONLY the improved markdown, no explanation."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_enhancement(original: str, enhanced: str) -> bool:
    """Basic sanity check: reject LLM output that looks like a regression.

    Returns True if the enhanced version looks safe to use.
    
    For math: allows enrichment blocks that add metadata and explanation
    around the original formula, so delimiter counts may differ.
    """
    # Must not be dramatically shorter (LLM dropped content)
    if len(enhanced) < len(original) * 0.70:
        return False

    # For math enrichment, the enhanced version will have MORE delimiters
    # because it includes the enrichment block. Just ensure it has AT LEAST
    # as many as the original.
    # Example: $formula$ becomes "### Corrected LaTeX:\n$formula$\n\n### RAG Enrichment Block:\n..."
    # So we check: enhanced should have >= original delimiter count (relaxed check)
    orig_delim = original.count("$$")
    enh_delim = enhanced.count("$$")
    if orig_delim > 0 and enh_delim < orig_delim:
        # Original had $$ blocks but enhanced has fewer - likely dropped content
        return False

    # Table rows must not disappear
    orig_pipes = original.count("|")
    enh_pipes = enhanced.count("|")
    if orig_pipes > 10 and enh_pipes < orig_pipes * 0.80:
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Markdown enhancer
# ─────────────────────────────────────────────────────────────────────────────

class AsyncMarkdownEnhancer:
    """LLM-based async markdown enhancement with per-element batching."""

    def __init__(self, llm_client: AsyncAPIClient, batch_enhancement: bool = False):
        self.llm = llm_client
        self.batch_enhancement = batch_enhancement

    @staticmethod
    def _count_table_header_candidates(content: str) -> int:
        lines = content.split("\n")
        count = 0
        i = 0
        while i < len(lines):
            if lines[i].strip().startswith("|"):
                table_lines = [lines[i]]
                i += 1
                while i < len(lines) and (lines[i].strip().startswith("|") or not lines[i].strip()):
                    if lines[i].strip():
                        table_lines.append(lines[i])
                    i += 1
                if AsyncMarkdownEnhancer._has_collapsed_br_headers("\n".join(table_lines)):
                    count += 1
            else:
                i += 1
        return count

    @staticmethod
    def _count_table_blocks(content: str) -> int:
        lines = content.split("\n")
        count = 0
        i = 0
        while i < len(lines):
            if lines[i].strip().startswith("|"):
                count += 1
                i += 1
                while i < len(lines) and (lines[i].strip().startswith("|") or not lines[i].strip()):
                    i += 1
            else:
                i += 1
        return count

    @staticmethod
    def _count_math_blocks(content: str) -> int:
        math_pattern = r"(\$\$[^$]*\$\$|\$[^$\n]+\$)"
        return len(list(re.finditer(math_pattern, content)))

    # ── Issue detection ───────────────────────────────────────────────────────

    def detect_issues(self, content: str) -> dict[str, list]:
        """Detect markdown elements that may benefit from LLM enhancement."""
        issues: dict[str, list] = {
            "math": [],
            "tables": [],
        }
        lines = content.split("\n")
        for i, line in enumerate(lines):
            # Math: inline $...$ or block $$...$$
            if "$" in line:
                if re.search(r"\$[^$\n]{5,200}\$", line):
                    issues["math"].append((i, "inline"))
                if "$$" in line:
                    issues["math"].append((i, "block"))
            # Tables
            if line.strip().startswith("|"):
                issues["tables"].append(i)
        return issues

    # ── Top-level pipeline ────────────────────────────────────────────────────

    async def enhance_content_async(
        self,
        filepath: Path,
        content: str,
        dry_run: bool = False,
        progress: Optional[FileProgressReporter] = None,
        existing_log: Optional[LLMPairLog] = None,
        progress_callback: Optional[Callable[[str, LLMPairLog], Awaitable[None]]] = None,
    ) -> tuple[str, LLMPairLog]:
        """Run the full enhancement pipeline on a markdown string.

        Returns
        -------
        (enhanced_content, log)
            enhanced_content : the fully processed markdown string.
            log              : LLMPairLog with all input/output pairs recorded
                               during this file's LLM calls.
        """
        content = preprocess(content)

        if existing_log is None:
            log = LLMPairLog()
            completed_steps = 0
        else:
            log = existing_log
            content = apply_logged_enhancements(content, log)
            completed_steps = 2 + len(log.math_pairs) + len(log.table_pairs)

        existing_header = sum(1 for entry in log.table_pairs if entry["task"] == "table_header")
        existing_table = sum(1 for entry in log.table_pairs if entry["task"] == "table")
        header_remaining = max(0, self._count_table_header_candidates(content) - existing_header)
        math_remaining = max(0, self._count_math_blocks(content) - len(log.math_pairs))
        table_remaining = max(0, self._count_table_blocks(content) - existing_table)
        total_issues = header_remaining + math_remaining + table_remaining

        if progress is None:
            total_steps = 2 + header_remaining + math_remaining + table_remaining
            progress = FileProgressReporter(
                str(filepath),
                total_steps=max(1, total_steps),
                completed_steps=completed_steps,
            )

        progress.update("preprocess")
        progress.update(f"detected {total_issues} issue(s)")

        if total_issues == 0 or dry_run:
            progress.finish("complete")
            return content, log

        if self.batch_enhancement:
            content, log = await self._enhance_batched(
                content,
                log,
                progress,
                progress_callback=progress_callback,
            )
        else:
            content, log = await self._enhance_table_headers_async(
                content,
                log,
                progress,
                progress_callback=progress_callback,
                skip_existing=sum(1 for entry in log.table_pairs if entry["task"] == "table_header"),
            )
            content, log = await self._enhance_math_async(
                content,
                log,
                progress,
                progress_callback=progress_callback,
                skip_existing=len(log.math_pairs),
            )
            content, log = await self._enhance_tables_async(
                content,
                log,
                progress,
                progress_callback=progress_callback,
                skip_existing=sum(1 for entry in log.table_pairs if entry["task"] == "table"),
            )

        progress.finish("complete")

        return content, log

    # ── Batched mode ──────────────────────────────────────────────────────────

    async def _enhance_batched(
        self,
        content: str,
        log: LLMPairLog,
        progress: Optional[FileProgressReporter] = None,
        progress_callback: Optional[Callable[[str, LLMPairLog], Awaitable[None]]] = None,
    ) -> tuple[str, LLMPairLog]:
        """Run all element-type enhancements sequentially on evolving content."""
        content, log = await self._enhance_table_headers_async(
            content,
            log,
            progress,
            progress_callback=progress_callback,
        )
        content, log = await self._enhance_math_async(
            content,
            log,
            progress,
            progress_callback=progress_callback,
            skip_existing=len(log.math_pairs),
        )
        content, log = await self._enhance_tables_async(
            content,
            log,
            progress,
            progress_callback=progress_callback,
            skip_existing=sum(1 for entry in log.table_pairs if entry["task"] == "table"),
        )
        return content, log

    # ── Table-header enhancer ─────────────────────────────────────────────────

    @staticmethod
    def _has_collapsed_br_headers(table_text: str) -> bool:
        """Return True if the table header row looks like it contained <br> artifacts.

        After preprocess() runs, every <br> becomes a space. We detect the
        resulting run-on column labels with a simple heuristic:
          • At least 2 header cells that are longer than 35 characters, OR
          • Any cell containing an ALL-CAPS run of 3+ words (typical of
            PDF-extracted table headers, e.g. "PIPE INSULATION OR CONDUIT").
        """
        header_lines = [l for l in table_text.splitlines() if l.strip().startswith("|")]
        if not header_lines:
            return False
        header = header_lines[0]
        cells = [c.strip() for c in header.split("|") if c.strip()]

        long_cells = [c for c in cells if len(c) > 35]
        if len(long_cells) >= 2:
            return True

        # ALL-CAPS multi-word cell (e.g. "END OF VALVE STEM OR BODY")
        caps_cells = [c for c in cells if re.search(r"\b[A-Z]{2,}\b \b[A-Z]{2,}\b", c)]
        return len(caps_cells) >= 2

    async def _enhance_table_headers_async(
        self,
        content: str,
        log: LLMPairLog,
        progress: Optional[FileProgressReporter] = None,
        progress_callback: Optional[Callable[[str, LLMPairLog], Awaitable[None]]] = None,
        skip_existing: int = 0,
    ) -> tuple[str, LLMPairLog]:
        """Find tables with collapsed-<br> headers and send them to the LLM.

        Only tables that pass the _has_collapsed_br_headers() heuristic are
        submitted; all others pass through unchanged.

        Every submitted table is recorded in log.table_pairs with
        task="table_header".  Failed-validation entries use output=None.
        """
        lines = content.split("\n")
        table_line_ranges: list[tuple[int, int, list[str]]] = []
        task_flags: list[bool] = []   # True = submit to LLM
        i = 0

        while i < len(lines):
            if lines[i].strip().startswith("|"):
                start_idx = i
                table_lines = [lines[i]]
                i += 1
                while i < len(lines) and (
                    lines[i].strip().startswith("|") or not lines[i].strip()
                ):
                    if lines[i].strip():
                        table_lines.append(lines[i])
                    i += 1
                table_text = "\n".join(table_lines)
                table_line_ranges.append((start_idx, i, table_lines))
                task_flags.append(self._has_collapsed_br_headers(table_text))
            else:
                i += 1

        if not any(task_flags) or sum(task_flags) <= skip_existing:
            return content, log

        header_total = sum(task_flags) - skip_existing
        header_done = 0
        if progress is not None:
            progress.update(f"table headers queued: {header_total}")

        # Process table-header submissions sequentially so progress and pair-logs
        # are persisted immediately for each completed item.
        remaining = skip_existing
        results_by_index: list[Optional[str]] = [None] * len(table_line_ranges)
        for idx, ((start_idx, end_idx, tbl), flag) in enumerate(zip(table_line_ranges, task_flags)):
            if flag and remaining > 0:
                remaining -= 1
                continue
            if flag:
                res = await self.llm.enhance_async("\n".join(tbl), "table_header")
                llm_output_str: Optional[str] = res if isinstance(res, str) and res else None

                original_text = "\n".join(tbl)
                valid = (
                    llm_output_str is not None
                    and validate_enhancement(original_text, llm_output_str)
                )

                results_by_index[idx] = llm_output_str if valid else None

                log.table_pairs.append({
                    "task": "table_header",
                    "input": original_text,
                    "output": llm_output_str if valid else None,
                })

                header_done += 1
                if progress is not None:
                    progress.update(f"table headers {header_done}/{header_total}")
                if progress_callback is not None:
                    # Reconstruct incremental content from logged pairs so far
                    snapshot = apply_logged_enhancements(content, log)
                    await progress_callback(snapshot, log)

        result_lines: list[str] = []
        i = 0
        remaining = skip_existing
        for (start_idx, end_idx, orig_lines), flag, out_result in zip(
            table_line_ranges, task_flags, results_by_index
        ):
            while i < start_idx:
                result_lines.append(lines[i])
                i += 1

            original_text = "\n".join(orig_lines)

            if flag and remaining > 0:
                result_lines.extend(orig_lines)
                remaining -= 1
            elif flag:
                llm_output_str = out_result
                valid = (
                    llm_output_str is not None
                    and validate_enhancement(original_text, llm_output_str)
                )

                log.table_pairs.append({
                    "task": "table_header",
                    "input": original_text,
                    "output": llm_output_str if valid else None,
                })

                if valid:
                    result_lines.append(llm_output_str)
                else:
                    result_lines.extend(orig_lines)

                if progress_callback is not None:
                    snapshot = "\n".join(result_lines + lines[i:])
                    await progress_callback(snapshot, log)
            else:
                result_lines.extend(orig_lines)

            i = end_idx

        while i < len(lines):
            result_lines.append(lines[i])
            i += 1

        return "\n".join(result_lines), log

    # ── Math enhancer ─────────────────────────────────────────────────────────

    async def _enhance_math_async(
        self,
        content: str,
        log: LLMPairLog,
        progress: Optional[FileProgressReporter] = None,
        progress_callback: Optional[Callable[[str, LLMPairLog], Awaitable[None]]] = None,
        skip_existing: int = 0,
    ) -> tuple[str, LLMPairLog]:
        """Enhance all math blocks in parallel; record every pair in log.math_pairs.

        Entries where validation fails are kept with output=None in the log,
        and the original text is retained in the markdown.
        """
        math_pattern = r"(\$\$[^$]*\$\$|\$[^$\n]+\$)"
        matches = list(re.finditer(math_pattern, content))
        if skip_existing >= len(matches):
            return content, log

        remaining_matches = matches[skip_existing:]
        if not remaining_matches:
            return content, log

        if progress is not None:
            progress.update(f"math queued: {len(remaining_matches)}")

        # Process math enhancements sequentially to ensure ordering and immediate
        # persistence of each math pair (helps resume after interruption).
        results_by_index: list[Optional[str]] = [None] * len(remaining_matches)
        offset = 0
        completed = 0
        # Keep the original content snapshot for apply_logged_enhancements
        original_content = content
        for idx, match in enumerate(remaining_matches):
            original = match.group(1)
            res = await self.llm.enhance_async(original, "math")
            llm_output_str: Optional[str] = res if isinstance(res, str) and res else None

            valid = (
                llm_output_str is not None
                and validate_enhancement(original, llm_output_str)
            )

            # Append in-document-order so log.math_pairs remains ordered
            log.math_pairs.append({
                "input": original,
                "output": llm_output_str if valid else None,
            })

            # Apply replacement immediately if valid
            if valid:
                content, offset = _replace_next_occurrence(content, original, llm_output_str, offset)

            completed += 1
            if progress is not None:
                progress.update(f"math {completed}/{len(remaining_matches)}")
            if progress_callback is not None:
                await progress_callback(content, log)

        return content, log

    # ── General table enhancer ────────────────────────────────────────────────

    async def _enhance_tables_async(
        self,
        content: str,
        log: LLMPairLog,
        progress: Optional[FileProgressReporter] = None,
        progress_callback: Optional[Callable[[str, LLMPairLog], Awaitable[None]]] = None,
        skip_existing: int = 0,
    ) -> tuple[str, LLMPairLog]:
        """Enhance all tables in parallel; record every pair in log.table_pairs.

        Entries where validation fails are kept with output=None in the log,
        and the original text is retained in the markdown.
        """
        lines = content.split("\n")
        table_line_ranges: list[tuple[int, int, list[str]]] = []
        i = 0

        while i < len(lines):
            if lines[i].strip().startswith("|"):
                start_idx = i
                table_lines = [lines[i]]
                i += 1
                while i < len(lines) and (
                    lines[i].strip().startswith("|") or not lines[i].strip()
                ):
                    if lines[i].strip():
                        table_lines.append(lines[i])
                    i += 1
                table_line_ranges.append((start_idx, i, table_lines))
            else:
                i += 1

        if not table_line_ranges or skip_existing >= len(table_line_ranges):
            return content, log

        total_pending = len(table_line_ranges) - skip_existing
        if progress is not None:
            progress.update(f"tables queued: {total_pending}")

        # Process general tables sequentially so each table log and content
        # snapshot is saved after it completes.
        results_by_index: list[Optional[str]] = [None] * len(table_line_ranges)
        remaining_skips = skip_existing
        result_lines: list[str] = []
        i = 0
        processed_index = 0
        for (start_idx, end_idx, orig_lines) in table_line_ranges:
            while i < start_idx:
                result_lines.append(lines[i])
                i += 1

            original_text = "\n".join(orig_lines)
            if remaining_skips > 0:
                result_lines.extend(orig_lines)
                remaining_skips -= 1
            else:
                res = await self.llm.enhance_async(original_text, "table")
                llm_output_str: Optional[str] = res if isinstance(res, str) and res else None
                valid = (
                    llm_output_str is not None
                    and validate_enhancement(original_text, llm_output_str)
                )

                log.table_pairs.append({
                    "task": "table",
                    "input": original_text,
                    "output": llm_output_str if valid else None,
                })

                if valid:
                    result_lines.append(llm_output_str)
                else:
                    result_lines.extend(orig_lines)

                processed_index += 1
                if progress is not None:
                    progress.update(f"tables {processed_index}/{total_pending}")
                if progress_callback is not None:
                    snapshot = "\n".join(result_lines + lines[i:])
                    await progress_callback(snapshot, log)

            i = end_idx

        while i < len(lines):
            result_lines.append(lines[i])
            i += 1

        return "\n".join(result_lines), log

    # ── Code enhancer (available but not wired into main pipeline by default) ─

    async def _enhance_code_async(
        self, content: str, log: LLMPairLog
    ) -> tuple[str, LLMPairLog]:
        """Enhance all code blocks in parallel, validate each result."""
        code_pattern = r"```[\w-]*\n(.*?)\n```"
        matches = list(re.finditer(code_pattern, content, flags=re.DOTALL))
        if not matches:
            return content, log

        tasks = [self.llm.enhance_async(m.group(0), "code") for m in matches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        offset = 0
        for match, result in zip(matches, results):
            if not isinstance(result, str) or not result:
                continue
            original = match.group(0)
            if not validate_enhancement(original, result):
                continue
            start = match.start() + offset
            end = match.end() + offset
            content = content[:start] + result + content[end:]
            offset += len(result) - len(original)

        return content, log

    # ── List enhancer (available but not wired into main pipeline by default) ─

    async def _enhance_lists_async(
        self, content: str, log: LLMPairLog
    ) -> tuple[str, LLMPairLog]:
        """Enhance all list blocks in parallel, validate each result."""
        lines = content.split("\n")
        list_line_ranges: list[tuple[int, int, list[str]]] = []
        tasks = []
        i = 0

        while i < len(lines):
            if re.match(r"^\s*[-*+]\s+", lines[i]):
                start_idx = i
                list_lines = [lines[i]]
                i += 1
                while i < len(lines) and (
                    re.match(r"^\s*[-*+]\s+", lines[i])
                    or (lines[i] and lines[i][0] == " ")
                ):
                    list_lines.append(lines[i])
                    i += 1
                list_line_ranges.append((start_idx, i, list_lines))
                tasks.append(self.llm.enhance_async("\n".join(list_lines), "list"))
            else:
                i += 1

        if not tasks:
            return content, log

        results = await asyncio.gather(*tasks, return_exceptions=True)

        result_lines: list[str] = []
        i = 0
        for (start_idx, end_idx, orig_lines), res in zip(list_line_ranges, results):
            while i < start_idx:
                result_lines.append(lines[i])
                i += 1
            original_text = "\n".join(orig_lines)
            if isinstance(res, str) and res and validate_enhancement(original_text, res):
                result_lines.append(res)
            else:
                result_lines.extend(orig_lines)
            i = end_idx

        while i < len(lines):
            result_lines.append(lines[i])
            i += 1

        return "\n".join(result_lines), log


# ─────────────────────────────────────────────────────────────────────────────
# JSON pair output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _json_path_for(out_file: Path, prefix: str) -> Path:
    """Build the sibling JSON path for a given output markdown file.

    Example:
        out_file = /out/subdir/chapter1.md
        prefix   = "Formulas"
        → /out/subdir/Formulas_chapter1.json
    """
    return out_file.parent / f"{prefix}_{out_file.stem}.json"


def write_pair_logs(out_file: Path, log: LLMPairLog) -> None:
    """Write Formulas_<stem>.json and Tables_<stem>.json next to out_file.

    Both files are always written (empty array when no elements were found).
    """
    formulas_path = _json_path_for(out_file, "Formulas")
    tables_path   = _json_path_for(out_file, "Tables")

    formulas_path.write_text(
        json.dumps(log.math_pairs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tables_path.write_text(
        json.dumps(log.table_pairs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_pair_logs(out_file: Path) -> LLMPairLog:
    """Load existing per-file LLM pair logs for resume support."""
    formulas_path = _json_path_for(out_file, "Formulas")
    tables_path = _json_path_for(out_file, "Tables")
    math_pairs: list[dict] = []
    table_pairs: list[dict] = []

    if formulas_path.exists():
        try:
            math_pairs = json.loads(formulas_path.read_text(encoding="utf-8"))
        except Exception:
            math_pairs = []

    if tables_path.exists():
        try:
            table_pairs = json.loads(tables_path.read_text(encoding="utf-8"))
        except Exception:
            table_pairs = []

    return LLMPairLog(math_pairs=math_pairs, table_pairs=table_pairs)


def _replace_next_occurrence(
    content: str,
    original: str,
    replacement: str,
    start: int = 0,
) -> tuple[str, int]:
    """Replace the next occurrence of original with replacement, returning new content and next search index."""
    idx = content.find(original, start)
    if idx == -1:
        idx = content.find(original)
    if idx == -1:
        return content, start

    content = content[:idx] + replacement + content[idx + len(original):]
    return content, idx + len(replacement)


def apply_logged_enhancements(content: str, log: LLMPairLog) -> str:
    """Reconstruct content by applying previously recorded LLM replacements."""
    cursor = 0
    for entry in [e for e in log.table_pairs if e["task"] == "table_header" and e.get("output")]:
        content, cursor = _replace_next_occurrence(content, entry["input"], entry["output"], cursor)

    cursor = 0
    for entry in log.math_pairs:
        if not entry.get("output"):
            continue
        content, cursor = _replace_next_occurrence(content, entry["input"], entry["output"], cursor)

    cursor = 0
    for entry in [e for e in log.table_pairs if e["task"] == "table" and e.get("output")]:
        content, cursor = _replace_next_occurrence(content, entry["input"], entry["output"], cursor)

    return content


def write_progress_state(out_file: Path, log: LLMPairLog, content: str) -> None:
    """Persist intermediate file progress and pair logs for resume support."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(content, encoding="utf-8")
    write_pair_logs(out_file, log)


# ─────────────────────────────────────────────────────────────────────────────
# File-level processing
# ─────────────────────────────────────────────────────────────────────────────

async def process_file_async(
    md_file: Path,
    md_root: Path,
    out_root: Path,
    enhancer: AsyncMarkdownEnhancer,
    dry_run: bool = False,
    overwrite: bool = False,
) -> tuple[bool, Optional[str]]:
    """Read, enhance, write a single markdown file and its two JSON pair logs."""
    rel_path = md_file.relative_to(md_root)
    out_file = out_root / rel_path
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(
            None,
            lambda: md_file.read_text(encoding="utf-8", errors="replace"),
        )

        existing_log = None
        if out_file.exists():
            existing_log = load_pair_logs(out_file)

        async def save_progress(current_content: str, log: LLMPairLog) -> None:
            if dry_run:
                return
            await loop.run_in_executor(
                None,
                lambda: write_progress_state(out_file, log, current_content),
            )

        enhanced, log = await enhancer.enhance_content_async(
            md_file,
            content,
            dry_run=dry_run,
            existing_log=existing_log,
            progress_callback=save_progress,
        )

        if not dry_run:
            await loop.run_in_executor(
                None,
                lambda: write_progress_state(out_file, log, enhanced),
            )

        return (True, None)

    except Exception as e:
        return (False, str(e))


async def process_files_concurrent(
    md_files: list[Path],
    md_root: Path,
    out_root: Path,
    enhancer: AsyncMarkdownEnhancer,
    workers: int = 1,
    dry_run: bool = False,
    overwrite: bool = False,
    debug: bool = False,
    checkpoint_file: Optional[Path] = None,
) -> EnhancementStats:
    """Process all files concurrently, bounded by a semaphore."""
    stats = EnhancementStats()
    start_time = time.time()
    semaphore = asyncio.Semaphore(workers)
    total_files = len(md_files)
    checkpoint_lock = asyncio.Lock()
    completed_files = _load_checkpoint(checkpoint_file) if checkpoint_file else set()

    async def bounded_process(md_file: Path) -> None:
        rel_path = str(md_file.relative_to(md_root))
        if rel_path in completed_files and not overwrite:
            stats.skipped += 1
            print(
                f"Skipped {stats.enhanced + stats.skipped + stats.failed + 1}/{total_files}: {out_root / md_file.relative_to(md_root)}",
                flush=True,
            )
            return

        async with semaphore:
            success, error = await process_file_async(
                md_file, md_root, out_root, enhancer, dry_run, overwrite
            )
            if success:
                if error == "skipped":
                    stats.skipped += 1
                    out_file = out_root / md_file.relative_to(md_root)
                    completed_files.add(rel_path)
                    if checkpoint_file:
                        async with checkpoint_lock:
                            _write_checkpoint(checkpoint_file, completed_files)
                    print(
                        f"Skipped {stats.enhanced + stats.skipped + stats.failed}/{total_files}: {out_file}",
                        flush=True,
                    )
                else:
                    stats.enhanced += 1
                    out_file = out_root / md_file.relative_to(md_root)
                    completed_files.add(rel_path)
                    if checkpoint_file:
                        async with checkpoint_lock:
                            _write_checkpoint(checkpoint_file, completed_files)
                    print(
                        f"Completed {stats.enhanced + stats.skipped + stats.failed}/{total_files}: {out_file}",
                        flush=True,
                    )
            else:
                stats.failed += 1
                if debug:
                    print(f"ERROR processing {md_file}: {error}", file=sys.stderr, flush=True)

    tasks = [bounded_process(f) for f in md_files]
    completed = 0
    for future in asyncio.as_completed(tasks):
        await future
        completed += 1
        if completed % 10 == 0 or completed == len(md_files):
            print(f"Progress: {completed}/{len(md_files)} files processed", flush=True)

    stats.total_time = time.time() - start_time
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parallel markdown enhancement using LLM API (async)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--markdown-root",
        default="./input_markdown",
        help="Root folder containing markdown files (recursive scan).",
    )
    parser.add_argument(
        "--out-root",
        default="./output_markdown",
        help="Root folder for improved markdown output.",
    )
    parser.add_argument(
        "--pattern",
        default="**/*.md",
        help="Glob pattern for markdown files (default: **/*.md).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Max files to process (0 = all).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the remote model provider. If omitted, uses YUNWU_API_KEY or OPENAI_API_KEY from the environment.",
    )
    parser.add_argument(
        "--api-base-url",
        default=YUNWU_BASE_URL,
        help="LLM API base URL.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODELS[0],
        help="LLM model name.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent file processing tasks (default: 16 for higher throughput).",
    )
    parser.add_argument(
        "--batch-enhancement",
        type=bool,
        default=False,
        help="Use batch enhancement mode (default: False).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be changed without modifying files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess files even if output already exists.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info.",
    )

    args = parser.parse_args()

    md_root = Path(args.markdown_root)
    out_root = Path(args.out_root)

    if not md_root.exists():
        print(f"ERROR: --markdown-root not found: {md_root}", file=sys.stderr)
        return 2

    out_root.mkdir(parents=True, exist_ok=True)

    md_files = sorted(md_root.glob(args.pattern))
    if not md_files:
        md_files = sorted(md_root.rglob("*.md"))

    if not md_files:
        print(f"ERROR: No .md files found under: {md_root}", file=sys.stderr)
        return 2

    if args.max_files > 0:
        md_files = md_files[: args.max_files]

    print(f"Found {len(md_files)} markdown file(s)")
    print(f"Using {args.workers} concurrent workers")

    print(f"\nConnecting to API at {args.api_base_url}...")
    llm = AsyncAPIClient(base_url=args.api_base_url, model=args.model, api_key=args.api_key)

    if not llm.session_ready:
        return 2

    enhancer = AsyncMarkdownEnhancer(llm, batch_enhancement=args.batch_enhancement)
    checkpoint_file = _checkpoint_path_for(out_root)

    print(f"Checkpoint file   : {checkpoint_file}")
    print("Rerun the same command without changing the output path to continue from completed files.")

    print(f"\nStarting enhancement of {len(md_files)} files...")
    print(
        "Each file will produce three outputs:\n"
        "  <stem>.md                 – enhanced markdown\n"
        "  Formulas_<stem>.json      – LLM input/output pairs for math elements\n"
        "  Tables_<stem>.json        – LLM input/output pairs for tables\n"
        "Outputs are written as each file completes, so you can inspect finished files while the run is still active.\n"
    )

    stats = asyncio.run(
        process_files_concurrent(
            md_files,
            md_root,
            out_root,
            enhancer,
            workers=args.workers,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            debug=args.debug,
            checkpoint_file=checkpoint_file,
        )
    )

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  {stats.summary()}")
    print(f"  Total time: {stats.total_time:.2f}s")
    print(f"\nImproved files saved to: {out_root}")

    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())