#!/usr/bin/env python3
"""Parallel PDF to Markdown conversion using marker-pdf with ProcessPoolExecutor.

This version optimizes PDF conversion using multi-core parallelism.

Example:
    python convert_pdfs_to_markdown.py \\
        --pdf-root ./data/01_pdf_collection \\
        --out-root ./data/02_markdown_extraction \\
        --workers 4 \\
        --gpus "0,1,2,3"

    # Single PDF test:
    python convert_pdfs_to_markdown.py \\
        --pdf ./data/01_pdf_collection/example.pdf \\
        --out-root /tmp/test_out

    # With GPU auto-selection:
    python convert_pdfs_to_markdown.py \\
        --pdf-root ./data/01_pdf_collection \\
        --workers 4 --gpus-num 4
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from shutil import which
import itertools
import concurrent.futures
from typing import Optional
import time


# ──────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION
# Defaults below are relative to wherever you run this script from. Override
# any of them with the matching CLI flag instead of editing this file.
# ──────────────────────────────────────────────────────────────────────────
DEFAULT_PDF_ROOT = "./data/01_pdf_collection"     # folder of source PDFs
DEFAULT_OUT_ROOT = "./data/02_markdown_extraction"  # where converted .md + images go
# ──────────────────────────────────────────────────────────────────────────


def _sanitize_path_component(name: str, max_len: int = 160) -> str:
    """Sanitize filename for filesystem compatibility."""
    name = name.strip()
    if not name:
        return "untitled"
    name = re.sub(r"[^A-Za-z0-9._()\- +]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name or "untitled"


def _has_markdown_files(folder: Path) -> bool:
    """Check if folder contains markdown files."""
    try:
        return any(p.suffix.lower() in {".md", ".markdown"} for p in folder.rglob("*"))
    except Exception:
        return False


def _validate_md_output(out_dir: Path) -> list[str]:
    """Return list of warnings for converted markdown."""
    warnings: list[str] = []

    md_files = sorted(out_dir.rglob("*.md"))
    if not md_files:
        warnings.append("No .md files produced")
        return warnings

    content = md_files[0].read_text(encoding="utf-8", errors="ignore")

    lines = content.splitlines()
    for i, line in enumerate(lines, 1):
        if line.startswith("|") and not line.rstrip().endswith("|"):
            warnings.append(f"Possibly broken table row at line {i}")
            break

    if "|" not in content:
        warnings.append("No Markdown tables detected")

    # Check for math
    has_inline_math = re.search(r"\$[^$\n]{1,120}\$", content) is not None
    has_block_math = re.search(r"\$\$[\s\S]{1,800}?\$\$", content) is not None
    has_math = has_inline_math or has_block_math
    if not has_math:
        looks_mathy = re.search(r"(Δ|≤|≥|≠|≈|√|∑|∫|\b[A-Za-z]{1,4}\s*=\s*[A-Za-z0-9(])", content)
        if looks_mathy:
            warnings.append("No LaTeX math detected")

    return warnings


def _pick_gpus_by_free_memory(limit: int) -> list[str]:
    """Return GPU indices sorted by free memory."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        pairs: list[tuple[int, int]] = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            gpu_index_text, free_mem_text = [part.strip() for part in line.split(",", 1)]
            pairs.append((int(gpu_index_text), int(free_mem_text)))
        pairs.sort(key=lambda item: item[1], reverse=True)
        return [str(index) for index, _free in pairs[:limit]]
    except Exception:
        return [str(i) for i in range(max(1, limit))]


def _run_marker_job(args: tuple) -> dict:
    """Run a single marker job. Designed to be called by ProcessPoolExecutor."""
    idx, pdf_path, out_dir, cmd, gpu, timeout_sec = args

    start_time = time.time()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu

    try:
        completed = subprocess.run(
            cmd,
            check=True,
            timeout=max(1, int(timeout_sec)),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Validate output
        warnings = _validate_md_output(out_dir)
        elapsed = time.time() - start_time

        return {
            "status": "success",
            "idx": idx,
            "pdf_path": str(pdf_path),
            "warnings": warnings,
            "elapsed": elapsed,
            "gpu": gpu,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    except subprocess.TimeoutExpired as e:
        return {
            "status": "timeout",
            "idx": idx,
            "pdf_path": str(pdf_path),
            "error": str(e),
            "elapsed": time.time() - start_time,
        }
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "idx": idx,
            "pdf_path": str(pdf_path),
            "error": str(e),
            "stderr": getattr(e, "stderr", ""),
            "stdout": getattr(e, "stdout", ""),
            "elapsed": time.time() - start_time,
        }
    except Exception as e:
        return {
            "status": "error",
            "idx": idx,
            "pdf_path": str(pdf_path),
            "error": str(e),
            "elapsed": time.time() - start_time,
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parallel batch convert PDFs to Markdown with marker-pdf"
    )
    parser.add_argument(
        "--pdf",
        dest="pdf_paths",
        nargs="+",
        default=None,
        help="Convert one or more explicit PDF files (overrides --pdf-root scan).",
    )
    parser.add_argument(
        "--pdf-root",
        dest="pdf_root",
        default=DEFAULT_PDF_ROOT,
        help="Root folder containing PDFs (searched recursively).",
    )
    parser.add_argument(
        "--out-root",
        default=DEFAULT_OUT_ROOT,
        help="Root folder to write per-PDF marker outputs.",
    )
    parser.add_argument("--langs", default="English", help="Language(s)")
    parser.add_argument(
        "--marker-cmd",
        default="marker_single",
        help="marker executable to run (default: marker_single)",
    )
    parser.add_argument(
        "--output-format",
        default="markdown",
        choices=["markdown", "json", "html"],
        help="Output format",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=10800,
        help="Timeout per PDF conversion (seconds).",
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=0,
        help="Process at most N PDFs (0 = all).",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR on all pages.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4)",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU indices (e.g., '0,1,2,3'). Auto-select if not provided.",
    )
    parser.add_argument(
        "--gpus-num",
        type=int,
        default=4,
        help="Number of GPUs to auto-select by free memory (default: 4).",
    )

    args = parser.parse_args()

    pdf_root = Path(args.pdf_root)
    out_root = Path(args.out_root)

    if which(args.marker_cmd) is None:
        print(
            f"ERROR: '{args.marker_cmd}' not found in PATH. "
            f"Install it with: pip install marker-pdf",
            file=sys.stderr,
        )
        return 2

    if args.pdf_paths:
        pdfs = []
        for pdf_text in args.pdf_paths:
            pdf_path = Path(pdf_text)
            if not (pdf_path.exists() and pdf_path.is_file()):
                print(f"ERROR: --pdf not found: {pdf_path}", file=sys.stderr)
                return 2
            pdfs.append(pdf_path)
    else:
        if not pdf_root.exists():
            print(f"ERROR: pdf-root not found: {pdf_root}", file=sys.stderr)
            return 2

        pdfs = sorted({*pdf_root.rglob("*.pdf"), *pdf_root.rglob("*.PDF")})
        if not pdfs:
            print(f"ERROR: No PDFs found under: {pdf_root}", file=sys.stderr)
            return 2
        if args.max_pdfs and args.max_pdfs > 0:
            pdfs = pdfs[: args.max_pdfs]

    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(pdfs)} PDFs under: {pdf_root}")
    print(f"Output root: {out_root}")
    print(f"Workers: {args.workers}")

    # Detect marker_single CLI flags for version compatibility
    help_proc = subprocess.run([args.marker_cmd, "--help"], capture_output=True, text=True)
    help_text = (help_proc.stdout or "") + "\n" + (help_proc.stderr or "")

    supports_output_dir = ("--output_dir" in help_text) or ("--output-dir" in help_text)
    output_dir_flag = "--output_dir" if "--output_dir" in help_text else "--output-dir"
    supports_output_format = ("--output_format" in help_text) or ("--output-format" in help_text)
    output_format_flag = "--output_format" if "--output_format" in help_text else "--output-format"
    supports_langs = "--langs" in help_text
    supports_format_lines = ("--format_lines" in help_text) or ("--format-lines" in help_text)
    format_lines_flag = "--format_lines" if "--format_lines" in help_text else "--format-lines"
    supports_ocr_all_pages = ("--ocr_all_pages" in help_text) or ("--ocr-all-pages" in help_text)
    ocr_all_pages_flag = "--ocr_all_pages" if "--ocr_all_pages" in help_text else "--ocr-all-pages"
    supports_force_ocr = "--force_ocr" in help_text

    # Build job list
    jobs = []
    for i, pdf_path in enumerate(pdfs, 1):
        try:
            rel_dir = pdf_path.parent.relative_to(pdf_root)
        except ValueError:
            rel_dir = Path(".")

        out_dir = out_root / rel_dir / _sanitize_path_component(pdf_path.stem)
        out_dir.mkdir(parents=True, exist_ok=True)

        has_md = _has_markdown_files(out_dir)

        if has_md:
            print(f"[{i}/{len(pdfs)}] SKIP (already has .md): {pdf_path}")
            continue

        cmd = [args.marker_cmd, str(pdf_path)]

        if supports_output_dir:
            cmd += [output_dir_flag, str(out_dir)]
        else:
            cmd += [str(out_dir)]

        if supports_output_format:
            cmd += [output_format_flag, str(args.output_format)]

        if supports_langs:
            cmd += ["--langs", str(args.langs)]

        if supports_format_lines:
            cmd += [format_lines_flag]

        if args.force_ocr:
            if supports_ocr_all_pages:
                cmd += [ocr_all_pages_flag]
            elif supports_force_ocr:
                cmd += ["--force_ocr"]

        jobs.append((i, pdf_path, out_dir, cmd))

    if not jobs:
        print("All PDFs already converted. Nothing to do.")
        return 0

    print(f"\nNeed to convert {len(jobs)} PDFs")

    # GPU assignment
    if args.gpus:
        gpu_list = [g.strip() for g in args.gpus.split(",")]
        print(f"Using specified GPUs: {', '.join(gpu_list)}")
    else:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cuda_visible:
            gpu_list = [str(i) for i in range(len(cuda_visible.split(",")))]
            print(f"Using CUDA_VISIBLE_DEVICES={cuda_visible}")
        else:
            gpu_count = max(1, int(args.gpus_num))
            gpu_list = _pick_gpus_by_free_memory(gpu_count)
            print(f"Auto-selected top {len(gpu_list)} GPU(s): {', '.join(gpu_list)}")

    if not gpu_list:
        gpu_list = [str(i) for i in range(max(1, int(args.gpus_num)))]

    # Assign GPUs to jobs round-robin
    jobs_with_gpu = [
        (idx, pdf_path, out_dir, cmd, gpu, args.timeout_sec)
        for (idx, pdf_path, out_dir, cmd), gpu in zip(jobs, itertools.cycle(gpu_list))
    ]

    # Run jobs in parallel using ProcessPoolExecutor
    converted = 0
    failed = 0
    start_time = time.time()

    if args.dry_run:
        print("\n[DRY RUN] Would execute:")
        for idx, pdf_path, out_dir, cmd, gpu, timeout_sec in jobs_with_gpu:
            print(f"  [{idx}] GPU {gpu}: {' '.join(cmd)}")
        return 0

    print(f"\nStarting parallel conversion with {args.workers} workers...\n")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_marker_job, job): job
            for job in jobs_with_gpu
        }

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            result = future.result()

            idx = result["idx"]
            pdf_name = Path(result["pdf_path"]).name
            elapsed = result["elapsed"]

            if result["status"] == "success":
                converted += 1
                print(f"[{idx}/{len(jobs)}] \u2713 {pdf_name} ({elapsed:.1f}s)")
                for w in result.get("warnings", []):
                    print(f"     \u26a0\ufe0f  {w}")
            else:
                failed += 1
                error = result.get("error", "unknown")
                print(f"[{idx}/{len(jobs)}] \u2717 {pdf_name}: {result['status']} - {error}")
                stderr_text = (result.get("stderr") or "").strip()
                stdout_text = (result.get("stdout") or "").strip()
                if stderr_text:
                    print(f"     stderr: {stderr_text[:1000]}")
                if stdout_text:
                    print(f"     stdout: {stdout_text[:1000]}")

            if completed % 5 == 0 or completed == len(jobs):
                elapsed_total = time.time() - start_time
                rate = completed / elapsed_total if elapsed_total > 0 else 0
                print(f"  Progress: {completed}/{len(jobs)} ({rate:.1f} jobs/min)")

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Converted: {converted}")
    print(f"  Failed:    {failed}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg time per PDF: {total_time / max(1, len(jobs)):.1f}s")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
