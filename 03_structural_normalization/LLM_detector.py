#!/usr/bin/env python3
"""LLM-based detector for document heading format.

For each markdown file in the input directory, prints exactly one word per line:
  numbered
  non-numbered

Defaults to Ollama at http://localhost:11434 using model "llama3.1:70b".
If the LLM call fails or returns an unexpected answer, the script falls back to a simple heuristic.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Literal

import urllib.error
import urllib.request


# ──────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION
# All three values below can also be set via environment variable or CLI
# flag (CLI flag takes priority, then env var, then the default here).
# ──────────────────────────────────────────────────────────────────────────
DEFAULT_INPUT_DIR = "./data/03_structural_normalization/01_cleaned"
DEFAULT_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:70b")
# ──────────────────────────────────────────────────────────────────────────


HeadingCategory = Literal["numbered", "non-numbered"]


ATX_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
SETEXT_UNDERLINE_RE = re.compile(r"^\s*(=+|-+)\s*$")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
NUMERIC_HEADING_RE = re.compile(r"^\s{0,3}(?P<num>\d+(?:\.\d+)*)(?P<sep>[.)]|[.:-])?\s+(?P<title>\S.*)$")


def _toggle_fence(in_fence: str | None, line: str) -> str | None:
    m = FENCE_RE.match(line)
    if not m:
        return in_fence
    fence = m.group(1)
    if in_fence is None:
        return fence
    if fence[0] == in_fence[0]:
        return None
    return in_fence


def extract_heading_matches(markdown: str) -> List[str]:
    lines = markdown.splitlines()
    in_fence: str | None = None
    headings: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        in_fence = _toggle_fence(in_fence, line)
        if in_fence is not None:
            i += 1
            continue
        atx = ATX_HEADING_RE.match(line)
        if atx:
            headings.append(atx.group(2).strip())
            i += 1
            continue
        if i + 1 < len(lines) and lines[i].strip() and SETEXT_UNDERLINE_RE.match(lines[i + 1]):
            headings.append(lines[i].strip())
            i += 2
            continue
        numeric = NUMERIC_HEADING_RE.match(line)
        if numeric:
            headings.append(line.strip())
        i += 1
    return headings


def heuristic_heading_category(heading_text: str) -> HeadingCategory:
    s = heading_text.strip()
    s = re.sub(r'(?i)^\s*<(?:strong|b|em|i)>\s*', '', s)
    s = re.sub(r'(?i)\s*</(?:strong|b|em|i)>\s*$', '', s)
    s = re.sub(r'^[\*_`"\']+', '', s)
    s = re.sub(r'[\*_`"\']+$', '', s)
    s = re.sub(r'^[\s>\-\*]+', '', s)
    return "numbered" if re.match(r"^\s*\d+(?:\.\d+)*\b", s) else "non-numbered"


def _strip_heading_markup(heading_text: str) -> str:
    s = heading_text.strip()
    s = re.sub(r'(?i)^\s*<(?:strong|b|em|i)>\s*', '', s)
    s = re.sub(r'(?i)\s*</(?:strong|b|em|i)>\s*$', '', s)
    s = re.sub(r'^[\*_`"\']+', '', s)
    s = re.sub(r'[\*_`"\']+$', '', s)
    s = re.sub(r'^[\s>\-\*]+', '', s)
    return s.strip()


def heuristic_document_category(headings: List[str]) -> HeadingCategory:
    numbered_headings = [h for h in headings if heuristic_heading_category(h) == "numbered"]
    stripped_numbered = [_strip_heading_markup(h) for h in numbered_headings]
    top_level_numbered = [h for h in stripped_numbered if re.match(r"^\s*\d+\b", h)]
    nested_numbered = [h for h in stripped_numbered if re.match(r"^\s*\d+\.\d+", h)]

    if len(top_level_numbered) >= 3 and (nested_numbered or len(numbered_headings) >= 5):
        return "numbered"

    non_numbered = len(headings) - len(numbered_headings)
    return "numbered" if len(numbered_headings) >= max(1, non_numbered) else "non-numbered"


def _ollama_generate(base_url: str, model: str, prompt: str, timeout: int = 30) -> str | None:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
        for k in ("response", "text", "result", "output"):
            if k in parsed:
                return str(parsed[k])
        return body
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def classify_with_llm(headings: List[str], base_url: str, model: str, timeout: int = 90) -> HeadingCategory | None:
    numbered_examples = [h for h in headings if heuristic_heading_category(h) == "numbered"][:6]

    prompt_lines = [
        "You are a markdown heading format classifier for each document not each heading.",
        "Classify the document as numbered or non-numbered based on the main body structure, not title pages or front matter.",
        "Ignore contents pages, disclaimers, notes, exceptions, and informative-note headings.",
        "A document is numbered when the body uses a numeric hierarchy such as 1, 2, 3, 3.1, 5.13.2, 7.2, or 7.1.4.",
        "If the document contains section-style numbered headings like 1. PURPOSE, 2. SCOPE, 3. DEFINITIONS, 7.2 System Start-Up, and 7.1.4 Protection of Occupied Areas, answer numbered.",
        'Return exactly one token: "numbered" or "non-numbered". Do not explain your answer.',
        "",
    ]
    for h in numbered_examples:
        prompt_lines.append(f"- {h}")

    prompt = "\n".join(prompt_lines)
    raw = _ollama_generate(base_url, model, prompt, timeout=timeout)
    if not raw:
        return None
    low = raw.strip().lower()
    match = re.search(r"\b(numbered|non-numbered|non numbered)\b", low)
    if match:
        token = match.group(1)
        return "non-numbered" if token.startswith("non") else "numbered"
    token = re.sub(r"[^a-zA-Z\- ]+", " ", low).strip()
    if token.startswith("numbered"):
        return "numbered"
    if token.startswith("non-numbered") or token.startswith("non numbered"):
        return "non-numbered"
    return None


def run(input_dir: Path, base_url: str, model: str, llm_only: bool = False, timeout_seconds: int = 90) -> int:
    files = sorted(f for f in input_dir.iterdir() if f.suffix.lower() == ".md")
    if not files:
        print("No markdown files found", file=sys.stderr)
        return 1

    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        headings = extract_heading_matches(text)
        llm_choice = classify_with_llm(headings, base_url, model, timeout=timeout_seconds)
        if llm_choice is None:
            if llm_only:
                llm_choice = classify_with_llm(headings, base_url, model, timeout=timeout_seconds * 2)
                choice = llm_choice if llm_choice is not None else "unknown"
            else:
                choice = heuristic_document_category(headings)
        else:
            choice = llm_choice
        print(choice)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="LLM detector: print one word (numbered/non-numbered) per .md file in input dir")
    p.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_BASE_URL)
    p.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    p.add_argument("--llm-only", action="store_true", help="Use only the LLM for decisions; do not fall back to heuristics")
    p.add_argument("--timeout-seconds", type=int, default=90, help="LLM decision timeout in seconds (default 90)")
    args = p.parse_args()
    return run(Path(args.input_dir), args.ollama_url, args.model, llm_only=args.llm_only, timeout_seconds=args.timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
