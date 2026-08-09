# Figure Pipeline

# This directory contains a VLM-based figure enrichment pipeline for Markdown books.

## What it does

# The pipeline:

# 1. Detects image references in Markdown files.
# 2. Resolves the referenced files on disk.
# 3. Harvests nearby document context.
# 4. Sends the image and context to a vision-language model.
# 5. Writes a JSON registry describing the extracted figure information.

## Usage

### Dry run

# ```bash
# python "Testing_Code_VLM_Final_1 copy.py" --root ./input_images --output ./output --dry-run
# ```

### Live run

# export YUNWU_API_KEY="your-api-key"
# export YUNWU_API_BASE_URL="https://api.example.com/v1"
# export YUNWU_MODEL="gpt-5.4-nano"

# python "Testing_Code_VLM_Final_1 copy.py" --root ./input_images --output ./output
# ```

## Notes

# #- The script defaults to repo-threading.local folders: `./input_images` and `./output`.
# - The API key is read from the environment and is not stored in the script.
# - The generated output is JSON-based and suitable for downstream indexing or RAG workflows.




#!/usr/bin/env python3
"""
figure_pipeline.py
==================
Production-ready pipeline that enriches book Markdown files by:
  1. Detecting figure/image references in each book's Markdown.
  2. Resolving each reference to the actual image file on disk.
  3. Harvesting surrounding document context.
  4. Sending image + context to a Vision-Language model (Qwen2.5-VL).
  5. Inserting the extracted structured information back into the Markdown.
  6. Writing one enriched Markdown + one JSON registry per book folder.

Models are served through an Ollama API endpoint
(e.g. local Ollama server with qwen2.5vl models).

Usage
-----
  python figure_pipeline.py \
      --root ./books \
      --output ./output \
      --api-base http://127.0.0.1:11434/v1 \
      --primary-model qwen2.5vl:7b \
      --fallback-model qwen2.5vl:72b \
      --confidence-threshold 0.6 \\
      --workers 4

  # Dry-run (no model calls, no files written):
  python figure_pipeline.py --root ./books --dry-run
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import dataclasses
import json
import logging
import mimetypes
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

# ===========================================================================
# LOGGING
# ===========================================================================

YUNWU_BASE_URL = os.getenv("YUNWU_API_BASE_URL", "https://api.wlai.vip/v1")
DEFAULT_MODELS = (
    os.getenv("YUNWU_MODEL", "gpt-5.4-nano"),
)
DEFAULT_PRIMARY_MODEL = DEFAULT_MODELS[0]
DEFAULT_FALLBACK_MODEL = DEFAULT_MODELS[1] if len(DEFAULT_MODELS) > 1 else DEFAULT_MODELS[0]
API_REQUEST_SEMAPHORE: Optional[threading.Semaphore] = None

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("figure_pipeline")


# ===========================================================================
# CONFIGURATION  (all values have CLI-overridable defaults)
# ===========================================================================

@dataclasses.dataclass
class Config:
    root_dir: Path = Path("./input_images")
    output_dir: Path = Path("./output")

    # --- API ---
    api_base: str = YUNWU_BASE_URL
    api_key: str = os.getenv("YUNWU_API_KEY", "")
    models: Tuple[str, ...] = DEFAULT_MODELS
    max_tokens: int = 8000
    temperature: float = 0.0               # low temperature for determinism
    max_retries: int = 5                   # per-model retry count
    request_delay: float = 1.0             # seconds between API requests

    # --- Confidence ---
    confidence_threshold: float = 0.5      # use for multi-model voting

    # --- Context harvesting ---
    context_chars_before: int = 400        # characters of Markdown before figure
    context_chars_after: int = 300         # characters of Markdown after figure
    max_heading_search_chars: int = 3000   # look back this far for section heading

    # --- Parallelism ---
    workers: int = 1                       # concurrent book workers
    figure_workers: int = 5               # concurrent figure workers per book
    max_concurrent_requests: int = 20      # limit concurrent API calls to avoid rate limiting

    # --- Misc ---
    dry_run: bool = False
    force: bool = False
    skip_validation: bool = False
    log_level: str = "INFO"
    enriched_suffix: str = "_enriched.md"
    registry_suffix: str = "_figure_registry.json"

    # Supported image extensions (lowercased)
    image_extensions: Tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff",
    )

    # Minimum confidence to accept primary-model answer without field-level check
    required_fields: Tuple[str, ...] = (
        "figure_type", "extracted_text", "keywords", "confidence",
    )


# ===========================================================================
# DATA MODELS
# ===========================================================================

@dataclasses.dataclass
class FigureContext:
    """Surrounding textual context harvested from the Markdown."""
    section_heading: str = ""
    caption: str = ""
    text_before: str = ""
    text_after: str = ""
    nearby_equations: List[str] = dataclasses.field(default_factory=list)
    nearby_tables: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class FigureRef:
    """One image reference found in a Markdown file."""
    original_ref: str          # complete original markdown token  e.g. ![](_page_42_Figure_0.jpeg)
    rel_path: str              # path as written in the markdown
    resolved_path: Optional[Path]  # absolute path on disk (None if not found)
    char_offset: int           # character offset in the original markdown string
    page: Optional[int] = None
    figure_id: str = ""        # assigned ID like "Fig_42_0"


@dataclasses.dataclass
class FigureExtraction:
    """Everything produced for one figure after model inference."""
    figure_ref: FigureRef
    context: FigureContext
    model_used: str = ""
    raw_response: str = ""
    parsed: Dict[str, Any] = dataclasses.field(default_factory=dict)
    markdown_block: str = ""   # the block that replaces the original reference


# ===========================================================================
# SECTION A — FIGURE DETECTION & IMAGE RESOLUTION
# ===========================================================================

# Matches:  ![alt text](path)  and  ![](path)
_MD_IMAGE_RE = re.compile(
    r'!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)',
    re.MULTILINE,
)


def detect_figures(md_text: str, book_dir: Path, cfg: Config) -> List[FigureRef]:
    """
    Parse `md_text` and return one FigureRef per image reference.
    Resolution order for each reference path:
      1. Exact relative match from book_dir.
      2. Case-insensitive match inside book_dir (recursive).
      3. Filename-only match anywhere inside book_dir.
    """
    refs: List[FigureRef] = []
    seen: Dict[str, int] = {}   # rel_path → occurrence count for unique IDs

    for m in _MD_IMAGE_RE.finditer(md_text):
        rel = m.group("path").strip()
        ext = Path(rel).suffix.lower()

        # Only process image files, skip web URLs
        if rel.startswith("http://") or rel.startswith("https://"):
            continue
        if ext not in cfg.image_extensions:
            log.debug("Skipping non-image reference: %s", rel)
            continue

        resolved = _resolve_image_path(rel, book_dir, cfg)
        page_num = _extract_page_from_filename(Path(rel).name)

        # Build a stable figure_id from the filename
        stem = Path(rel).stem              # e.g. _page_42_Figure_0
        stem_clean = re.sub(r'^_+', '', stem)   # remove leading underscores
        count = seen.get(rel, 0)
        seen[rel] = count + 1
        suffix = f"_{count}" if count > 0 else ""
        figure_id = f"Fig_{stem_clean}{suffix}"

        refs.append(FigureRef(
            original_ref=m.group(0),
            rel_path=rel,
            resolved_path=resolved,
            char_offset=m.start(),
            page=page_num,
            figure_id=figure_id,
        ))

    return refs


def _resolve_image_path(rel: str, book_dir: Path, cfg: Config) -> Optional[Path]:
    """Try to locate the image file; return its absolute path or None."""
    # 1. Direct relative path
    candidate = (book_dir / rel).resolve()
    if candidate.is_file():
        return candidate

    # 2. Case-insensitive recursive search by filename
    target_name = Path(rel).name.lower()
    for f in book_dir.rglob("*"):
        if f.name.lower() == target_name and f.suffix.lower() in cfg.image_extensions:
            return f.resolve()

    # 3. Partial stem match (handles minor renames)
    stem = Path(rel).stem.lower()
    for f in book_dir.rglob("*"):
        if f.stem.lower() == stem and f.suffix.lower() in cfg.image_extensions:
            return f.resolve()

    return None


def _extract_page_from_filename(name: str) -> Optional[int]:
    """Parse page number from filenames like _page_42_Figure_0.jpeg."""
    m = re.search(r'page_(\d+)', name, re.IGNORECASE)
    return int(m.group(1)) if m else None


# ===========================================================================
# SECTION B — CONTEXT HARVESTING
# ===========================================================================

_HEADING_RE  = re.compile(r'^#{1,6}\s+.+', re.MULTILINE)
_CAPTION_RE  = re.compile(
    r'(?i)(?:^|\n)\s*(?:fig(?:ure)?\.?\s*\d[\d.\-]*|caption)\s*[:\-–]?\s*([^\n]{5,200})',
)
_EQUATION_RE = re.compile(r'\$\$[\s\S]+?\$\$|\$[^$\n]+\$', re.MULTILINE)
_TABLE_RE    = re.compile(r'(?:^\|.+\|\s*\n)+', re.MULTILINE)


def harvest_context(md_text: str, fig: FigureRef, cfg: Config) -> FigureContext:
    """
    Collect context surrounding a figure reference in the Markdown.
    Everything is heuristic and intentionally modular — replace/extend freely.
    """
    offset = fig.char_offset

    # ---- Text windows -------------------------------------------------------
    start_before = max(0, offset - cfg.context_chars_before)
    text_before   = md_text[start_before:offset]
    end_after     = min(len(md_text), offset + len(fig.original_ref) + cfg.context_chars_after)
    text_after    = md_text[offset + len(fig.original_ref):end_after]

    # ---- Nearest heading (look back) ----------------------------------------
    heading_search_start = max(0, offset - cfg.max_heading_search_chars)
    search_region = md_text[heading_search_start:offset]
    headings = _HEADING_RE.findall(search_region)
    section_heading = headings[-1].strip() if headings else ""

    # ---- Caption (look in a tight window around the figure) -----------------
    caption_window = md_text[max(0, offset - 300): end_after]
    cap_m = _CAPTION_RE.search(caption_window)
    caption = cap_m.group(1).strip() if cap_m else ""

    # ---- Nearby equations ---------------------------------------------------
    eq_window = md_text[max(0, offset - 500): end_after]
    equations = _EQUATION_RE.findall(eq_window)

    # ---- Nearby tables ------------------------------------------------------
    tbl_window = md_text[max(0, offset - 600): end_after]
    tables = _TABLE_RE.findall(tbl_window)

    return FigureContext(
        section_heading=section_heading,
        caption=caption,
        text_before=text_before.strip(),
        text_after=text_after.strip(),
        nearby_equations=[e.strip() for e in equations[:3]],
        nearby_tables=[t.strip() for t in tables[:2]],
    )


# ===========================================================================
# SECTION C — VISION-LANGUAGE MODEL INTERACTION
# ===========================================================================

_SYSTEM_PROMPT = """You are a technical figure-analysis and image-description assistant.

Analyze the provided figure image together with the surrounding document context.

Your goal is to produce highly informative, retrieval-friendly record for RAG.
Describe the image as if you were explaining it to a person who cannot see it.
Be exhaustive and precise.

- Do not omit visible text, numbers, labels, legends, titles, axis names, table headers, row labels, units, symbols, or annotations.
- If the figure contains multiple parts, describe each part.
- If the figure is a chart or table, capture the structure and the values as completely as possible.
- If the caption or context helps interpret the figure, incorporate that meaning into the description.


{
  "figure_id":               "<string — suggested label, e.g. Fig 4-3>",
  "figure_title":            "<string — short title for the figure>",
  "figure_name":             "<string — descriptive name, e.g. steam table>",
  "figure_type":             "<one of: chart, graph, table, diagram, equation, photo, illustration, other>",
  "caption_summary":         "<string — summary of the caption or null>",
  "extracted_text":          "<string — all text visible inside the figure>",
  "visual_elements":         ["<list of key visual elements>"],
  "keywords":                ["<keyword1>", "<keyword2>"],
  "confidence":              <float between 0 and 1>,
  "notes":                   "<string — comprehensive plain-language description of the entire figure, including the main purpose, layout, all visible components, and how they relate>",
  "source_page_if_inferable": <integer or null>
}

Guidance for fields:
- notes: explain what the reader should understand from the structure and values. write the most useful paragraph for RAG.
- extracted_text: include every readable word and number exactly as shown when possible.
"""

_USER_TEMPLATE = """Section heading: {section_heading}
Caption: {caption}
Text before figure:
{text_before}

Text after figure:
{text_after}

{equation_block}{table_block}

Please analyze the figure image and return a full RAG-friendly description.
Focus on: the main message, all visible text, all numbers, layout, labels, units, legends, and any relationships between elements.
If this is a chart, graph, or table, describe it as completely as possible.
If anything is unclear, say so explicitly."""


def _build_user_message(ctx: FigureContext) -> str:
    eq_block = (
        "Nearby equations:\n" + "\n".join(ctx.nearby_equations) + "\n\n"
        if ctx.nearby_equations else ""
    )
    tbl_block = (
        "Nearby tables:\n" + "\n".join(ctx.nearby_tables) + "\n\n"
        if ctx.nearby_tables else ""
    )
    return _USER_TEMPLATE.format(
        section_heading=ctx.section_heading or "(none)",
        caption=ctx.caption or "(none)",
        text_before=ctx.text_before or "(none)",
        text_after=ctx.text_after or "(none)",
        equation_block=eq_block,
        table_block=tbl_block,
    )


def _encode_image(image_path: Path) -> Tuple[str, str]:
    """Return (base64_data, mime_type) for an image file."""
    mime, _ = mimetypes.guess_type(str(image_path))
    mime = mime or "image/jpeg"
    with open(image_path, "rb") as fh:
        data = base64.standard_b64encode(fh.read()).decode("utf-8")
    return data, mime


def _call_vlm(
    model: str,
    img_data: str,
    img_mime: str,
    user_text: str,
    cfg: Config,
) -> str:
    """
    Call the Vision-Language model with image + context.
    Returns the raw text response from the model.
    Retries up to cfg.max_retries times on transient errors.
    """
    url = cfg.api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    if cfg.api_key and cfg.api_key != "not-required":
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img_mime};base64,{img_data}"},
                    },
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
    }

    data = json.dumps(payload).encode("utf-8")
    last_exc: Optional[Exception] = None
    for attempt in range(cfg.max_retries + 1):
        wait: Optional[float] = None
        if API_REQUEST_SEMAPHORE is not None:
            API_REQUEST_SEMAPHORE.acquire()
        try:
            if cfg.request_delay > 0:
                time.sleep(cfg.request_delay)
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req) as resp:
                body = resp.read().decode("utf-8")
            response = json.loads(body)
            result = response["choices"][0]["message"]["content"]
            return result
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429:
                wait = 15 * (2 ** attempt)
            else:
                wait = 2 ** attempt
            retry_after = None
            if exc.code == 429:
                retry_header = exc.headers.get("Retry-After") if hasattr(exc, "headers") else None
                if retry_header:
                    try:
                        retry_after = int(retry_header)
                    except ValueError:
                        pass
                wait = max(wait, retry_after or wait)
                log.warning(
                    "Model call failed (attempt %d/%d): HTTP 429 Too Many Requests — retrying in %ds",
                    attempt + 1,
                    cfg.max_retries + 1,
                    int(wait),
                )
            else:
                log.warning(
                    "Model call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1,
                    cfg.max_retries + 1,
                    exc,
                    int(wait),
                )
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as exc:
            last_exc = exc
            wait = 2 ** attempt
            log.warning(
                "Model call failed (attempt %d/%d): %s — retrying in %ds",
                attempt + 1,
                cfg.max_retries + 1,
                exc,
                int(wait),
            )
        finally:
            if API_REQUEST_SEMAPHORE is not None:
                API_REQUEST_SEMAPHORE.release()

        if wait is None:
            break
        if attempt >= cfg.max_retries:
            break
        time.sleep(wait + random.uniform(0, 1))

    raise RuntimeError(f"Model call failed after {cfg.max_retries + 1} attempts") from last_exc


def _infer_with_model(
    img_data: str,
    img_mime: str,
    user_text: str,
    cfg: Config,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Call the configured single model and return its parsed output."""
    if not cfg.models:
        raise ValueError("No model configured for inference")

    model = cfg.models[0]
    raw = _call_vlm(model, img_data, img_mime, user_text, cfg)
    parsed, is_valid = parse_model_output(raw)
    confidence = assess_confidence(parsed, is_valid, cfg)
    parsed["confidence"] = confidence
    parsed["_model_used"] = model
    return parsed, [parsed]


def _list_available_models(cfg: Config) -> List[str]:
    """Return the list of model IDs available from the YUNWU server."""
    url = cfg.api_base.rstrip("/") + "/models"
    headers = {"Content-Type": "application/json"}
    if cfg.api_key and cfg.api_key != "not-required":
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
    data = json.loads(body)
    return [item["id"] for item in data.get("data", []) if isinstance(item, dict) and "id" in item]


def _validate_models(cfg: Config) -> None:
    """Check that the configured models exist on the server."""
    available = _list_available_models(cfg)
    log.debug("Available models: %s", available)
    missing = [m for m in cfg.models if m not in available]
    if missing:
        filtered = [m for m in cfg.models if m in available]
        if filtered:
            cfg.models = tuple(filtered)
            log.warning(
                "Some configured models are not available: %s. Using available models: %s.",
                ", ".join(missing),
                ", ".join(filtered),
            )
        else:
            log.error(
                "None of the configured models are available: %s.",
                ", ".join(cfg.models),
            )
            sys.exit(1)


# ===========================================================================
# SECTION D — OUTPUT PARSING & CONFIDENCE SCORING
# ===========================================================================

def _strip_fences(text: str) -> str:
    """Remove markdown code fences that some models add despite being told not to."""
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text.strip())
    return text.strip()


def parse_model_output(raw: str) -> Tuple[Dict[str, Any], bool]:
    """
    Parse model output into a dict.
    Returns (parsed_dict, is_valid).
    """
    cleaned = _strip_fences(raw)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj, True
    except json.JSONDecodeError:
        pass
    # Try to extract JSON object with a regex fallback
    m = re.search(r'\{[\s\S]+\}', cleaned)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj, True
        except json.JSONDecodeError:
            pass
    return {"raw_text": raw}, False


def assess_confidence(parsed: Dict[str, Any], is_valid: bool, cfg: Config) -> float:
    """
    Return a confidence score in [0, 1].
    Uses the model's own 'confidence' field if present, otherwise estimates it.
    """
    if not is_valid:
        return 0.0

    # Model self-reported confidence
    model_conf = parsed.get("confidence")
    if isinstance(model_conf, (int, float)) and 0 <= model_conf <= 1:
        base = float(model_conf)
    else:
        base = 0.5  # neutral when absent

    # Penalise missing required fields
    missing = [f for f in cfg.required_fields if not parsed.get(f)]
    penalty = len(missing) * 0.1

    # Penalise very short extractions
    extracted = str(parsed.get("extracted_text", ""))
    if len(extracted) < 10:
        penalty += 0.15

    return max(0.0, min(1.0, base - penalty))


# ===========================================================================
# SECTION E — PER-FIGURE ORCHESTRATION
# ===========================================================================

def process_figure(
    fig: FigureRef,
    md_text: str,
    cfg: Config,
    dry_run: bool = False,
) -> FigureExtraction:
    """
    Full lifecycle for a single figure:
      context harvest → primary VLM call → confidence check → optional fallback.
    """
    ctx = harvest_context(md_text, fig, cfg)
    extraction = FigureExtraction(figure_ref=fig, context=ctx)

    # ---- Unresolved image -----------------------------------------------
    if fig.resolved_path is None:
        log.warning("Image not found for reference: %s", fig.rel_path)
        extraction.parsed = {"error": "image_not_found"}
        extraction.markdown_block = _build_warning_block(fig)
        return extraction

    if dry_run:
        # Dry run: skip model calls, insert a placeholder block
        extraction.parsed = {"dry_run": True, "confidence": 1.0}
        extraction.model_used = "dry-run"
        extraction.markdown_block = _build_markdown_block(fig, ctx, extraction.parsed)
        return extraction

    img_data, img_mime = _encode_image(fig.resolved_path)
    user_text = _build_user_message(ctx)

    # ---- Call the configured model -------------------------------------
    model_name = cfg.models[0] if cfg.models else "unknown"
    log.info("  [%s] calling model %s …", fig.figure_id, model_name)
    parsed, all_results = _infer_with_model(img_data, img_mime, user_text, cfg)

    extraction.raw_response = ""  # No single raw response anymore
    extraction.parsed = parsed
    extraction.model_used = parsed.get("_model_used", "unknown")
    extraction.markdown_block = _build_markdown_block(fig, ctx, extraction.parsed, all_results)
    return extraction


# ===========================================================================
# SECTION E (cont.) — MARKDOWN BLOCK BUILDER
# ===========================================================================

def _kw_list(kws: Any) -> str:
    if isinstance(kws, list):
        return ", ".join(str(k) for k in kws)
    return str(kws) if kws else ""


def _build_markdown_block(
    fig: FigureRef,
    ctx: FigureContext,
    p: Dict[str, Any],
    all_results: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Build the human+machine-readable block that replaces the image reference.

    The block uses a fence with a custom info string so downstream
    chunking tools can split on  <!--FIGURE_START-->  markers.
    
    If all_results is provided, includes a model comparison table.
    """
    fid        = p.get("figure_id") or fig.figure_id
    ftitle     = p.get("figure_title") or ""
    fname      = p.get("figure_name") or ""
    ftype      = p.get("figure_type") or "unknown"
    caption    = p.get("caption_summary") or ctx.caption or ""
    extracted  = p.get("extracted_text") or ""

    visual_el  = p.get("visual_elements") or []

    keywords   = _kw_list(p.get("keywords"))
    confidence = p.get("confidence", 0.0)
    notes      = p.get("notes") or ""
    src_page   = p.get("source_page_if_inferable") or fig.page
    model_used = p.get("_model_used") or ""

    visual_el_str = (
        "\n".join(f"  - {v}" for v in visual_el) if visual_el else "  (none)"
    )

    block = f"""
<!--FIGURE_START id="{fid}"-->
- **Figure {fid}
- **Figure title**: {ftitle}
- **Figure name**: {fname}
- **Figure type**: {ftype}
- **Image file**: {fig.rel_path}
- **Section**: {ctx.section_heading or "(none)"}
- **Caption**: {caption}
- **Extracted text**: {extracted}
- **Visual elements**: {visual_el_str}
- **Keywords**: {keywords}
- **Notes**: {notes}
- **Selected model**: {model_used} (confidence: {confidence:.2f})"""

    # ---- Add model comparison table if available ----
    if all_results and len(all_results) > 1:
        block += "\n\n**Model Comparison Results**:\n| Model | Confidence |\n|---|---|\n"
        for result in sorted(all_results, key=lambda x: x.get("confidence", 0.0), reverse=True):
            result_model = result.get("_model_used", "unknown")
            result_conf = result.get("confidence", 0.0)
            selected = " ← SELECTED" if result_model == model_used else ""
            block += f"| {result_model} | {result_conf:.2f}{selected} |\n"

    block += f"\n<!--FIGURE_END id=\"{fid}\"-->\n"
    return block.strip()


def _build_warning_block(fig: FigureRef) -> str:
    return (
        f'\n<!--FIGURE_START id="{fig.figure_id}"-->\n'
        f'### Figure {fig.figure_id}\n'
        f'> ⚠️ **Image not found**: `{fig.rel_path}`  \n'
        f'> Original reference preserved for manual review.\n\n'
        f'{fig.original_ref}\n'
        f'<!--FIGURE_END id="{fig.figure_id}"-->\n'
    )


# ===========================================================================
# SECTION F — MARKDOWN REINSERTION
# ===========================================================================

def reinsert_extractions(
    md_text: str,
    extractions: List[FigureExtraction],
) -> str:
    """
    Replace every original image reference in `md_text` with its
    extracted figure block.  We iterate in reverse offset order so that
    earlier char offsets remain valid after each replacement.
    """
    sorted_extr = sorted(
        extractions, key=lambda e: e.figure_ref.char_offset, reverse=True
    )
    for extr in sorted_extr:
        ref   = extr.figure_ref
        start = ref.char_offset
        end   = start + len(ref.original_ref)
        md_text = md_text[:start] + extr.markdown_block + md_text[end:]
    return md_text


# ===========================================================================
# SECTION G — REGISTRY BUILDER
# ===========================================================================

def build_registry(extractions: List[FigureExtraction]) -> Dict[str, Any]:
    registry: Dict[str, Any] = {}
    for extr in extractions:
        ref  = extr.figure_ref
        ctx  = extr.context
        fid  = extr.parsed.get("figure_id") or ref.figure_id

        # De-duplicate: if the model returned the same ID for two different figures,
        # append the canonical figure_id from the filename so keys stay unique.
        if fid in registry:
            fid = f"{fid}__{ref.figure_id}"

        # Keep a compact registry without transient/model-specific fields
        registry[fid] = {
            "figure_id":        fid,
            "figure_title":     extr.parsed.get("figure_title"),     
            "figure_name":      extr.parsed.get("figure_name"),
            "figure_type":      extr.parsed.get("figure_type"),
            "rel_path":         ref.rel_path,
            "section_heading":  ctx.section_heading,
            "caption":          ctx.caption,
            "Extracted_text":   extr.parsed.get("extracted_text"),
            "visual_elements":  extr.parsed.get("visual_elements"),
            "keywords":         extr.parsed.get("keywords"),
            "notes":            extr.parsed.get("notes"),
            "original_reference": ref.original_ref,
            "figure_id_from_model": extr.parsed.get("figure_id"),
        }
    return registry


# ===========================================================================
# PER-BOOK ORCHESTRATION
# ===========================================================================

def process_book(
    book_dir: Path,
    cfg: Config,
) -> Dict[str, Any]:
    """
    Process all figures in a single book folder.
    Returns a summary dict for the final report.
    """
    log.info("Processing book: %s", book_dir.name)

    # ---- Find the Markdown file -----------------------------------------
    md_files = list(book_dir.glob("*.md"))
    if not md_files:
        log.warning("No Markdown file found in %s — skipping.", book_dir)
        return {"book": book_dir.name, "error": "no_markdown"}

    if len(md_files) > 1:
        log.warning("%s has multiple .md files — using the first: %s",
                    book_dir.name, md_files[0].name)
    md_path = md_files[0]
    md_text = md_path.read_text(encoding="utf-8", errors="replace")

    # ---- Detect figures -------------------------------------------------
    figures = detect_figures(md_text, book_dir, cfg)
    log.info("  Found %d figure reference(s) in %s", len(figures), md_path.name)
    missing = [f for f in figures if f.resolved_path is None]
    if missing:
        log.warning("  %d image(s) could not be resolved.", len(missing))

    # ---- Process figures in parallel ------------------------------------
    extractions: List[FigureExtraction] = []
    if figures:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=cfg.figure_workers
        ) as pool:
            futures = {
                pool.submit(process_figure, fig, md_text, cfg, cfg.dry_run): fig
                for fig in figures
            }
            iterator = concurrent.futures.as_completed(futures)
            if tqdm is not None:
                iterator = tqdm(
                    iterator,
                    total=len(futures),
                    desc=f"figures: {book_dir.name}",
                    unit="fig",
                    leave=False,
                )
            for fut in iterator:
                fig = futures[fut]
                try:
                    extractions.append(fut.result())
                except Exception as exc:
                    log.error("  Error processing %s: %s", fig.figure_id, exc)
                    # Insert an error placeholder so we don't silently lose refs
                    ctx = FigureContext()
                    err_extr = FigureExtraction(figure_ref=fig, context=ctx)
                    err_extr.parsed = {"error": str(exc), "confidence": 0.0}
                    err_extr.markdown_block = _build_warning_block(fig)
                    extractions.append(err_extr)

    # Restore original order (parallel execution scrambles it)
    extractions.sort(key=lambda e: e.figure_ref.char_offset)

    # ---- Reinsert into Markdown -----------------------------------------
    enriched_md = reinsert_extractions(md_text, extractions)

    # ---- Build registry -------------------------------------------------
    registry = build_registry(extractions)

    # ---- Write outputs --------------------------------------------------
    if not cfg.dry_run:
        # Write only a registry JSON named after the book folder into the
        # configured output directory (no Markdown outputs).
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        registry_path = cfg.output_dir / f"{book_dir.name}.json"
        registry_path.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("  Wrote figure registry   → %s", registry_path)
    else:
        log.info("  [DRY RUN] Would write registry JSON for %s", book_dir.name)

    # ---- Gather stats for summary ---------------------------------------
    return {
        "source":        book_dir.name,
        "figures_found": len(figures),
        "missing_images": len(missing),
        "fallbacks_used": 0,
        "error":         None,
    }


def _normalize_figure_id(name: str) -> str:
    stem_clean = re.sub(r'[^A-Za-z0-9]+', '_', name.strip('_'))
    stem_clean = re.sub(r'_+', '_', stem_clean).strip('_')
    return f"Fig_{stem_clean}" if stem_clean else "Fig_image"


def discover_image_files(root: Path, cfg: Config) -> List[Path]:
    """Return image files found under `root`."""
    images: List[Path] = []
    if root.is_file():
        if root.suffix.lower() in cfg.image_extensions:
            images = [root]
    elif root.is_dir():
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in cfg.image_extensions:
                images.append(path)
    if not images:
        log.warning("No image files found under %s", root)
    return images


def process_image_file(image_path: Path, cfg: Config) -> FigureExtraction:
    fig = FigureRef(
        original_ref=str(image_path.name),
        rel_path=str(image_path.name),
        resolved_path=image_path,
        char_offset=0,
        page=None,
        figure_id=_normalize_figure_id(image_path.stem),
    )
    ctx = FigureContext()
    extraction = FigureExtraction(figure_ref=fig, context=ctx)

    if cfg.dry_run:
        extraction.parsed = {"dry_run": True, "confidence": 1.0}
        extraction.model_used = "dry-run"
        extraction.markdown_block = _build_markdown_block(fig, ctx, extraction.parsed, [])
        return extraction

    img_data, img_mime = _encode_image(image_path)
    user_text = _build_user_message(ctx)
    parsed, all_results = _infer_with_model(img_data, img_mime, user_text, cfg)

    extraction.raw_response = ""  # No single raw response anymore
    extraction.parsed = parsed
    extraction.model_used = parsed.get("_model_used", "unknown")
    extraction.markdown_block = _build_markdown_block(fig, ctx, extraction.parsed, all_results)
    return extraction


def process_image_files(
    image_files: List[Path],
    cfg: Config,
) -> List[Dict[str, Any]]:
    log.info("Processing %d image file(s) in %s", len(image_files), cfg.root_dir)

    extractions: List[FigureExtraction] = []
    if image_files:
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            futures = {
                pool.submit(process_image_file, img, cfg): img
                for img in image_files
            }
            iterator = concurrent.futures.as_completed(futures)
            if tqdm is not None:
                iterator = tqdm(
                    iterator,
                    total=len(futures),
                    desc=f"images: {cfg.root_dir.name}",
                    unit="img",
                    leave=False,
                )
            for fut in iterator:
                img = futures[fut]
                try:
                    extractions.append(fut.result())
                except Exception as exc:
                    log.error("  Error processing %s: %s", img.name, exc)
                    ctx = FigureContext()
                    err_fig = FigureRef(
                        original_ref=str(img.name),
                        rel_path=str(img.name),
                        resolved_path=img,
                        char_offset=0,
                        page=None,
                        figure_id=_normalize_figure_id(img.stem),
                    )
                    err_extr = FigureExtraction(figure_ref=err_fig, context=ctx)
                    err_extr.parsed = {"error": str(exc), "confidence": 0.0}
                    err_extr.markdown_block = _build_warning_block(err_fig)
                    extractions.append(err_extr)

    model_name = cfg.models[0] if cfg.models else "model"
    if not cfg.dry_run:
        # Write only a single registry JSON named after the source folder
        # (cfg.root_dir.name) into the configured output directory.
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        registry = build_registry(extractions)
        registry_path = cfg.output_dir / f"{cfg.root_dir.name}.json"
        registry_path.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("  Wrote combined figure registry → %s", registry_path)
    else:
        log.info("  [DRY RUN] Would write combined registry for %d image(s) using model %s", len(extractions), model_name)

    summaries: List[Dict[str, Any]] = []
    for extr in extractions:
        summaries.append({
            "source": extr.figure_ref.resolved_path.name if extr.figure_ref.resolved_path else extr.figure_ref.figure_id,
            "figures_found": 1,
            "missing_images": 0,
            "fallbacks_used": 0,
            "error": None if "error" not in extr.parsed else extr.parsed.get("error"),
        })
    return summaries


# ===========================================================================
# ROOT ORCHESTRATION
# ===========================================================================

def discover_books(root: Path) -> List[Path]:
    """Return directories under `root` that contain at least one .md file.

    This searches recursively so book folders nested in subdirectories are
    discovered as well.
    """
    books = []
    seen: set[Path] = set()
    for md_path in sorted(root.rglob("*.md")):
        parent = md_path.parent
        if parent not in seen:
            seen.add(parent)
            books.append(parent)
    if not books:
        log.warning("No book subdirectories (with .md files) found under %s", root)
    return books


def _run_single_model(cfg: Config, model_name: str) -> None:
    books = discover_books(cfg.root_dir)
    summaries: List[Dict[str, Any]] = []

    if books:
        log.info("Found %d book(s) under %s", len(books), cfg.root_dir)
        # Prepare tasks but skip books that already have outputs (checkpoint)
        tasks: List[concurrent.futures.Future] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            for book_dir in books:
                try:
                    rel = book_dir.relative_to(cfg.root_dir)
                except Exception:
                    rel = Path(book_dir.name)
                target_output_dir = cfg.output_dir / rel
                registry_path = target_output_dir / f"{book_dir.name}.json"
                if registry_path.exists() and not cfg.force:
                    log.info("Skipping already-processed book (checkpoint): %s", book_dir)
                    continue
                model_cfg = dataclasses.replace(cfg, output_dir=target_output_dir)
                target_output_dir.mkdir(parents=True, exist_ok=True)
                tasks.append(pool.submit(process_book, book_dir, model_cfg))

            for fut in concurrent.futures.as_completed(tasks):
                try:
                    summaries.append(fut.result())
                except Exception as exc:
                    log.error("Unexpected error processing a book: %s", exc)
                    summaries.append({
                        "source": "(unknown)",
                        "error": str(exc),
                        "figures_found": 0,
                        "missing_images": 0,
                        "fallbacks_used": 0,
                    })
    else:
        # If the root contains subfolders with images, process each subfolder
        # independently so we produce one JSON per folder. Otherwise,
        # fall back to processing image files directly under `root`.
        image_subdirs: List[Path] = []
        for sub in sorted(cfg.root_dir.iterdir()):
            if sub.is_dir():
                imgs = discover_image_files(sub, cfg)
                if imgs:
                    image_subdirs.append(sub)

        summaries = []
        if image_subdirs:
            for sub in image_subdirs:
                try:
                    rel = sub.relative_to(cfg.root_dir)
                except Exception:
                    rel = Path(sub.name)
                target_output_dir = cfg.output_dir / rel
                registry_path = target_output_dir / f"{sub.name}.json"
                if registry_path.exists() and not cfg.force:
                    log.info("Skipping already-processed folder (checkpoint): %s", sub)
                    continue
                imgs = discover_image_files(sub, cfg)
                sub_cfg = dataclasses.replace(cfg, root_dir=sub, output_dir=target_output_dir)
                target_output_dir.mkdir(parents=True, exist_ok=True)
                summaries.extend(process_image_files(imgs, sub_cfg))
        else:
            image_files = discover_image_files(cfg.root_dir, cfg)
            if not image_files:
                log.error("No books or image files to process — exiting.")
                sys.exit(0)
            # For files directly under root, mirror them into a single output dir
            target_output_dir = cfg.output_dir
            target_output_dir.mkdir(parents=True, exist_ok=True)
            cfg = dataclasses.replace(cfg, output_dir=target_output_dir)
            summaries = process_image_files(image_files, cfg)

    # ---- Summary report -------------------------------------------------
    _print_summary(summaries, cfg)

    # ---- Write summary JSON ---------------------------------------------
    if not cfg.dry_run:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = cfg.output_dir / f"pipeline_summary_{model_name}.json"
        summary_path.write_text(
            json.dumps({"config": dataclasses.asdict(cfg), "results": summaries},
                       indent=2, default=str),
            encoding="utf-8",
        )
        log.info("Summary report written to %s", summary_path)


def run(cfg: Config) -> None:
    """Main entry point — discover books, set up the model client, process all books."""
    logging.getLogger().setLevel(cfg.log_level.upper())

    # ---- Configure logging ------------------------------------------------
    log_level = cfg.log_level.upper()
    logging.getLogger().setLevel(log_level)
    for handler in logging.root.handlers:
        handler.setLevel(log_level)

    global API_REQUEST_SEMAPHORE
    API_REQUEST_SEMAPHORE = threading.Semaphore(cfg.max_concurrent_requests)

    # ---- Validate configured models -------------------------------------
    # Skip validation when running in dry-run mode or with --skip-validation.
    if not cfg.dry_run and not cfg.skip_validation:
        try:
            _validate_models(cfg)
        except Exception as exc:
            log.error("Model validation failed: %s", exc)
            sys.exit(1)
    elif cfg.skip_validation:
        log.info("Skipping model validation: using configured API directly.")
    else:
        log.info("Dry-run mode: skipping model validation.")

    books = discover_books(cfg.root_dir)

    if len(cfg.models) > 1:
        for model in cfg.models:
            model_output_dir = cfg.output_dir / model if books else cfg.output_dir
            model_cfg = dataclasses.replace(cfg, models=(model,), output_dir=model_output_dir)
            log.info("Processing model sequentially: %s", model)
            _run_single_model(model_cfg, model)
        return

    _run_single_model(cfg, cfg.models[0])


def _print_summary(summaries: List[Dict[str, Any]], cfg: Config) -> None:
    """Print a human-readable summary table to stdout."""
    total_items   = len(summaries)
    total_figs    = sum(s.get("figures_found", 0)  for s in summaries)
    total_missing = sum(s.get("missing_images", 0) for s in summaries)
    total_fb      = sum(s.get("fallbacks_used", 0) for s in summaries)
    errors        = [s for s in summaries if s.get("error")]

    banner = "=" * 62
    print(f"\n{banner}")
    print("  FIGURE PIPELINE SUMMARY")
    print(banner)
    print(f"  Mode            : {'DRY RUN' if cfg.dry_run else 'LIVE'}")
    print(f"  Items processed : {total_items}")
    print(f"  Figures found   : {total_figs}")
    print(f"  Missing images  : {total_missing}")
    print(f"  Fallback uses   : {total_fb}")
    print(f"  Errors          : {len(errors)}")
    print(banner)
    for s in sorted(summaries, key=lambda x: x.get("source", "")):
        tag = " ⚠ " if s.get("error") else "   "
        source = s.get("source", "(unknown)")
        print(
            f"{tag}{source:<30}  "
            f"figs={s.get('figures_found', 0):>3}  "
            f"missing={s.get('missing_images', 0):>2}  "
            f"fb={s.get('fallbacks_used', 0):>2}"
        )
    print(banner + "\n")


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="figure_pipeline",
        description="Enrich book Markdown files with VLM-extracted figure information.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    defaults = Config()

    p.add_argument("--root",      default=str(defaults.root_dir),
                   help="Input directory containing either book sub-folders or image files.")
    p.add_argument("--output",    default=str(defaults.output_dir),
                   help="Output root directory for enriched markdown and registries.")
    p.add_argument("--api-base",  default=defaults.api_base,
                   help="API base URL. Defaults to YUNWU_API_BASE_URL or the built-in default.")
    p.add_argument("--api-key",   default=defaults.api_key,
                   help="API key. If omitted, uses YUNWU_API_KEY from the environment.")
    p.add_argument("--max-tokens", type=int, default=defaults.max_tokens,
                   help="Max tokens for model responses.")
    p.add_argument("--temperature", type=float, default=defaults.temperature)
    p.add_argument("--workers",  type=int, default=defaults.workers,
                   help="Number of parallel book workers.")
    p.add_argument("--figure-workers", type=int, default=defaults.figure_workers,
                   help="Number of parallel figure workers per book.")
    p.add_argument("--max-concurrent-requests", type=int,
                   default=defaults.max_concurrent_requests,
                   help="Maximum number of concurrent API requests to avoid rate limiting.")
    p.add_argument("--request-delay", type=float,
                   default=defaults.request_delay,
                   help="Seconds to wait before each API request.")
    p.add_argument("--context-before", type=int,
                   default=defaults.context_chars_before,
                   help="Characters of Markdown to include before each figure.")
    p.add_argument("--context-after", type=int,
                   default=defaults.context_chars_after,
                   help="Characters of Markdown to include after each figure.")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse and detect figures without calling any model.")
    p.add_argument("--force", action="store_true",
                   help="Reprocess and overwrite existing outputs (ignore checkpoint).")
    p.add_argument("--skip-validation", action="store_true",
                   help="Skip model validation at startup (use configured API directly).")
    p.add_argument("--log-level", default=defaults.log_level,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args   = parser.parse_args(argv)

    cfg = Config(
        root_dir              = Path(args.root),
        output_dir            = Path(args.output),
        api_base              = args.api_base,
        api_key               = args.api_key,
        max_tokens            = args.max_tokens,
        temperature           = args.temperature,
        workers               = args.workers,
        figure_workers        = args.figure_workers,
        max_concurrent_requests = args.max_concurrent_requests,
        request_delay         = args.request_delay,
        context_chars_before  = args.context_before,
        context_chars_after   = args.context_after,
        dry_run               = args.dry_run,
        force                 = args.force,
        skip_validation       = args.skip_validation,
        log_level             = args.log_level,
    )
    run(cfg)


if __name__ == "__main__":
    main()