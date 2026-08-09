# Stage 3: Markdown Structural Normalization
[](https://github.com/ssaadatnasab/MRP-Multimodal-RAG-Preprocessing/blob/main/03_structural_normalization/03-README.md#stage-3-markdown-structural-normalization)

PDF-to-Markdown conversion reliably preserves text content but frequently introduces two kinds of structural noise: leftover page-marker artifacts and flattened heading hierarchies (headings that were originally at different logical levels all collapse to the same Markdown level, e.g. everything becomes `#` or `##`). This stage repairs both, in three steps.

This folder supports Stage 3 of the preprocessing pipeline using current scripts:

- `LLM_detector.py` — classifies each Markdown document as `numbered` or `non-numbered`.
- `normalize_heading_pipeline.py` — normalizes heading hierarchy and routes documents automatically.
- `normalize_headings_mdast.py` — direct mdast-based heading normalizer.
- `normalize_headings_reheader.py` — direct numbered-heading reheader normalizer.

## 1. Clean page-marker artifacts

Strips page-marker span tags (e.g. `<span id="page-31-0">Page 31</span>`) and unwraps page-anchor links (e.g. `[Foreword](#page-6-0)` → `Foreword`).

```bash
python clean_page_artifacts.py \
    --input-dir /path/to/markdown \
    --output-dir /path/to/cleaned
```

## 2. Detect heading format

Use `LLM_detector.py` to classify cleaned Markdown documents. It prints one label per file:

```bash
python "$(pwd)/Markdown Refinement/LLM detector/LLM_detector.py" \
    --input-dir /path/to/cleaned \
    --ollama-url http://localhost:11434 \
    --model llama3.1:70b
```

If you want to force LLM-only behavior, add:

```bash
--llm-only
```

The script falls back to a deterministic heuristic when the LLM is unavailable or ambiguous.

## 3. Normalize heading hierarchy

The current codebase provides two direct normalization tools and one recommended orchestration script.

- `normalize_headings_mdast.py` is the `mdast`-based normalizer.
- `normalize_headings_reheader.py` is the numbered-heading reheader normalizer.
- `normalize_heading_pipeline.py` is the central Stage 3 orchestrator that detects document format and routes each file to the correct backend.

Recommended usage: run the unified refinement pipeline on the cleaned documents:

```bash
python "$(pwd)/Markdown Refinement/normalize_heading_pipeline.py" \
    --input-dir /path/to/cleaned \
    --output-dir /path/to/normalized
```

This orchestrator will:

- detect each document's heading format internally,
- route non-numbered documents through `mdast-normalize-headings` using `normalize_headings_mdast.py`,
- route numbered documents through `reheader`/`md-reheader` using `normalize_headings_reheader.py`,
- fall back to local rule-based normalization when external tools are unavailable.

If you prefer to run the direct normalizer scripts instead of the orchestrator:

```bash
python "$(pwd)/Markdown Refinement/Mdast-Util-Normalize-Headings/normalize_headings_mdast.py" \
    --input-dir /path/to/non_numbered \
    --output-root /path/to/normalized
```

```bash
python "$(pwd)/Markdown Refinement/MD-Reheader/normalize_headings_reheader.py" \
    --input-dir /path/to/numbered \
    --output-root /path/to/normalized
```

## Optional section testing

To inspect intermediate behavior for a single file:

```bash
python "$(pwd)/Markdown Refinement/normalize_heading_pipeline.py" \
    --section heuristic --file /path/to/file.md
```

```bash
python "$(pwd)/Markdown Refinement/normalize_heading_pipeline.py" \
    --section llm --file /path/to/file.md
```

```bash
python "$(pwd)/Markdown Refinement/normalize_heading_pipeline.py" \
    --section marktripy --file /path/to/file.md \
    --output-file /path/to/out.md
```

```bash
python "$(pwd)/Markdown Refinement/normalize_heading_pipeline.py" \
    --section mdast --file /path/to/file.md \
    --output-file /path/to/out.md
```

## Output

The output is structurally consistent Markdown with restored heading hierarchy, ready for later enrichment stages.

## Key change from previous docs

- `detect_heading_format.py` now maps to `Markdown Refinement/LLM detector/LLM_detector.py`.
- The direct Stage 3 normalizers are now:
  - `Markdown Refinement/Mdast-Util-Normalize-Headings/normalize_headings_mdast.py`
  - `Markdown Refinement/MD-Reheader/normalize_headings_reheader.py`
- `normalize_heading_pipeline.py` is the central normalization entrypoint for Stage 3.
