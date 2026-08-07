# Stage 3: Markdown Structural Normalization

PDF-to-Markdown conversion reliably preserves text content but frequently
introduces two kinds of structural noise: leftover page-marker artifacts and
flattened heading hierarchies (headings that were originally at different
logical levels all collapse to the same Markdown level, e.g. everything
becomes `#` or `##`). This stage repairs both, in three steps.

## 1. Clean page-marker artifacts
Strips page-marker span tags (e.g. `<span id="page-31-0">Page 31</span>`)
and unwraps page-anchor links (e.g. `[Foreword](#page-6-0)` → `Foreword`).
```bash
python clean_page_artifacts.py \
    --input-dir /path/to/markdown \
    --output-dir /path/to/cleaned
```

## 2. Detect heading format
Classifies each document as following a numbered heading convention
(`1`, `2`, `2.1`, `2.1.1`, ...) or a non-numbered one, using deterministic
rules with an LLM fallback for ambiguous cases. Prints one label
(`numbered` / `non-numbered`) per file — capture this to decide routing
in the next step.
```bash
python detect_heading_format.py \
    --input-dir /path/to/cleaned \
    --model llama3.1:70b          # served locally via Ollama by default
```
Requires an [Ollama](https://ollama.com) server running locally
(`OLLAMA_BASE_URL`, default `http://localhost:11434`), or point `--ollama-url`
at a different endpoint.

## 3. Normalize heading hierarchy
Route each document to the normalizer matching its detected format:
```bash
# non-numbered documents
python normalize_headings_mdast.py \
    --input-dir /path/to/non_numbered \
    --output-root /path/to/normalized

# numbered documents
python normalize_headings_reheader.py \
    --input-dir /path/to/numbered \
    --output-root /path/to/normalized
```
`normalize_headings_mdast.py` enforces a consistent heading hierarchy for
non-numbered documents; `normalize_headings_reheader.py` reconstructs
heading levels for numbered documents directly from their numeric prefixes.
Both fall back to a local rule-based normalizer if their respective external
tool (Node.js `mdast-normalize-headings`, or a `reheader`/`rehead` CLI) isn't
available on `PATH`.

## Output
Structurally consistent Markdown with restored heading hierarchy, ready for
Stage 4.

