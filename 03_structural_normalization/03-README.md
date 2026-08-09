# Stage 3: Markdown Structural Normalization

This folder supports Stage 3 of the preprocessing pipeline using two current scripts:

- `LLM_detector.py` — classifies each Markdown document as `numbered` or `non-numbered`.
- `MD_Refinement.py` — normalizes heading hierarchy and routes documents automatically.

## Adjusted Stage 3 Workflow

### 1. Clean page-marker artifacts

This step remains the same as before:

```bash
python clean_page_artifacts.py \
    --input-dir /path/to/markdown \
    --output-dir /path/to/cleaned
```

### 2. Detect heading format

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

### 3. Normalize heading hierarchy

In the current codebase, `MD_Refinement.py` replaces the previous separate `normalize_headings_mdast.py` and `normalize_headings_reheader.py` steps.

Run the unified refinement pipeline on the cleaned documents:

```bash
python "$(pwd)/Markdown Refinement/MD_Refinement.py" \
    --input-dir /path/to/cleaned \
    --output-dir /path/to/normalized
```

This script will:

- detect each document's heading format internally,
- route non-numbered documents through `mdast-normalize-headings`,
- route numbered documents through `marktripy`,
- fall back to local rule-based normalization when external tools are unavailable.

### Optional section testing

To inspect intermediate behavior for a single file:

```bash
python "$(pwd)/Markdown Refinement/MD_Refinement.py" \
    --section heuristic --file /path/to/file.md
```

```bash
python "$(pwd)/Markdown Refinement/MD_Refinement.py" \
    --section llm --file /path/to/file.md
```

```bash
python "$(pwd)/Markdown Refinement/MD_Refinement.py" \
    --section marktripy --file /path/to/file.md \
    --output-file /path/to/out.md
```

```bash
python "$(pwd)/Markdown Refinement/MD_Refinement.py" \
    --section mdast --file /path/to/file.md \
    --output-file /path/to/out.md
```

## Output

The output is structurally consistent Markdown with restored heading hierarchy, ready for later enrichment stages.

## Key change from previous docs

- `detect_heading_format.py` now maps to `Markdown Refinement/LLM detector/LLM_detector.py`.
- `normalize_headings_mdast.py` and `normalize_headings_reheader.py` are not used directly in the current code.
- `MD_Refinement.py` is the central normalization entrypoint for Stage 3.
