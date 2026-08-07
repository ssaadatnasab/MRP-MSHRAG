# Stage 6: Unified Representation, Chunking & Multimodal Corpora

## 1. Merge stage outputs

Merges the structured figure metadata from Stage 5 into the enriched
Markdown from Stage 4, using each image's original Markdown reference as
the matching key, producing one unified per-document Markdown file.

```bash
python merge_stage_outputs.py \
    --md-root /path/to/stage4_output \
    --json-root /path/to/stage5_output \
    --output-root /path/to/unified
```

If a Markdown file and its corresponding JSON output don't share an
identical filename, they're paired by fuzzy filename similarity
(`--threshold`, default `0.80`).

## 2. Chunk the unified corpus

No single chunking strategy is optimal for every content type or
downstream use case, so four independent strategies are provided, each
producing its own complete chunked index of the corpus. Run whichever
subset is relevant to your evaluation:

```bash
# (i) Markdown-aware Atomic Block Protection — masks equations/tables with
#     placeholders before splitting, guaranteeing no chunk boundary falls
#     inside a formula or table
python chunk_atomic_block_protection.py \
    --input /path/to/unified --output /path/to/chunks/atomic_block

# (ii) Small-to-Large (Parent–Child) — small child chunks for retrieval,
#      larger parent chunks returned to the LLM for context
python chunk_parent_child.py \
    --input /path/to/unified --output /path/to/chunks/parent_child

# (iii) Content-Type-Aware — routes text, equations, and tables through
#       independent pipelines; equations/tables become single, unsplit chunks
python chunk_content_type_aware.py \
    --input /path/to/unified --output /path/to/chunks/content_type_aware

# (iv) Hierarchical — splits along the document's own heading structure
#      (H1–H4) first, recursively sub-splitting oversized sections
python chunk_hierarchical.py \
    --input-dir /path/to/unified --output-dir /path/to/chunks/hierarchical \
    --max-tokens 2048 --overlap 400
```

All four accept a chunk-size/overlap pair (flag names vary slightly by
script — see `--help`); the parent–child script additionally uses a
smaller child chunk size than its parent chunk size, since the child tier
is meant to be the fine-grained retrieval unit rather than the
context-bearing one.

## 3. Build the multimodal chunk corpora (Image / Formula / Table)

Builds the three specialized retrieval corpora — image, formula, and
table — that the agentic RAG framework (Stage 09, "Stage 0.5 multimodal
triage") queries independently before main-corpus retrieval.

`merge_multimodal_json_to_jsonl.py` walks a directory containing three
subfolders (`Formula/`, `Image/`, `Table/`), each holding per-document
JSON chunk files produced upstream by steps 1–2 above (and by the
formula/table enrichment and figure-extraction stages further upstream),
and merges each subfolder's JSON files into a single `.jsonl` file:

```
Formula/Formula.jsonl
Image/Image.jsonl
Table/Table.jsonl
```

Each line in the output is one chunk record (a JSON object), ready to be
embedded and indexed.

**Usage:** edit `BASE_DIR` at the top of the script to point at your
local multimodal chunk directory (wherever your Formula/Image/Table JSON
files live), then run:

```bash
python merge_multimodal_json_to_jsonl.py
```

## Output

- A unified per-document Markdown file plus three modality-separated JSON
  files (formulas, tables, figures) — from step 1
- One chunked JSON/JSONL index per strategy run — from step 2
- `Formula.jsonl`, `Image.jsonl`, `Table.jsonl` — from step 3, which are
  what you point `--formula_corpus`, `--image_corpus`, and
  `--table_corpus` at in Stage 09 (`09_agentic_rag_starag/`)

All outputs are ready for embedding and indexing into a vector store.
