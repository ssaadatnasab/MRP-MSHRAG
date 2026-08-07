# Stage 4: Formula & Table Enrichment

Formulas and tables are frequently corrupted during PDF-to-Markdown
conversion — broken LaTeX (missing braces, malformed subscripts) and
misaligned or duplicated table headers. This stage repairs and enriches
both, strictly additively: original content is never removed or rewritten,
only supplemented with generated descriptions and metadata. Each formula
and table is detected and submitted to the LLM individually with a
targeted prompt, keeping the model's context narrow and the process fast.

## 1. Enrich formulas and tables
```bash
python enrich_formulas_and_tables.py \
    --markdown-root /path/to/normalized \
    --out-root /path/to/enriched \
    --model llama3.1:70b
```
Reads `LLM_API_KEY` / `LLM_API_BASE_URL` from your environment (see the
repo-root `.env.example`). For formulas, this corrects LaTeX syntax and
generates a natural-language description plus structured metadata (name,
aliases, use case, domain keywords). For tables, it repairs header/column
misalignment and generates analogous metadata (subject, property type,
column/unit definitions, keywords).

Progress is checkpointed per element, so an interrupted run can be safely
re-launched with the same command and will resume rather than reprocess
completed work. Useful flags: `--workers N` (concurrent files),
`--max-files N` (process a subset first), `--dry-run` (preview without
writing), `--overwrite` (reprocess files that already have output).

## 2. Clean up enrichment artifacts
The LLM occasionally echoes its own reasoning or restates output labels
inline (e.g. a stray `### STEP 1 — Fix the table formatting` line). This
strips those residual markers without touching the enriched content itself.
```bash
python clean_enrichment_artifacts.py \
    --input-path /path/to/enriched \
    --output-path /path/to/enriched_cleaned
```

## 3. Normalize table widths
Re-inserted tables can end up with visually inconsistent column widths.
This re-renders every table with column widths matched to its widest cell
and a correctly formatted separator row, leaving all non-table text
untouched.
```bash
python adjust_table_widths.py /path/to/enriched_cleaned /path/to/final
```

## Output
Per document: one enriched Markdown file plus two JSON files (structured
formula metadata, structured table metadata).

