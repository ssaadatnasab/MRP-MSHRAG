# Stage 5: Figure Information Extraction

Transforms the raw image references embedded in each Markdown file (e.g.
`![](_page_42_Figure_0.jpeg)`) into structured, searchable figure metadata.
For each detected image reference, contextual information — section
heading, caption, and surrounding text, equations, and tables — is
harvested and submitted together with the image to a vision-language model.

## Run
```bash
python extract_figure_metadata.py \
    --root /path/to/enriched \
    --output /path/to/figures \
    --workers 4
```
Reads `LLM_API_KEY` / `LLM_API_BASE_URL` from your environment by default
(override with `--api-key` / `--api-base` if needed). Produces title, name,
visual-element descriptions, extracted text, domain keywords, and
explanatory notes for every figure.

Processing is checkpointed at the book level, so an interrupted run resumes
without reprocessing already-completed documents; use `--force` to
reprocess anyway. Useful flags: `--figure-workers` / `--max-concurrent-requests`
(throughput tuning), `--context-before` / `--context-after` (characters of
surrounding text harvested per figure), `--dry-run` (preview without
writing), `--skip-validation` (skip the confidence/field validation pass).

## Output
One JSON file per document, containing structured metadata for every
figure detected in that document.

