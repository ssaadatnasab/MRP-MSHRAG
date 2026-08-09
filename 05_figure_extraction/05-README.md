# Stage 5: Figure Information Extraction

Transforms the raw image references embedded in each Markdown file (for example, `![](_page_42_Figure_0.jpeg)`) into structured, searchable figure metadata. For each detected image reference, contextual information such as the section heading, caption, surrounding text, equations, and tables is harvested and submitted together with the image to a vision-language model.

## 1. Extract figure metadata

```bash
export YUNWU_API_KEY="your-api-key"
export YUNWU_API_BASE_URL="https://api.example.com/v1"
export YUNWU_MODEL="gpt-5.4-nano"

python "Testing_Code_VLM_Final_1 copy.py" \
  --root ./input_images \
  --output ./output \
  --workers 4
```

Reads `YUNWU_API_KEY`, `YUNWU_API_BASE_URL`, and `YUNWU_MODEL` from your environment by default. The pipeline detects figure references, resolves the underlying files, builds a context window around each figure, calls the model, and writes structured JSON metadata for downstream use.

Useful flags include `--dry-run` (preview without writing), `--force` (reprocess existing items), `--figure-workers` and `--max-concurrent-requests` (throughput tuning), `--context-before` and `--context-after` (control how much surrounding text is harvested per figure), and `--skip-validation` (skip the validation pass).

## 2. Insert extracted content back into Markdown

The companion script `extract_figure_metadata.py` can take the generated JSON results and insert the extracted content back into the Markdown at the matching image-reference locations. This makes the enriched observations visible directly inside the document.

```bash
python "extract_figure_metadata.py" \
  --json /path/to/figure_metadata.json \
  --md /path/to/input.md \
  --output /path/to/output.md
```

## Output

Per document: one JSON file containing structured metadata for every figure detected in that document. When used with the insertion utility, the enriched content can also be written back into the Markdown source itself.
