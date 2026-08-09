# Stage 5: Figure Information Extraction

Transforms the raw image references embedded in each Markdown file (for example, `![](_page_42_Figure_0.jpeg)`) into structured, searchable figure metadata. For each detected image reference, contextual information such as the section heading, caption, surrounding text, equations, and tables is harvested and submitted together with the image to a vision-language model.

The main pipeline script is `Testing_Code_VLM_Final_1 copy.py`. It detects figure references, resolves the underlying files, builds a context window around each figure, calls the model, and writes structured JSON metadata for downstream use.

## Run

```bash
export YUNWU_API_KEY="your-api-key"
export YUNWU_API_BASE_URL="https://api.example.com/v1"
export YUNWU_MODEL="gpt-5.4-nano"

python "Testing_Code_VLM_Final_1 copy.py" \
  --root ./input_images \
  --output ./output \
  --workers 4
```

The script reads `YUNWU_API_KEY`, `YUNWU_API_BASE_URL`, and `YUNWU_MODEL` from the environment by default. Useful flags include:

- `--dry-run` to preview the workflow without writing output
- `--force` to reprocess existing items
- `--figure-workers` and `--max-concurrent-requests` for throughput tuning
- `--context-before` and `--context-after` to control the amount of surrounding text harvested per figure
- `--skip-validation` to skip the validation pass

## Output

One JSON file per processed document, containing structured metadata for every figure detected in that document. The output is suitable for indexing, retrieval, and downstream RAG workflows.

## Companion utility: JSON to Markdown insertion

The companion script `JSON_To_Markdown_Adding copy 3.py` can take the generated JSON results and insert the extracted content back into the Markdown at the matching image-reference locations. This is useful when you want the enriched observations to be visible directly inside the document.

## Notes

- The scripts default to repo-local folders: `./input_images` and `./output`.
- The API key is read from the environment and is not stored in the script.
- The generated output is JSON-based and suitable for downstream indexing or RAG workflows.
