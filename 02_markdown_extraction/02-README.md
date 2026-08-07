# Stage 2: Markdown Extraction (PDF → Markdown)

Converts each source PDF into structured Markdown using
[Marker](https://github.com/VikParuchuri/marker), preserving document
hierarchy, tables, and embedded images. Runs PDFs in parallel across worker
processes and (optionally) multiple GPUs.

Before running at scale, it's worth comparing Marker against an alternative
converter (e.g., [Docling](https://github.com/DS4SD/docling)) on a handful
of your own PDFs — conversion quality varies by document layout, and which
tool wins on heading-hierarchy preservation vs. table/formula extraction can
differ across corpora.

## Requirements
```bash
pip install marker-pdf
```

## Run
```bash
python convert_pdfs_to_markdown.py \
    --pdf-root /path/to/pdfs \
    --out-root /path/to/output \
    --workers 4 \
    --gpus "0,1,2,3"          # omit to auto-select by free GPU memory
```

Useful flags: `--max-pdfs N` (process a subset first), `--force-ocr`
(force OCR on all pages), `--dry-run` (preview commands without running),
`--timeout-sec` (per-PDF timeout, default 10800s). Already-converted PDFs
(folders that already contain `.md` output) are skipped automatically, so
the command is safe to re-run on a partially completed corpus.

## Output
One subfolder per PDF under `--out-root`, each containing the converted
Markdown file and its extracted images.

