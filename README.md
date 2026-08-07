# MRP: Multimodal-RAG-Preprocessing
Six-stage pipeline that converts a corpus of engineering textbooks (PDF) into a chunked, retrieval-ready knowledge base, using domain-adapted LLM enrichment for formulas, tables, and figures, and four distinct chunking strategies for downstream RAG evaluation.
> Companion code for: *[Paper title], [Authors], [Venue, Year]* — [link/DOI once available]

## Pipeline overview

<img width="800" height="250" alt="pipeline_diagram" src="https://github.com/user-attachments/assets/785b483c-9626-4cf0-8e84-15034f7b4156" />


## Pipeline stages

| # | Stage | What it does | Code |
|---|---|---|---|
| 1 | PDF Collection | Textbooks | *(corpus manifest, not code * |
| 2 | Markdown Extraction | PDF → Markdown via [Marker](https://github.com/VikParuchuri/marker); structure and images preserved | `02_markdown_extraction/` |
| 3 | Structural Normalization | Page-artifact cleanup, heading-format detection, heading-hierarchy normalization | `03_structural_normalization/` |
| 4 | Formula & Table Enrichment | LLM-generated natural-language descriptions and metadata attached to every formula and table | `04_formula_table_enrichment/` |
| 5 | Figure Info Extraction | Vision-language model produces semantic descriptions and structured metadata for every figure | `05_figure_extraction/` |
| 6 | Unified Representation & Chunking | Merges all stage outputs into one enriched Markdown corpus, then applies four chunking strategies | `06_unified_chunking/` |

Each stage folder has its own short README with the exact command to run it.

## Repository structure

```
.
├── 01_pdf_collection/          # corpus manifest / book list (no code)
├── 02_markdown_extraction/
│   └── convert_pdfs_to_markdown.py
├── 03_structural_normalization/
│   ├── clean_page_artifacts.py
│   ├── detect_heading_format.py
│   ├── normalize_headings_mdast.py
│   └── normalize_headings_reheader.py
├── 04_formula_table_enrichment/
│   ├── enrich_formulas_and_tables.py
│   ├── clean_enrichment_artifacts.py
│   └── adjust_table_widths.py
├── 05_figure_extraction/
│   └── extract_figure_metadata.py
├── 06_unified_chunking/
│   ├── merge_stage_outputs.py
│   ├── chunk_atomic_block_protection.py
│   ├── chunk_parent_child.py
│   ├── chunk_content_type_aware.py
│   └── chunk_hierarchical.py
├── docs/
│   └── pipeline_diagram.png
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### Prerequisites
- Python 3.10+
- [Marker](https://github.com/VikParuchuri/marker) (`pip install marker-pdf`) for Stage 2
- [Ollama](https://ollama.com) running locally (or an OpenAI-compatible API endpoint) for the LLM-based classification/enrichment/extraction steps
- Node.js 18+ with `npm install remark remark-parse remark-stringify mdast-normalize-headings` if using the mdast-based heading normalizer in Stage 3

### Install
```bash
git clone https://github.com/[your-username]/[repo-name].git
cd [repo-name]
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your API key / endpoint
```

### Run a stage
```bash
python 02_markdown_extraction/convert_pdfs_to_markdown.py \
    --pdf-root /path/to/pdfs --out-root /path/to/output --workers 4
```
See each stage folder for its full argument list (`--help` on any script).

## Citation
If you use this pipeline, please cite:
```bibtex
@article{[citekey],
  title   = {[Paper title]},
  author  = {[Authors]},
  journal = {[Venue]},
  year    = {[Year]},
  doi     = {[DOI]}
}
```
