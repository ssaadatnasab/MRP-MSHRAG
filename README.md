# MRP: Multimodal-RAG-Preprocessing
Six-stage pipeline that converts a corpus of engineering textbooks (PDF) into a chunked, retrieval-ready knowledge base, using domain-adapted LLM enrichment for formulas, tables, and figures, and four distinct chunking strategies for downstream RAG evaluation.
> Companion code for: *[Paper title], [Authors], [Venue, Year]* — [link/DOI once available]

## Pipeline overview

<img width="4638" height="1714" alt="Picture 1 (1)" src="https://github.com/user-attachments/assets/18c0680e-5e8c-434c-b027-32525da24f0c" />

## Domain-adapted processing architecture

<img width="3327" height="1760" alt="image" src="https://github.com/user-attachments/assets/5dc1b0bb-372d-4857-8462-2aa1acf8135a" />

Formulas, tables, texts, and images are first converted to a unified Markdown
representation, then repaired (page-artifact removal, heading-hierarchy
restoration) before formula/table content is enriched by an LLM and figure
content by a vision-language model. Both paths are guided by domain-informed,
tailored prompts and constrained to Pydantic schemas, which validate the
model's output format before it is converted into natural-language
descriptions and structured metadata. The two paths converge into one
per-book representation, which is then split using four chunking strategies,
embedded, and indexed.

The dashed red path shows the generic baseline this is evaluated against: a
naive PDF-utility text extraction with no structure repair, no schema
validation, and no content-type-specific handling — text, tables, and
figures are all flattened the same way before chunking.

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
├── 01_pdf_collection/
│   └── 01-README.md                    # corpus manifest / book list (no code)
├── 02_markdown_extraction/
│   ├── 02-README.md
│   └── convert_pdfs_to_markdown.py
├── 03_structural_normalization/
│   ├── 03-README.md
│   ├── clean_page_artifacts.py
│   ├── LLM_detector.py
│   ├── normalize_heading_pipeline.py
│   ├── normalize_headings_mdast.py
│   └── normalize_headings_reheader.py
├── 04_formula_table_enrichment/
│   ├── 04-README.md
│   ├── enrich_formulas_and_tables.py
│   ├── clean_enrichment_artifacts.py
│   └── adjust_table_widths.py
├── 05_figure_extraction/
│   ├── 05-README.md
│   ├── extract_figure_metadata.py
│   └── json_to_markdown_injector.py
├── 06_unified_chunking/
│   ├── 06-README.md
│   ├── merge_multimodal_json_to_jsonl.py
│   ├── merge_stage_outputs.py
│   ├── chunk_atomic_block_protection.py
│   ├── chunk_parent_child.py
│   ├── chunk_content_type_aware.py
│   └── chunk_hierarchical.py
└── README.md

```

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) running locally (or an OpenAI-compatible API endpoint) for the LLM-based classification/enrichment/extraction steps

### Install
```bash
git clone https://github.com/[your-username]/[repo-name].git
cd [repo-name]

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Python dependencies
pip install -r requirements.txt

# PDF → Markdown converter (Stage 2)
pip install marker-pdf

# Optional: only needed if you use the mdast-based heading normalizer (Stage 3)
npm install remark remark-parse remark-stringify mdast-normalize-headings

# API / model configuration
cp .env.example .env   # then fill in your LLM_API_KEY and LLM_API_BASE_URL
```

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
