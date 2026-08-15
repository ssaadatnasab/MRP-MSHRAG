# MRP & MSHRAG: Multimodal-RAG-Preprocessing & Multi-Stage-Hybrid-RAG
Two-part pipeline. First, a six-stage preprocessing pipeline converts a
corpus of engineering textbooks (PDF) into a chunked, retrieval-ready
knowledge base, using domain-adapted LLM enrichment for formulas, tables,
and figures, and four distinct chunking strategies. Second, a RAG &
Judging toolkit runs hybrid retrieval inference against that knowledge
base and automatically scores the generated answers.

> Companion code for: *[Paper title], [Authors], [Venue, Year]* — [link/DOI once available]

## Preprocessing Pipeline overview

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

## Preprocessing Pipeline stages

| # | Stage | What it does | Code |
|---|---|---|---|
| 1 | [PDF Collection](https://github.com/ssaadatnasab/MRP-MSHRAG/tree/main/01_pdf_collection) | Textbooks | *(corpus manifest, not code * |
| 2 | [Markdown Extraction](https://github.com/ssaadatnasab/MRP-MSHRAG/tree/main/02_markdown_extraction) | PDF → Markdown via [Marker](https://github.com/VikParuchuri/marker); structure and images preserved | `02_markdown_extraction/` |
| 3 | [Structural Normalization](https://github.com/ssaadatnasab/MRP-MSHRAG/tree/main/03_structural_normalization) | Page-artifact cleanup, heading-format detection, heading-hierarchy normalization | `03_structural_normalization/` |
| 4 | [Formula & Table Enrichment](https://github.com/ssaadatnasab/MRP-MSHRAG/tree/main/04_formula_table_enrichment) | LLM-generated natural-language descriptions and metadata attached to every formula and table | `04_formula_table_enrichment/` |
| 5 | [Figure Info Extraction](https://github.com/ssaadatnasab/MRP-MSHRAG/tree/main/05_figure_extraction) | Vision-language model produces semantic descriptions and structured metadata for every figure | `05_figure_extraction/` |
| 6 | [Unified Representation & Chunking](https://github.com/ssaadatnasab/MRP-MSHRAG/tree/main/06_unified_chunking) | Merges all stage outputs into one enriched Markdown corpus, then applies four chunking strategies | `06_unified_chunking/` |

Each stage folder has its own short README with the exact command to run it.

## RAG & Judging

A lightweight toolkit that runs hybrid retrieval-augmented generation
against the Stage 6 chunk corpora and automatically judges the generated
answers:

- Hybrid retrieval: BM25 + dense embeddings + FAISS + MMR diversification + cross-encoder reranking
- Flat-corpus and parent/child (small-to-large) corpus support
- An OpenAI-compatible API client and a local Ollama client variant for on-prem models
- An LLM-based judger that scores generated answers against reference answers

| Script | What it does |
|---|---|
| `multi-stage_hybrid_RAG_API` | Hybrid RAG over a flat corpus of JSONL/Markdown chunks |
| `multi-stage_hybrid_RAG_parent_child_API` | Hybrid RAG over a parent/child (small-to-large) corpus |
| `multi-stage_hybrid_RAG_Ollama.py` | Hybrid RAG over a flat corpus of JSONL/Markdown chunks targeting a local Ollama / Ollama-compatible endpoint |
| `multi-stage_hybrid_RAG_parent_child_Ollama.py` | Hybrid RAG over a parent/child (small-to-large) corpus targeting a local Ollama / Ollama-compatible endpoint |
| `LLM_base_inference.py` | Ollama-compatible inference variant |
| `LLM_judger` | Scores `Result` answers against reference `Answer`s, writes structured scores |

See `RAG/RAG-README.md` for full usage, CLI flags, and troubleshooting.

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
│   ├── chunk_atomic_block_protection.py
│   ├── chunk_parent_child.py
│   ├── chunk_content_type_aware.py
│   └── chunk_hierarchical.py
├── RAG/
│   ├── RAG-README.md
│   ├── LLM_base_inference.py
│   ├── multi-stage_hybrid_RAG_API.py
│   ├── multi-stage_hybrid_RAG_parent_child_API.py
│   ├── multi-stage_hybrid_RAG_Ollama.py
│   ├── multi-stage_hybrid_RAG_parent_child_Ollama.py
│   ├── LLM_judger.py
└── README.md

```

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) running locally (or an OpenAI-compatible API endpoint) for the LLM-based classification/enrichment/extraction steps

### Install
```bash
git clone https://github.com/ssaadatnasab/MRP&MSHRAG-Multimodal-RAG-Preprocessing & Multi-Stage-Hybrid-RAG.git
cd MRP&MSHRAG-Multimodal-RAG-Preprocessing & Multi-Stage-Hybrid-RAG

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
