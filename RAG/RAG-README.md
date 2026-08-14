# RAG & Judger Tools

Lightweight toolkit for retrieval-augmented generation (RAG) workflows and automated answer judging. The repository contains scripts to build a hybrid retriever (BM25 + dense FAISS + MMR + reranking), run LLM inference (including an Ollama-compatible client), and evaluate/generated answers with a simple judger.

**Key features:**
- Hybrid retrieval pipeline: BM25 + dense embeddings + FAISS index + MMR + Cross-Encoder reranker.
- Parent/child corpus support for small-to-large retrieval.
- Local Ollama HTTP client variant for on-prem models.
- Judger script to evaluate LLM outputs against reference answers.

## Quick setup

- Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Provide API credentials via environment variables. The scripts check `OPENAI_API_KEY` and `YUNWU_API_KEY` (preferred). Do not hardcode secrets in files.

```bash
export OPENAI_API_KEY="sk-..."
# or
export YUNWU_API_KEY="sk-..."
```

Tip: you can create a `.env` file and use `python-dotenv` in local runs; do not commit `.env`.

## Requirements

The Python dependencies are listed in `requirements.txt`. Key packages used by the scripts include:

- pandas
- numpy
- sentence-transformers
- faiss-cpu
- rank_bm25
- langchain
- langchain-huggingface
- langchain-community
- python-dotenv

# Core data and ML stack
pip install pandas numpy

# Embeddings, FAISS and reranking
pip install sentence-transformers faiss-cpu rank_bm25

# LangChain and related helpers
pip install langchain langchain-huggingface langchain-community

# API client, environment helpers, and utilities
pip install openai python-dotenv requests

## Layout & important files

- `Inference_result+BM25+Dense_FAISS_RAG.py` — RAG pipeline for a flat corpus of JSONL/Markdown chunks.
- `Inference_result+BM25+Dense_FAISS_RAG _Parent_Child.py` — Parent/child corpus loader and retriever (children for search, parents for LLM context).
- `Inference_result+BM25+Dense_FAISS_RAG_Ollama.py` and `Inference_result_Ollama.py` — Variants that target local Ollama HTTP server or Ollama-compatible model endpoints.
- `Qwen_3_max_judger.py` — Judger that evaluates generated answers against references and writes structured scores.

## Usage examples

Run the flat-corpus RAG pipeline (example):

```bash
python "Inference_result+BM25+Dense_FAISS_RAG.py" \
	--input examples/input.xlsx \
	--output output/result.xlsx \
	--corpus corpus/
```

Run the parent-child pipeline (example):

```bash
python "Inference_result+BM25+Dense_FAISS_RAG _Parent_Child.py" \
	--input examples/input.xlsx \
	--output output/result_parent_child.xlsx \
	--corpus corpus/small-to-large-parent-child \
	--parent_child
```

Run with a local Ollama server (example):

```bash
python Inference_result_Ollama.py \
	--input examples/1487_questions.xlsx \
	--output output/output_qwen3.5_1487.xlsx \
	--base_url http://127.0.0.1:11434/v1
```

Run the judger:

```bash
python Qwen_3_max_judger.py \
	--input_file examples/output_semantic_headers_FAISS.xlsx \
	--output_file output/judged.xlsx
```


### Stage: Retrieval + Answering

The RAG script runs a fused retrieval: BM25 candidates + dense cosine similarity via FAISS, then prunes via MMR and reranks with a cross-encoder. Behavior is controlled via CLI flags, for example `--bm25_top_k`, `--dense_top_k`, `--fused_top_k`, `--mmr_lambda`, and `--mmr_target_k`.

Example CLI (practical flags):

```bash
python "Inference_result+BM25+Dense_FAISS_RAG.py" \
	--corpus corpus/ \
	--rag_k 5 \
	--bm25_top_k 100 \
	--dense_top_k 100 \
	--fused_top_k 50 \
	--mmr_lambda 0.7 \
	--mmr_target_k 20
```

### Stage: Judging (automated evaluation)

The judger (`Qwen_3_max_judger.py`) expects an Excel with `question`, `Answer` (reference), and `Result` (model output) columns. It calls a model (Responses API or chat fallback) to evaluate and emits a structured Excel with score columns.

```bash
python Qwen_3_max_judger.py --input_file examples/output.xlsx --output_file output/judged.xlsx --model qwen3-max
```

## Outputs

- Each script writes an Excel file under `output/` by default. Files contain the original question, retrieved `RAG_Context`, the LLM `Result`, and any evaluation columns produced by the judger.

## Environment & reproducibility

- The repo looks for API keys in the environment (`OPENAI_API_KEY` or `YUNWU_API_KEY`).
- Use `--start_row` flags to resume long runs or chunk processing.
- Where applicable, embeddings and FAISS indexes are cached to `corpus/` to speed repeated runs.

## Troubleshooting

- Missing dependencies: run `pip install -r requirements.txt` and ensure `faiss-cpu` and `sentence-transformers` installed for embeddings.
- Out-of-memory / GPU issues: these scripts default to CPU-friendly behavior; switch embedding device or install GPU builds of dependencies if available.


