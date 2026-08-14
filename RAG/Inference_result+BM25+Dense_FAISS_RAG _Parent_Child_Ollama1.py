
import argparse
import importlib
import json
import time
import random
import pickle
import hashlib
import re
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from openai import OpenAI
import sys
import os
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

print("Program started successfully", flush=True)


def _normalize_col(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def resolve_input_columns(df: pd.DataFrame) -> tuple[str, str]:
    normalized_map = {_normalize_col(c): c for c in df.columns}

    question_col = next(
        (normalized_map[k] for k in ["question", "questions", "prompt", "query"] if k in normalized_map),
        None,
    )
    answer_col = next(
        (normalized_map[k] for k in ["answer", "answers", "goldanswer", "referenceanswer", "groundtruth"] if k in normalized_map),
        None,
    )

    if question_col is None:
        raise ValueError(f"Input file missing question column. Detected columns: {list(df.columns)}")
    if answer_col is None:
        raise ValueError(f"Input file missing answer column. Detected columns: {list(df.columns)}")

    return question_col, answer_col

OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
SYSTEM_PROMPT = (
    "You are an expert in building energy science.\n\n"
    "Answer the questions using the provided retrieved context.\n\n"
    "Guidelines:\n"
    "1. Prioritize the retrieved context as the primary source of information.\n"
    "2. If the context is sufficient, base your answer strictly on it.\n"
    "3. If the context is partially relevant or insufficient, you may use your general knowledge to supplement the answer, but clearly rely on context where possible."
)
HARDCODED_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("YUNWU_API_KEY") or ""
DEFAULT_INPUT = "examples/input.xlsx"
DEFAULT_OUTPUT = "output/output_parent_child_FAISS.xlsx"
DEFAULT_CORPUS = "corpus/small-to-large-parent-child"
PARENT_CHILDREN_FILE = "strategy3_children.jsonl"
PARENT_PARENTS_FILE = "strategy3_parents.jsonl"
DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_BM25_TOP_K = 100
DEFAULT_DENSE_TOP_K = 100
DEFAULT_FUSED_TOP_K = 50
DEFAULT_RERANK_TOP_K = 5
DEFAULT_MMR_TARGET_K = 20
DEFAULT_MAX_TOKENS = 8192

ILLEGAL_EXCEL_CHAR_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")


def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return [1.0 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0] if b.ndim > 1 else 0), dtype=float)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_safe = np.divide(a, np.where(a_norm == 0, 1.0, a_norm))
    b_safe = np.divide(b, np.where(b_norm == 0, 1.0, b_norm))
    return np.clip(a_safe @ b_safe.T, 0.0, 1.0)


def _tokenize_for_bm25(text: str) -> list[str]:
    return [token for token in str(text).lower().split() if token]


def _sanitize_excel_value(value):
    if isinstance(value, str):
        return ILLEGAL_EXCEL_CHAR_RE.sub("", value)
    return value


def _get_token_encoder():
    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _truncate_to_max_tokens(text: str, max_tokens: int, encoder):
    if max_tokens <= 0:
        return "", 0, bool(text)
    if not text:
        return "", 0, False

    if encoder is None:
        parts = text.split()
        if len(parts) <= max_tokens:
            return text, len(parts), False
        return " ".join(parts[:max_tokens]), max_tokens, True

    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text, len(tokens), False

    truncated = encoder.decode(tokens[:max_tokens]).strip()
    return truncated, max_tokens, True


def _extract_answer(resp):
    if not resp or not getattr(resp, "choices", None):
        return ""

    choice = resp.choices[0]
    message = None
    if hasattr(choice, "message"):
        message = choice.message
    elif isinstance(choice, dict):
        message = choice.get("message")

    if not message:
        return ""
    if hasattr(message, "content"):
        return (message.content or "").strip()
    if isinstance(message, dict):
        return (message.get("content") or "").strip()
    return ""


def _resolve_output_path(output_path: str) -> Path:
    path = Path(output_path)
    if path.suffix.lower() not in {".xlsx", ".xls", ".xlsm", ".xlsb"}:
        path = path.with_suffix(".xlsx")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_output(df: pd.DataFrame, question_col: str, answer_col: str, output_path: str):
    out_df = df[[question_col, answer_col, "RAG_Context", "Result"]].rename(
        columns={question_col: "question", answer_col: "Answer"}
    ).copy()
    for col in out_df.columns:
        out_df[col] = out_df[col].map(_sanitize_excel_value)
    output_file = _resolve_output_path(output_path)
    out_df.to_excel(output_file, index=False)


def _compute_corpus_signature(jsonl_files: list[Path]) -> str:
    """Build a stable signature from corpus file names, sizes, and mtimes."""
    hasher = hashlib.sha256()
    for file_path in sorted(jsonl_files, key=lambda p: p.name):
        stat = file_path.stat()
        hasher.update(f"{file_path.name}|{stat.st_size}|{int(stat.st_mtime)}".encode("utf-8"))
    return hasher.hexdigest()


def _is_parent_child_corpus(corpus_dir: Path) -> bool:
    return (
        (corpus_dir / PARENT_CHILDREN_FILE).exists()
        and (corpus_dir / PARENT_PARENTS_FILE).exists()
    )


def _load_parent_child_corpus(corpus_dir: Path, Document):
    """Load child chunks for retrieval and parent chunks for LLM context.

    JSONL field contract (from Small-to-large parent-child chunker):

    Parents (strategy3_parents.jsonl) — fields read:
      chunk_id, text, source_file, section_heading
    Parents — fields ignored:
      child_ids, has_variables, chunk_role (role is assigned here as "parent")

    Children (strategy3_children.jsonl) — fields read:
      chunk_id, text, parent_id, source_file, section_heading
    Children — fields ignored:
      child_ids, has_variables, chunk_role (role is assigned here as "child")

    Linking uses child.parent_id -> parent.chunk_id only; parent.child_ids is not used.
    source_file is stored in Document metadata as "source".
    """
    children_path = corpus_dir / PARENT_CHILDREN_FILE
    parents_path = corpus_dir / PARENT_PARENTS_FILE

    parent_lookup: dict[str, object] = {}
    with open(parents_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️  Skipping invalid parent JSON: {parents_path.name}:{line_num}", flush=True)
                continue

            chunk_id = obj.get("chunk_id")
            text = obj.get("text", "").strip()
            if not chunk_id or not text:
                continue

            parent_lookup[chunk_id] = Document(
                page_content=text,
                metadata={
                    "chunk_id": chunk_id,
                    "source": obj.get("source_file", parents_path.name),
                    "section_heading": obj.get("section_heading", ""),
                    "chunk_role": "parent",
                    "line": line_num,
                },
            )

    child_docs = []
    with open(children_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️  Skipping invalid child JSON: {children_path.name}:{line_num}", flush=True)
                continue

            text = obj.get("text", "").strip()
            parent_id = obj.get("parent_id")
            if not text or not parent_id:
                continue

            child_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "chunk_id": obj.get("chunk_id"),
                        "parent_id": parent_id,
                        "source": obj.get("source_file", children_path.name),
                        "section_heading": obj.get("section_heading", ""),
                        "chunk_role": "child",
                        "line": line_num,
                        "chunk_index": len(child_docs),
                    },
                )
            )

    if not child_docs:
        raise ValueError(f"No child chunks loaded from {children_path}")
    if not parent_lookup:
        raise ValueError(f"No parent chunks loaded from {parents_path}")

    missing_parents = {
        doc.metadata["parent_id"]
        for doc in child_docs
        if doc.metadata["parent_id"] not in parent_lookup
    }
    if missing_parents:
        print(
            f"⚠️  {len(missing_parents)} child chunks reference missing parents; "
            "those children will be skipped at retrieval time.",
            flush=True,
        )

    return child_docs, parent_lookup


def _mmr_select(
    docs: list,
    relevance_scores: list[float],
    doc_embeddings: np.ndarray,
    lambda_mult: float,
    target_k: int,
) -> list[int]:
    if not docs:
        return []

    target_k = max(1, min(target_k, len(docs)))
    lambda_mult = float(lambda_mult)
    selected: list[int] = []
    remaining = list(range(len(docs)))

    while remaining and len(selected) < target_k:
        if not selected:
            best_idx = max(remaining, key=lambda idx: relevance_scores[idx])
            selected.append(best_idx)
            remaining.remove(best_idx)
            continue

        best_idx = None
        best_value = -1e9
        for idx in remaining:
            relevance = relevance_scores[idx]
            selected_embeddings = doc_embeddings[selected]
            candidate_embedding = doc_embeddings[idx : idx + 1]
            max_similarity = float(_cosine_similarity_matrix(candidate_embedding, selected_embeddings).max()) if selected_embeddings.size else 0.0
            mmr_value = lambda_mult * relevance - (1.0 - lambda_mult) * max_similarity
            if mmr_value > best_value:
                best_value = mmr_value
                best_idx = idx

        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def build_hybrid_retriever_from_corpus(
    corpus_path: str,
    k: int,
    embedding_model: str,
    reranker_model: str,
    bm25_top_k: int,
    dense_top_k: int,
    fused_top_k: int,
    mmr_lambda: float,
    mmr_target_k: int,
    parent_child: bool | None = None,
):
    """Load JSONL corpus and prepare BM25, dense, MMR, and reranking components."""
    try:
        Document = importlib.import_module("langchain_core.documents").Document
        try:
            HuggingFaceEmbeddings = importlib.import_module("langchain_huggingface").HuggingFaceEmbeddings
        except Exception:
            HuggingFaceEmbeddings = importlib.import_module("langchain_community.embeddings").HuggingFaceEmbeddings
        CrossEncoder = importlib.import_module("sentence_transformers").CrossEncoder
    except Exception as import_err:
        raise ImportError(
            "Missing hybrid retrieval dependencies. Install with: "
            "pip install langchain langchain-community langchain-huggingface rank_bm25 sentence-transformers numpy"
        ) from import_err

    corpus_dir = Path(corpus_path)
    if not corpus_dir.exists():
        raise ValueError(f"Corpus folder not found: {corpus_path}")

    parent_lookup: dict[str, object] | None = None
    use_parent_child = _is_parent_child_corpus(corpus_dir) if parent_child is None else parent_child

    if use_parent_child:
        if not _is_parent_child_corpus(corpus_dir):
            raise ValueError(
                f"Parent-child corpus not found in {corpus_path}. "
                f"Expected {PARENT_CHILDREN_FILE} and {PARENT_PARENTS_FILE}."
            )
        docs, parent_lookup = _load_parent_child_corpus(corpus_dir, Document)
        jsonl_files = [corpus_dir / PARENT_CHILDREN_FILE]
        corpus_signature = _compute_corpus_signature(jsonl_files)
        print(
            f"📚 Loaded parent-child corpus: {len(docs)} child chunks for retrieval, "
            f"{len(parent_lookup)} parent chunks for LLM context.",
            flush=True,
        )
    else:
        docs = []
        jsonl_files = sorted(corpus_dir.glob("*.jsonl"))
        markdown_files = sorted(corpus_dir.rglob("*.md"))

        if jsonl_files:
            corpus_signature = _compute_corpus_signature(jsonl_files)
            print(f"📚 Loading corpus from {len(jsonl_files)} JSONL files...", flush=True)
            for jsonl_file in jsonl_files:
                try:
                    chunk_idx = 0
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, start=1):
                            if not line.strip():
                                continue
                            try:
                                obj = json.loads(line)
                                text = obj.get("text", "").strip()
                                if text:
                                    metadata = {
                                        "source": jsonl_file.name,
                                        "line": line_num,
                                        "chunk_index": chunk_idx,
                                    }
                                    docs.append(Document(page_content=text, metadata=metadata))
                                    chunk_idx += 1
                            except json.JSONDecodeError:
                                print(f"⚠️  Skipping invalid JSON: {jsonl_file.name}:{line_num}", flush=True)
                                continue
                except Exception as e:
                    print(f"⚠️  Failed to read file {jsonl_file.name}: {e}", flush=True)
                    continue
        elif markdown_files:
            corpus_signature = _compute_corpus_signature(markdown_files)
            print(f"📚 Loading corpus from {len(markdown_files)} markdown files...", flush=True)
            for md_index, markdown_file in enumerate(markdown_files):
                try:
                    text = markdown_file.read_text(encoding="utf-8").strip()
                    if text:
                        docs.append(
                            Document(
                                page_content=text,
                                metadata={"source": markdown_file.name, "chunk_index": md_index},
                            )
                        )
                except Exception as e:
                    print(f"⚠️  Failed to read markdown file {markdown_file.name}: {e}", flush=True)
        else:
            raise ValueError(f"No .jsonl or .md files found in {corpus_path}")

    if not docs:
        raise ValueError(f"Cannot load documents from Corpus. Check if {corpus_path} contains valid content.")

    print(
        f"✅ Loaded {len(docs)} documents into RAG knowledge base.",
        flush=True,
    )

    tokenized_corpus = [_tokenize_for_bm25(doc.page_content) for doc in docs]
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    cache_prefix = "children_" if use_parent_child else ""
    cache_path = Path(corpus_path) / f"{cache_prefix}embeddings_cache.pkl"
    cache_meta_path = Path(corpus_path) / f"{cache_prefix}embeddings_cache_meta.json"
    expected_meta = {
        "embedding_model": embedding_model,
        "doc_count": len(docs),
        "corpus_signature": corpus_signature,
    }

    cache_valid = False
    if cache_path.exists() and cache_meta_path.exists():
        try:
            with open(cache_meta_path, "r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            doc_embedding_matrix = np.asarray(cached, dtype=float)

            cache_valid = (
                cached_meta.get("embedding_model") == expected_meta["embedding_model"]
                and int(cached_meta.get("doc_count", -1)) == expected_meta["doc_count"]
                and cached_meta.get("corpus_signature") == expected_meta["corpus_signature"]
                and doc_embedding_matrix.ndim == 2
                and doc_embedding_matrix.shape[0] == len(docs)
            )
            if cache_valid:
                print("✅ Loading cached embeddings...", flush=True)
            else:
                print("⚠️ Embedding cache metadata mismatch. Recomputing embeddings...", flush=True)
        except Exception as cache_error:
            print(f"⚠️ Failed to load embedding cache ({cache_error}). Recomputing embeddings...", flush=True)
            cache_valid = False

    if not cache_valid:
        print("⚙️ Computing embeddings...", flush=True)
        doc_embedding_matrix = np.asarray(
            embeddings.embed_documents([doc.page_content for doc in docs]),
            dtype=float,
        )
        with open(cache_path, "wb") as f:
            pickle.dump(doc_embedding_matrix, f)
        with open(cache_meta_path, "w", encoding="utf-8") as f:
            json.dump(expected_meta, f, ensure_ascii=True, indent=2)

    # Build or load FAISS index for fast dense search (inner product on normalized vectors == cosine)
    use_faiss = False
    faiss_index = None
    try:
        import faiss

        dim = int(doc_embedding_matrix.shape[1])
        index_path = Path(corpus_path) / f"{cache_prefix}faiss_index.idx"
        # Try loading existing index file if present and compatible
        if index_path.exists():
            try:
                idx = faiss.read_index(str(index_path))
                # Validate index size
                if getattr(idx, "ntotal", -1) == doc_embedding_matrix.shape[0]:
                    faiss_index = idx
                    use_faiss = True
                    print(f"✅ Loaded FAISS index from {index_path}", flush=True)
                else:
                    print("⚠️ FAISS index present but size mismatch — rebuilding index.", flush=True)
            except Exception:
                print("⚠️ Failed to read FAISS index file; rebuilding.", flush=True)

        if not use_faiss:
            faiss_index = faiss.IndexFlatIP(dim)
            faiss_index.add(doc_embedding_matrix.astype('float32'))
            try:
                faiss.write_index(faiss_index, str(index_path))
                print(f"✅ Built and saved FAISS IndexFlatIP (dim={dim}) to {index_path}", flush=True)
            except Exception:
                print("⚠️ Built FAISS index but failed to save to disk.", flush=True)
            use_faiss = True
    except Exception:
        print("⚠️ FAISS not available or failed to build; falling back to matrix cosine search.", flush=True)

    query_embed = embeddings.embed_query
    reranker = CrossEncoder(reranker_model)

    bm25_model = importlib.import_module("rank_bm25").BM25Okapi(tokenized_corpus)

    def retrieve(query: str):
        query_tokens = _tokenize_for_bm25(query)
        bm25_scores = list(map(float, bm25_model.get_scores(query_tokens)))
        bm25_norm = _min_max_normalize(bm25_scores)

        query_vector = np.asarray(query_embed(query), dtype=float)
        # Use FAISS for dense search when available
        if use_faiss and faiss_index is not None:
            try:
                qvec = query_vector.reshape(1, -1).astype('float32')
                distances, topk = faiss_index.search(qvec, max(1, dense_top_k))
                dense_top_indices = topk.reshape(-1).tolist()
                # create dense_scores array and fill returned indices with distances (clip 0..1)
                dense_scores_arr = np.zeros(len(docs), dtype=float)
                for idx_pos, score in zip(dense_top_indices, distances.reshape(-1)):
                    dense_scores_arr[int(idx_pos)] = float(np.clip(score, 0.0, 1.0))
                dense_scores = dense_scores_arr.tolist()
            except Exception:
                dense_scores = _cosine_similarity_matrix(doc_embedding_matrix, query_vector.reshape(1, -1)).reshape(-1).tolist()
                dense_top_indices = np.argsort(dense_scores)[::-1][: max(1, dense_top_k)].tolist()
        else:
            dense_scores = _cosine_similarity_matrix(doc_embedding_matrix, query_vector.reshape(1, -1)).reshape(-1).tolist()
            dense_top_indices = np.argsort(dense_scores)[::-1][: max(1, dense_top_k)].tolist()

        dense_norm = _min_max_normalize(dense_scores)

        bm25_top_indices = np.argsort(bm25_scores)[::-1][: max(1, bm25_top_k)]
        candidate_indices = list(dict.fromkeys(list(bm25_top_indices) + list(dense_top_indices)))

        fused_candidates = []
        for idx in candidate_indices:
            fused_score = (0.4 * bm25_norm[idx]) + (0.6 * dense_norm[idx])
            fused_candidates.append((idx, float(np.clip(fused_score, 0.0, 1.0))))

        fused_candidates.sort(key=lambda item: item[1], reverse=True)
        fused_candidates = fused_candidates[: max(1, fused_top_k)]

        fused_doc_indices = [idx for idx, _ in fused_candidates]
        fused_relevance = [score for _, score in fused_candidates]
        fused_doc_embeddings = doc_embedding_matrix[fused_doc_indices]
        mmr_selected_local = _mmr_select(
            docs=[docs[idx] for idx in fused_doc_indices],
            relevance_scores=fused_relevance,
            doc_embeddings=fused_doc_embeddings,
            lambda_mult=mmr_lambda,
            target_k=max(1, mmr_target_k),
        )

        diversified_indices = [fused_doc_indices[idx] for idx in mmr_selected_local]
        diversified_docs = [docs[idx] for idx in diversified_indices]

        rerank_pairs = [(query, doc.page_content) for doc in diversified_docs]
        rerank_scores = reranker.predict(rerank_pairs).tolist() if rerank_pairs else []
        rerank_pool_size = max(k * 5, mmr_target_k) if use_parent_child else k
        reranked = sorted(
            list(zip(diversified_indices, diversified_docs, rerank_scores)),
            key=lambda item: item[2],
            reverse=True,
        )[: max(1, rerank_pool_size)]

        if use_parent_child and parent_lookup is not None:
            final_docs = []
            seen_parent_ids: set[str] = set()
            for doc_idx, child_doc, score in reranked:
                parent_id = child_doc.metadata.get("parent_id")
                if not parent_id or parent_id in seen_parent_ids:
                    continue
                parent_doc = parent_lookup.get(parent_id)
                if parent_doc is None:
                    continue

                seen_parent_ids.add(parent_id)
                resolved_doc = Document(
                    page_content=parent_doc.page_content,
                    metadata=dict(parent_doc.metadata),
                )
                resolved_doc.metadata["bm25_score"] = float(bm25_scores[doc_idx])
                resolved_doc.metadata["dense_score"] = float(dense_scores[doc_idx])
                resolved_doc.metadata["fused_score"] = float((0.4 * bm25_norm[doc_idx]) + (0.6 * dense_norm[doc_idx]))
                resolved_doc.metadata["rerank_score"] = float(score)
                resolved_doc.metadata["matched_child_id"] = child_doc.metadata.get("chunk_id")
                resolved_doc.metadata["matched_child_text"] = child_doc.page_content[:240]
                final_docs.append(resolved_doc)
                if len(final_docs) >= max(1, k):
                    break
            return final_docs

        final_docs = []
        for doc_idx, doc, score in reranked[: max(1, k)]:
            doc.metadata = dict(doc.metadata)
            doc.metadata["bm25_score"] = float(bm25_scores[doc_idx])
            doc.metadata["dense_score"] = float(dense_scores[doc_idx])
            doc.metadata["fused_score"] = float((0.4 * bm25_norm[doc_idx]) + (0.6 * dense_norm[doc_idx]))
            doc.metadata["rerank_score"] = float(score)
            final_docs.append(doc)
        return final_docs

    return retrieve


def build_rag_context(retriever, query: str, max_chars: int = 16000) -> str:
    """Retrieve relevant context from the corpus for a given query."""
    retrieved = retriever.invoke(query) if hasattr(retriever, "invoke") else retriever(query)
    segments = []
    total = 0

    for i, doc in enumerate(retrieved, start=1):
        source = doc.metadata.get("source", "?")
        rerank_score = doc.metadata.get("rerank_score", None)
        score_text = f" | rerank={rerank_score:.4f}" if isinstance(rerank_score, (int, float)) else ""
        section = doc.metadata.get("section_heading")
        section_text = f" | section={section}" if section else ""
        matched_child_id = doc.metadata.get("matched_child_id")
        child_text = (
            f" | matched_child={matched_child_id}"
            if matched_child_id
            else ""
        )
        segment = f"[{i}] From {source}{section_text}{score_text}{child_text}\n{doc.page_content}\n"
        remaining = max_chars - total
        if len(segment) > remaining:
            if not segments and remaining > 0:
                segments.append(segment[:remaining])
            print(f"⚠️ Context truncated at chunk {i}/{len(retrieved)}", flush=True)
            break

        segments.append(segment)
        total += len(segment)

    return "\n".join(segments).strip()


def main():
    parser = argparse.ArgumentParser(description="Read Question column, call YUNWU API with retries, and save Result to new xlsx.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input xlsx file path (must contain Question column)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output xlsx file path (will contain Result column)")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, help="RAG knowledge base folder (parent-child: strategy3_children.jsonl + strategy3_parents.jsonl)")
    parser.add_argument("--parent_child", action=argparse.BooleanOptionalAction, default=True, help="Use small-to-large parent-child retrieval (search children, return parents)")
    parser.add_argument("--key", type=str, default=HARDCODED_API_KEY)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--rag_k", type=int, default=5, help="RAG retrieval top-k")
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_EMBEDDING_MODEL, help="Dense embedding model")
    parser.add_argument("--reranker_model", type=str, default=DEFAULT_RERANKER_MODEL, help="Cross-encoder reranker model")
    parser.add_argument("--bm25_top_k", type=int, default=DEFAULT_BM25_TOP_K, help="BM25 candidate fetch size")
    parser.add_argument("--dense_top_k", type=int, default=DEFAULT_DENSE_TOP_K, help="Dense candidate fetch size")
    parser.add_argument("--fused_top_k", type=int, default=DEFAULT_FUSED_TOP_K, help="Fused and MMR candidate pool size")
    parser.add_argument("--mmr_lambda", type=float, default=0.7, help="MMR lambda: relevance vs diversity tradeoff")
    parser.add_argument("--mmr_target_k", type=int, default=DEFAULT_MMR_TARGET_K, help="MMR target_k after fusion (pruning before reranking)")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum completion tokens for LLM response")
    parser.add_argument("--max_context_chars", type=int, default=16000, help="Maximum context characters to inject into prompt")
    parser.add_argument("--start_row", type=int, default=1, help="Start row for reading (1-based, e.g. 101 starts from row 101)")
    args = parser.parse_args()

    df = pd.read_excel(args.input)

    question_col, answer_col = resolve_input_columns(df)

    if "Result" not in df.columns:
        df["Result"] = ""
    if "RAG_Context" not in df.columns:
        df["RAG_Context"] = ""

    start_pos = max(1, args.start_row) - 1
    if start_pos >= len(df):
        print(f"Start row {args.start_row} exceeds total rows {len(df)}. No processing needed.", flush=True)
        _save_output(df, question_col, answer_col, args.output)
        print(f"Processing complete. Saved to {args.output}", flush=True)
        return

    client = OpenAI(api_key=args.key, base_url=OLLAMA_BASE_URL)
    encoder = _get_token_encoder()

    retriever = build_hybrid_retriever_from_corpus(
        corpus_path=args.corpus,
        k=args.rag_k,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        fused_top_k=args.fused_top_k,
        mmr_lambda=args.mmr_lambda,
        mmr_target_k=args.mmr_target_k,
        parent_child=args.parent_child,
    )
    if args.parent_child:
        print(
            "✅ Parent-child RAG enabled: retrieve on child chunks -> map to parent chunks -> "
            "BM25 top-100 + Dense cosine top-100 -> weighted fusion (0.4/0.6) -> MMR -> Cross-Encoder rerank",
            flush=True,
        )
    else:
        print(
            "✅ LangChain RAG enabled: BM25 top-100 + Dense cosine top-100 -> "
            "weighted fusion (0.4/0.6, min-max normalized) -> MMR -> Cross-Encoder rerank",
            flush=True,
        )

    processed_count = 0

    for pos, q in df.loc[start_pos:, question_col].items():
        existing = df.at[pos, "Result"]
        if isinstance(existing, str) and existing.strip():
            print(f"⏭️ Skipping row {pos+1} (already answered)", flush=True)
            continue

        q_text = "" if pd.isna(q) else str(q)
        human_row = pos + 1
        rag_context = build_rag_context(
            retriever=retriever,
            query=q_text,
            max_chars=max(320, args.max_context_chars),
        )

        user_prompt = (
            "Question:\n"
            f"{q_text}\n\n"
            "Retrieved Context (parent-child RAG from Corpus):\n"
            f"{rag_context if rag_context else 'No relevant context found.'}\n\n"
            "Instruction:\n"
            "Answer the question using the retrieved context when relevant."
        )

        answer = ""
        provider_completion_tokens = 0
        was_truncated = False
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=args.max_tokens,
                    max_tokens=args.max_tokens,
                    stream=False,
                    extra_body={
                        "options": {
                            "num_predict": args.max_tokens,
                        }
                    },
                )
                answer = _extract_answer(resp)
                provider_completion_tokens = int(resp.usage.completion_tokens) if getattr(resp, "usage", None) else 0

                if not answer and provider_completion_tokens > 0:
                    raw_fallback = None
                    try:
                        choice = resp.choices[0]
                        raw_fallback = getattr(choice, "text", None)
                        if not raw_fallback and isinstance(choice, dict):
                            raw_fallback = choice.get("text") or choice.get("message")
                    except Exception:
                        raw_fallback = None

                    if not raw_fallback:
                        try:
                            raw_fallback = str(resp)
                        except Exception:
                            raw_fallback = ""

                    answer, _, was_truncated = _truncate_to_max_tokens(raw_fallback, args.max_tokens, encoder)
                    print(
                        f"⚠️ Provider returned no message.content. Saved fallback output, tokens_reported={provider_completion_tokens}",
                        flush=True,
                    )
                else:
                    answer, _, was_truncated = _truncate_to_max_tokens(answer, args.max_tokens, encoder)

                print(f"\n🔄 Running row {human_row} ...", flush=True)
                print(f"😊 Question: {q_text[:80]}{'...' if len(q_text)>80 else ''}", flush=True)
                print(f"📚 RAG Context: {rag_context[:120]}{'...' if len(rag_context)>120 else ''}", flush=True)
                print(f"✅ DS Answer: {answer[:120]}{'...' if len(answer)>120 else ''}", flush=True)
                if was_truncated:
                    print(f"⚠️ Saved answer was locally truncated to {args.max_tokens} tokens.", flush=True)
                break
            except Exception as e:
                wait_time = min(60, (2 ** attempt) + random.random())
                print(f"[{pos+1}] Call failed: {e}, retrying in {wait_time:.1f} seconds ({attempt+1}/5)...", flush=True)
                time.sleep(wait_time)

        df.at[pos, "Result"] = answer
        df.at[pos, "RAG_Context"] = rag_context
        processed_count += 1

        if processed_count % 1 == 0:
            _save_output(df, question_col, answer_col, args.output)
            print(f"✅ Checkpoint saved at row {pos+1}.", flush=True)

    _save_output(df, question_col, answer_col, args.output)
    print(f"Processing complete. Saved to {args.output}", flush=True)

if __name__ == "__main__":
    main()