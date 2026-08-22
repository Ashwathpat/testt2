"""
hybrid_retriever.py - Advanced Hybrid Search with Multi-Query RRF & Reranking

Architecture (from the $200K AI Engineer diagram):

User Query
    ├── Dense Embeddings → Qdrant Vector Search
    ├── Sparse Embeddings → BM25 Keyword Search
    └── Late Interaction  → Cross-Encoder Reranking

Multi-Query Flow:
    User Query → Generate Similar Queries (LLM)
    → [Vector Search Q1, Q2, Q3, Q4, Original Query]
    → Reciprocal Rank Fusion
    → Re-ranked Results
    → Generative Output
"""

import os
import time
import math
import threading
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Lazy imports to avoid startup crashes if optional deps are missing
# ---------------------------------------------------------------------------


def _get_bm25_class():
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Re-use the existing embedding model and Qdrant client from retrieve.py
# ---------------------------------------------------------------------------
from app.retrieve import (
    embed_texts,
    client as qdrant_client,
    _embed_query_cached,
    _SEMANTIC_CACHE,
    COLLECTION_NAME,
    DEFAULT_K,
)
from app.query_expander import expand_query

# ---------------------------------------------------------------------------
# BM25 Sparse Index (built once at startup from Qdrant payloads)
# ---------------------------------------------------------------------------

_BM25_INDEX = None
_BM25_DOCS = []  # list of {"text": ..., "source_id": ..., "score": ...}
_BM25_LOCK = threading.Lock()


def build_bm25_index(max_docs: int = 5000):
    """
    Build an in-memory BM25 index from the Qdrant collection payloads.
    Called once at server startup. BM25 search is < 1ms after this.
    """
    global _BM25_INDEX, _BM25_DOCS

    BM25Okapi = _get_bm25_class()
    if BM25Okapi is None:
        print("[HybridRetriever] rank_bm25 not installed, BM25 disabled")
        return

    try:
        from qdrant_client.http import models

        # Scroll through the collection to get all document texts
        all_docs = []
        offset = None
        batch_size = 100

        while len(all_docs) < max_docs:
            result = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=batch_size,
                offset=offset,
                with_payload=["window_text", "source_id"],
                with_vectors=False,
            )
            points, next_offset = result

            if not points:
                break

            for point in points:
                payload = point.payload or {}
                text = payload.get("window_text", "")
                source_id = payload.get("source_id", "")
                if text.strip():
                    all_docs.append({
                        "text": text,
                        "source_id": source_id,
                        "point_id": point.id,
                    })

            offset = next_offset
            if offset is None:
                break

        if not all_docs:
            print("[HybridRetriever] No documents found for BM25 index")
            return

        # Tokenize and build BM25 index
        tokenized = [doc["text"].lower().split() for doc in all_docs]

        with _BM25_LOCK:
            _BM25_INDEX = BM25Okapi(tokenized)
            _BM25_DOCS = all_docs

        print(f"[HybridRetriever] BM25 index built with {len(all_docs)} documents")

    except Exception as e:
        print(f"[HybridRetriever] BM25 index build failed: {e}")


def _bm25_search(query: str, k: int = 10) -> list[dict]:
    """
    Sparse keyword search using BM25.
    Returns list of {"text", "source_id", "score", "rank"}.
    """
    if _BM25_INDEX is None or not _BM25_DOCS:
        return []

    tokenized_query = query.lower().split()

    with _BM25_LOCK:
        scores = _BM25_INDEX.get_scores(tokenized_query)

    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for rank, idx in enumerate(top_indices):
        if scores[idx] <= 0:
            break
        doc = _BM25_DOCS[idx]
        results.append({
            "text": doc["text"],
            "source_id": doc["source_id"],
            "score": float(scores[idx]),
            "rank": rank + 1,
            "method": "bm25_sparse",
        })

    return results


# ---------------------------------------------------------------------------
# Dense Vector Search (wraps existing Qdrant search)
# ---------------------------------------------------------------------------


def _dense_search(query: str, k: int = 5) -> list[dict]:
    """
    Dense vector search via Qdrant (same as existing retrieve_context but
    returns rank metadata for RRF).
    """
    from qdrant_client.http import models

    q_clean = query.strip().lower()
    q_vec_tuple = _embed_query_cached(q_clean)

    try:
        response = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=list(q_vec_tuple),
            limit=k,
            search_params=models.SearchParams(hnsw_ef=32),
            with_vectors=False,
            with_payload=["window_text", "source_id"],
        )

        results = []
        for rank, hit in enumerate(response.points):
            payload = hit.payload or {}
            results.append({
                "text": payload.get("window_text", ""),
                "source_id": payload.get("source_id", ""),
                "score": float(hit.score),
                "rank": rank + 1,
                "method": "dense_vector",
            })

        return results

    except Exception as e:
        print(f"[HybridRetriever] Dense search error: {e}")
        return []


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

RRF_K = 60  # Standard RRF constant


def reciprocal_rank_fusion(
    result_lists: list[list[dict]], k: int = RRF_K
) -> list[dict]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.
    RRF score = sum(1 / (k + rank_i)) for each list where the doc appears.

    This is the standard approach from the diagram:
    [Vector Search Q1..Q4, Original] → RRF → Re-ranked Results
    """
    # Track RRF scores by document text (as unique key)
    doc_scores = defaultdict(float)
    doc_data = {}  # Store full doc data keyed by text

    for result_list in result_lists:
        for item in result_list:
            doc_key = item["text"][:200]  # Use first 200 chars as unique key
            doc_scores[doc_key] += 1.0 / (k + item["rank"])

            # Keep the version with the highest individual score
            if doc_key not in doc_data or item["score"] > doc_data[doc_key]["score"]:
                doc_data[doc_key] = item

    # Sort by RRF score descending
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    fused_results = []
    for doc_key, rrf_score in sorted_docs:
        doc = doc_data[doc_key].copy()
        doc["rrf_score"] = rrf_score
        doc["method"] = "hybrid_rrf"
        fused_results.append(doc)

    return fused_results


# ---------------------------------------------------------------------------
# Late Interaction Reranking (Lightweight Cross-Encoder via Semantic Sim)
# ---------------------------------------------------------------------------


def rerank_results(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates using late interaction / cross-encoder style scoring.
    Uses the existing FastEmbed model for query-document semantic similarity
    as a lightweight alternative to a full cross-encoder.

    This implements the "Rerank" box from the architecture diagram.
    """
    if not candidates:
        return []

    try:
        # Embed query and all candidate texts together
        texts_to_embed = [f"query: {query}"] + [
            f"passage: {c['text'][:512]}" for c in candidates[:20]
        ]

        embeddings = embed_texts(texts_to_embed)
        query_emb = np.array(embeddings[0], dtype=np.float32)

        reranked = []
        for i, candidate in enumerate(candidates[:20]):
            doc_emb = np.array(embeddings[i + 1], dtype=np.float32)
            # Cosine similarity as reranking score
            rerank_score = float(np.dot(query_emb, doc_emb))

            reranked_doc = candidate.copy()
            reranked_doc["rerank_score"] = rerank_score
            # Combine RRF score (if present) with rerank score
            rrf = candidate.get("rrf_score", 0.0)
            reranked_doc["final_score"] = 0.4 * rrf * 100 + 0.6 * rerank_score
            reranked.append(reranked_doc)

        # Sort by final combined score
        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked[:top_k]

    except Exception as e:
        print(f"[HybridRetriever] Reranking failed: {e}")
        # Fall back to original order
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# Main Hybrid Retrieval Pipeline
# ---------------------------------------------------------------------------


def hybrid_retrieve_context(
    query: str,
    k: int = DEFAULT_K,
    enable_multi_query: bool = True,
    enable_bm25: bool = True,
    enable_reranking: bool = True,
) -> tuple[list[dict], dict]:
    """
    Full hybrid retrieval pipeline:

    1. Check semantic cache (instant return if hit)
    2. Generate query variations via LLM (multi-query expansion)
    3. Run dense vector search for each query variation (parallel)
    4. Run BM25 sparse search for original query
    5. Reciprocal Rank Fusion across all result sets
    6. Rerank fused results via late interaction scoring
    7. Return top-k results + retrieval metadata

    Returns:
        (results, metadata) where metadata contains timing and method info
    """
    t_start = time.perf_counter()
    metadata = {
        "retrieval_method": "hybrid",
        "query_variations": 0,
        "bm25_results": 0,
        "dense_results": 0,
        "rrf_candidates": 0,
        "cache_hit": False,
        "timings": {},
    }

    q_clean = query.strip().lower()

    # ──────────────────────────────────────────────────────
    # Step 0: Check semantic cache first (< 0.5ms)
    # ──────────────────────────────────────────────────────
    q_vec_tuple = _embed_query_cached(q_clean)
    q_vec = np.array(q_vec_tuple, dtype=np.float32)

    best_sim = 0.0
    best_hits = None
    for cached_vec, cached_hits in _SEMANTIC_CACHE:
        sim = float(np.dot(q_vec, cached_vec))
        if sim > best_sim:
            best_sim = sim
            best_hits = cached_hits

    if best_sim >= 0.95 and best_hits:
        metadata["cache_hit"] = True
        metadata["retrieval_method"] = "semantic_cache"
        metadata["timings"]["total_ms"] = (
            time.perf_counter() - t_start
        ) * 1000
        return best_hits[:k], metadata

    # ──────────────────────────────────────────────────────
    # Step 1: Multi-Query Expansion
    # ──────────────────────────────────────────────────────
    t_expand = time.perf_counter()
    if enable_multi_query:
        query_variations = expand_query(query, n_variations=4)
    else:
        query_variations = [query]

    metadata["query_variations"] = len(query_variations)
    metadata["timings"]["expansion_ms"] = (
        time.perf_counter() - t_expand
    ) * 1000

    # ──────────────────────────────────────────────────────
    # Step 2: Parallel Dense Search for all query variations
    # ──────────────────────────────────────────────────────
    t_search = time.perf_counter()
    all_dense_results = []

    with ThreadPoolExecutor(max_workers=min(len(query_variations), 5)) as executor:
        future_to_query = {
            executor.submit(_dense_search, q, k=k + 3): q
            for q in query_variations
        }
        for future in as_completed(future_to_query):
            try:
                results = future.result(timeout=3)
                all_dense_results.append(results)
            except Exception as e:
                print(f"[HybridRetriever] Dense search thread error: {e}")

    dense_count = sum(len(r) for r in all_dense_results)
    metadata["dense_results"] = dense_count
    metadata["timings"]["dense_search_ms"] = (
        time.perf_counter() - t_search
    ) * 1000

    # ──────────────────────────────────────────────────────
    # Step 3: BM25 Sparse Search
    # ──────────────────────────────────────────────────────
    t_bm25 = time.perf_counter()
    bm25_results = []
    if enable_bm25 and _BM25_INDEX is not None:
        bm25_results = _bm25_search(query, k=k + 3)
        metadata["bm25_results"] = len(bm25_results)
    metadata["timings"]["bm25_ms"] = (time.perf_counter() - t_bm25) * 1000

    # ──────────────────────────────────────────────────────
    # Step 4: Reciprocal Rank Fusion
    # ──────────────────────────────────────────────────────
    t_rrf = time.perf_counter()
    all_result_lists = all_dense_results
    if bm25_results:
        all_result_lists.append(bm25_results)

    if len(all_result_lists) > 1:
        fused = reciprocal_rank_fusion(all_result_lists)
    elif all_result_lists:
        fused = all_result_lists[0]
    else:
        fused = []

    metadata["rrf_candidates"] = len(fused)
    metadata["timings"]["rrf_ms"] = (time.perf_counter() - t_rrf) * 1000

    # ──────────────────────────────────────────────────────
    # Step 5: Reranking via Late Interaction Scoring
    # ──────────────────────────────────────────────────────
    t_rerank = time.perf_counter()
    if enable_reranking and len(fused) > k:
        final_results = rerank_results(query, fused, top_k=k)
    else:
        final_results = fused[:k]

    metadata["timings"]["rerank_ms"] = (time.perf_counter() - t_rerank) * 1000

    # ──────────────────────────────────────────────────────
    # Step 6: Format output & update semantic cache
    # ──────────────────────────────────────────────────────
    formatted = []
    for item in final_results:
        formatted.append({
            "text": item["text"],
            "source_id": item.get("source_id", ""),
            "score": item.get("rerank_score", item.get("score", 0.0)),
        })

    # Cache for future semantic similarity hits
    if formatted:
        if len(_SEMANTIC_CACHE) > 1000:
            _SEMANTIC_CACHE.pop(0)
        _SEMANTIC_CACHE.append((q_vec, formatted))

    metadata["timings"]["total_ms"] = (time.perf_counter() - t_start) * 1000

    print(
        f"[HybridRetriever] {metadata['retrieval_method']} | "
        f"queries={metadata['query_variations']} | "
        f"dense={metadata['dense_results']} | "
        f"bm25={metadata['bm25_results']} | "
        f"rrf={metadata['rrf_candidates']} | "
        f"total={metadata['timings']['total_ms']:.0f}ms"
    )

    return formatted, metadata
