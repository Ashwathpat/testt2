"""
rag_evaluator.py - RAG Evaluation Metrics

Implements the RAG Evaluation Metrics wheel from the diagram:
  - Consider User Experience (latency, responsiveness)
  - Evaluate Retrieval Component (context relevance, precision)
  - Evaluate Retrieval Quality (chunk diversity, coverage)
  - Assess End-to-End Performance (faithfulness, answer relevance, completeness)

All metrics are computed per-query and returned alongside the answer.
"""

import time
import re
import numpy as np
from app.grounding import normalize_text
from app.retrieve import embed_texts


# ──────────────────────────────────────────────────────────
# 1. RETRIEVAL METRICS — "Evaluate Retrieval Component"
# ──────────────────────────────────────────────────────────


def context_relevance_score(
    query: str, chunks: list[dict], threshold: float = 0.5
) -> tuple[float, list[float]]:
    """
    Measures how relevant each retrieved chunk is to the query.
    Returns (avg_score, per_chunk_scores).

    Uses semantic similarity between query embedding and each chunk.
    """
    if not chunks:
        return 0.0, []

    texts = [f"query: {query}"] + [
        f"passage: {c.get('text', '')[:512]}" for c in chunks
    ]
    embeddings = embed_texts(texts)
    query_emb = np.array(embeddings[0], dtype=np.float32)

    per_chunk = []
    for i in range(1, len(embeddings)):
        chunk_emb = np.array(embeddings[i], dtype=np.float32)
        sim = float(np.dot(query_emb, chunk_emb))
        per_chunk.append(round(sim, 4))

    avg = sum(per_chunk) / len(per_chunk) if per_chunk else 0.0
    return round(avg, 4), per_chunk


def retrieval_precision(
    chunk_scores: list[float], threshold: float = 0.5
) -> float:
    """
    Fraction of retrieved chunks that are actually relevant (above threshold).
    Precision = relevant_chunks / total_chunks
    """
    if not chunk_scores:
        return 0.0

    relevant = sum(1 for s in chunk_scores if s >= threshold)
    return round(relevant / len(chunk_scores), 4)


def chunk_diversity(chunks: list[dict]) -> float:
    """
    Measures diversity among retrieved chunks.
    Low diversity = chunks are near-duplicates (bad).
    High diversity = chunks cover different aspects (good).

    Computed as 1 - avg(pairwise_similarity).
    """
    if len(chunks) < 2:
        return 1.0

    texts = [f"passage: {c.get('text', '')[:512]}" for c in chunks]
    embeddings = embed_texts(texts)
    emb_array = np.array(embeddings, dtype=np.float32)

    # Compute pairwise cosine similarities
    similarities = []
    for i in range(len(emb_array)):
        for j in range(i + 1, len(emb_array)):
            sim = float(np.dot(emb_array[i], emb_array[j]))
            similarities.append(sim)

    if not similarities:
        return 1.0

    avg_sim = sum(similarities) / len(similarities)
    diversity = 1.0 - avg_sim
    return round(max(0.0, diversity), 4)


# ──────────────────────────────────────────────────────────
# 2. GENERATION METRICS — "Assess End-to-End Performance"
# ──────────────────────────────────────────────────────────


def faithfulness_score(answer: str, chunks: list[dict]) -> float:
    """
    Measures what fraction of the answer's claims are supported
    by the retrieved context. Uses sentence-level grounding check.

    1.0 = fully faithful, 0.0 = entirely hallucinated.
    """
    if not answer.strip() or not chunks:
        return 0.0

    # Split answer into sentences/claims
    sentences = re.split(r"(?<=[.!?|।])\s+", answer.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return 1.0  # Very short answer, assume faithful

    context = " ".join(c.get("text", "") for c in chunks)
    context_words = normalize_text(context)

    # Pre-embed context and all sentences at once
    texts_to_embed = [f"passage: {context[:1000]}"] + [f"query: {s}" for s in sentences]
    all_embs = embed_texts(texts_to_embed)
    
    context_emb = np.array(all_embs[0])
    sentence_embs = all_embs[1:]

    # Check each claim against context using lexical + semantic
    supported = 0
    for i, sentence in enumerate(sentences):
        # Lexical check
        answer_words = normalize_text(sentence)
        if answer_words:
            overlap = len(answer_words & context_words) / len(answer_words)
            if overlap >= 0.3:
                supported += 1
                continue

        # Semantic check
        sim = float(np.array(sentence_embs[i]) @ context_emb)
        if sim >= 0.5:
            supported += 1

    return round(supported / len(sentences), 4) if sentences else 0.0


def answer_relevance_score(query: str, answer: str) -> float:
    """
    Measures how relevant the generated answer is to the original question.
    Uses semantic similarity between query and answer.
    """
    if not answer.strip():
        return 0.0

    embs = embed_texts([
        f"query: {query}",
        f"passage: {answer[:512]}",
    ])

    sim = float(np.array(embs[0]) @ np.array(embs[1]))
    return round(max(0.0, sim), 4)


# ──────────────────────────────────────────────────────────
# 3. USER EXPERIENCE METRICS — "Consider User Experience"
# ──────────────────────────────────────────────────────────


def latency_budget_check(
    total_ms: float, target_ms: float = 200.0
) -> dict:
    """
    Check whether the pipeline met the 200ms latency target.
    """
    met_target = total_ms <= target_ms
    overshoot_pct = max(0, (total_ms - target_ms) / target_ms * 100)

    return {
        "target_ms": target_ms,
        "actual_ms": round(total_ms, 2),
        "met_target": met_target,
        "overshoot_pct": round(overshoot_pct, 1),
    }


# ──────────────────────────────────────────────────────────
# 4. AGGREGATE EVALUATION — Full RAG Quality Report
# ──────────────────────────────────────────────────────────


def evaluate_rag(
    query: str,
    answer: str,
    chunks: list[dict],
    total_latency_ms: float = 0.0,
    retrieval_metadata: dict = None,
) -> dict:
    """
    Run all RAG evaluation metrics and return a comprehensive quality report.
    Optimized: uses a SINGLE batched embed_texts() call for all metrics.
    """
    t0 = time.perf_counter()

    chunk_texts = [c.get("text", "")[:512] for c in chunks if c.get("text", "").strip()]
    answer_sentences = re.split(r"(?<=[.!?|।])\s+", answer.strip())
    answer_sentences = [s.strip() for s in answer_sentences if len(s.strip()) > 10]

    # --- Batch ALL texts into ONE embed_texts call ---
    # Layout: [query, answer, chunk1, chunk2, ..., sent1, sent2, ...]
    texts_to_embed = []
    texts_to_embed.append(f"query: {query}")           # idx 0 = query
    texts_to_embed.append(f"passage: {answer[:512]}")   # idx 1 = answer
    chunk_start = 2
    for ct in chunk_texts:
        texts_to_embed.append(f"passage: {ct}")
    chunk_end = chunk_start + len(chunk_texts)
    sent_start = chunk_end
    for s in answer_sentences:
        texts_to_embed.append(f"query: {s}")
    # Total: 2 + len(chunk_texts) + len(answer_sentences)

    all_embs = embed_texts(texts_to_embed)

    # Fast-path: detect if embeddings are dummy zeros (API was down)
    if all(v == 0.0 for v in all_embs[0][:10]):
        eval_ms = (time.perf_counter() - t0) * 1000
        latency_check = latency_budget_check(total_latency_ms)
        return {
            "retrieval": {"context_relevance": 0.0, "per_chunk_scores": [], "precision": 0.0, "diversity": 1.0},
            "generation": {"faithfulness": 1.0, "answer_relevance": 0.0},
            "user_experience": latency_check,
            "overall_quality": 0.35,  # Faithfulness default
            "eval_latency_ms": round(eval_ms, 2),
            "retrieval_method": (retrieval_metadata.get("retrieval_method", "unknown") if retrieval_metadata else "unknown"),
        }

    query_emb = np.array(all_embs[0], dtype=np.float32)
    answer_emb = np.array(all_embs[1], dtype=np.float32)
    chunk_embs = [np.array(all_embs[i], dtype=np.float32) for i in range(chunk_start, chunk_end)]
    sent_embs = [np.array(all_embs[i], dtype=np.float32) for i in range(sent_start, sent_start + len(answer_sentences))]

    # --- Context Relevance ---
    per_chunk_scores = []
    for ce in chunk_embs:
        sim = float(np.dot(query_emb, ce))
        per_chunk_scores.append(round(sim, 4))
    ctx_relevance = round(sum(per_chunk_scores) / len(per_chunk_scores), 4) if per_chunk_scores else 0.0
    precision = round(sum(1 for s in per_chunk_scores if s >= 0.5) / len(per_chunk_scores), 4) if per_chunk_scores else 0.0

    # --- Chunk Diversity ---
    if len(chunk_embs) >= 2:
        sims = []
        for i in range(len(chunk_embs)):
            for j in range(i + 1, len(chunk_embs)):
                sims.append(float(np.dot(chunk_embs[i], chunk_embs[j])))
        diversity = round(max(0.0, 1.0 - sum(sims) / len(sims)), 4)
    else:
        diversity = 1.0

    # --- Faithfulness ---
    if answer_sentences:
        context_full = " ".join(c.get("text", "") for c in chunks)
        context_words = normalize_text(context_full)
        supported = 0
        for i, sentence in enumerate(answer_sentences):
            words = normalize_text(sentence)
            if words:
                overlap = len(words & context_words) / len(words)
                if overlap >= 0.3:
                    supported += 1
                    continue
            if i < len(sent_embs) and chunk_embs:
                best_sim = max(float(np.dot(sent_embs[i], ce)) for ce in chunk_embs)
                if best_sim >= 0.5:
                    supported += 1
        faith = round(supported / len(answer_sentences), 4)
    else:
        faith = 1.0

    # --- Answer Relevance ---
    ans_relevance = round(max(0.0, float(np.dot(query_emb, answer_emb))), 4)

    # --- Latency ---
    latency_check = latency_budget_check(total_latency_ms)
    latency_score = 1.0 if latency_check["met_target"] else max(0.0, 1.0 - latency_check["overshoot_pct"] / 500)
    overall = 0.30 * ctx_relevance + 0.35 * faith + 0.25 * ans_relevance + 0.10 * latency_score

    eval_ms = (time.perf_counter() - t0) * 1000

    return {
        "retrieval": {
            "context_relevance": ctx_relevance,
            "per_chunk_scores": per_chunk_scores,
            "precision": precision,
            "diversity": diversity,
        },
        "generation": {
            "faithfulness": faith,
            "answer_relevance": ans_relevance,
        },
        "user_experience": latency_check,
        "overall_quality": round(overall, 4),
        "eval_latency_ms": round(eval_ms, 2),
        "retrieval_method": (
            retrieval_metadata.get("retrieval_method", "unknown")
            if retrieval_metadata else "unknown"
        ),
    }

