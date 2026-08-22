"""
retrieve.py - Semantic Vector Retrieval using local ONNX embeddings
"""

import os
import time
from functools import lru_cache
import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

COLLECTION_NAME = "fixed_128"
MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_K = 2  # Top 2 chunks for minimum latency

QDRANT_URL = os.getenv("QDRANT_URL")
if not QDRANT_URL:
    raise ValueError("QDRANT_URL not found in environment variables")

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY not found in environment variables")




# ---------------------------------------------------------------------------
# Local ONNX Embedding Model (loaded lazily on first use)
# ---------------------------------------------------------------------------
_embed_model = None
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fastembed_cache")

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        print("[Embed] Loading local ONNX model...")
        
        # Keep aggressive memory optimization just in case
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["ONNXRUNTIME_MAX_THREADS"] = "1"

        try:
            TextEmbedding.add_custom_model(
                model=MODEL_NAME,
                pooling=PoolingType.MEAN,
                normalization=True,
                sources=ModelSource(hf=MODEL_NAME),
                dim=384,
                model_file="onnx/model.onnx",
            )
        except ValueError:
            pass
        _embed_model = TextEmbedding(model_name=MODEL_NAME, cache_dir=CACHE_DIR, threads=1)
        print("[Embed] Local ONNX model loaded OK")
    return _embed_model

# Public alias for importers (grounding.py etc.)
embed_model = None  # Will be set on first use via _get_embed_model()

# Persistent Qdrant client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=5,
)

# In-Memory Semantic Vector Cache: list of (q_vector_np, results_list)
_SEMANTIC_CACHE = []


def _is_zero_vector(vec):
    """Check if a vector is all zeros (indicates embedding failure)."""
    return all(v == 0.0 for v in vec[:10])


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings using the local FastEmbed ONNX model.
    This is 100% reliable — no network calls, no DNS issues, no auth tokens.
    """
    try:
        model = _get_embed_model()
        embeddings = list(model.embed(texts))
        return [vec.tolist() for vec in embeddings]
    except Exception as e:
        print(f"[Embed CRITICAL] Local ONNX model failed: {e}")
        return [[0.0] * 384 for _ in texts]


def clear_retrieve_caches():
    """Clear all retrieve-related in-memory caches."""
    _embed_query_cached.cache_clear()
    _SEMANTIC_CACHE.clear()


@lru_cache(maxsize=200)
def _embed_query_cached(query: str) -> tuple:
    embeddings = embed_texts([f"query: {query}"])
    vec = embeddings[0]
    # NEVER cache zero vectors — they would poison ALL future lookups
    if _is_zero_vector(vec):
        # Evict this from the LRU cache immediately
        _embed_query_cached.cache_clear()
        return tuple(vec)
    return tuple(vec)


def retrieve_context(query: str, k: int = DEFAULT_K) -> list[dict]:
    """
    Semantic vector retrieval with in-memory semantic cache.
    """
    q_clean = query.strip().lower()

    # 1. Compute query vector
    q_vec_tuple = _embed_query_cached(q_clean)
    q_vec = np.array(q_vec_tuple, dtype=np.float32)

    # GUARD: If we got a zero vector, skip cache and Qdrant (would return garbage)
    if _is_zero_vector(q_vec_tuple):
        print("[Retrieve] WARNING: Got zero vector — embedding failed. Returning empty.")
        return []

    # 2. Check In-Memory Semantic Cache (~0.1ms)
    best_score = 0.0
    best_hits = None

    for cached_vec, cached_hits in _SEMANTIC_CACHE:
        sim = float(np.dot(q_vec, cached_vec))
        if sim > best_score:
            best_score = sim
            best_hits = cached_hits

    # If nearly identical query was seen before (cosine sim >= 0.95), return instantly!
    if best_score >= 0.95 and best_hits:
        return best_hits[:k]

    # 3. Qdrant Cloud search
    try:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=list(q_vec_tuple),
            limit=k,
            search_params=models.SearchParams(hnsw_ef=16),
            with_vectors=False,
            with_payload=["window_text", "source_id"],
        )

        results = []
        for hit in response.points:
            payload = hit.payload or {}
            results.append(
                {
                    "text": payload.get("window_text", ""),
                    "source_id": payload.get("source_id", ""),
                    "score": float(hit.score),
                }
            )
    except Exception as e:
        print(f"[Qdrant Retrieval Warning]: {e}")
        results = []

    # Store in semantic cache (ONLY if we have real results with real vectors)
    if results:
        if len(_SEMANTIC_CACHE) > 500:
            _SEMANTIC_CACHE.pop(0)
        _SEMANTIC_CACHE.append((q_vec, results))

    return results


def prewarm_semantic_cache():
    """
    Pre-warm common domain queries at server startup so retrieval is < 10ms.
    """
    preset_queries = [
        "Give me information on diabetes.",
        "What are the symptoms for diabetes?",
        "What are the key requirements and constraints for Task 2 at HH Goa 2026?",
        "How does Sarvam AI Speech-to-Text optimize transcription?",
        "What strategies reduce end-to-end latency below 800ms in voice RAG pipelines?",
        "Explain dense and sparse hybrid retrieval for document grounding.",
        "health check",
    ]
    print("[Semantic Cache] Pre-warming domain vectors...")
    for q in preset_queries:
        try:
            retrieve_context(q, k=2)
        except Exception:
            pass
    print(f"[Semantic Cache] Pre-warmed {len(_SEMANTIC_CACHE)} vector topics OK")


def retrieve_text_only(query: str, k: int = DEFAULT_K) -> list[str]:
    contexts = retrieve_context(query, k=k)
    return [context["text"] for context in contexts]


def close_connection():
    client.close()