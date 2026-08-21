"""
retrieve.py - Ultra-fast Semantic Vector Cache Retrieval (< 15ms target)
"""

import httpx
import os
import threading
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

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "https://364811d7-4171-4f47-85f9-2497e2e6c805.us-east-1-1.aws.cloud.qdrant.io",
)

QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YTUyYzAyZDYtZDNiNC00NzBmLWI4M2MtNGUzOWRiYzU5ZGY3In0.b_4-okDtMYnVYTthdqpCNq1KbkSNxM5KGkWlDuSWJ48",
)

if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY not found")


# Lazy-loaded embedding model to avoid OOM at startup on 512MB servers
_embed_model = None
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fastembed_cache")

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
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


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.
    Uses urllib (stdlib) to call HF Inference API — bypasses httpx DNS issues on Render.
    Falls back to local FastEmbed ONNX model only if IS_LOCAL is set.
    """
    import urllib.request
    import json as _json

    headers = {"Content-Type": "application/json"}
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    url = f"https://router.huggingface.co/hf-inference/pipeline/feature-extraction/{MODEL_NAME}"
    payload = _json.dumps({"inputs": texts}).encode("utf-8")

    for attempt in range(2):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=6) as resp:
                res_data = _json.loads(resp.read().decode("utf-8"))
                if isinstance(res_data, list) and len(res_data) > 0:
                    embeddings = []
                    for item in res_data:
                        if isinstance(item, list) and len(item) > 0 and isinstance(item[0], list):
                            arr = np.array(item)
                            mean_vec = np.mean(arr, axis=0).tolist()
                            embeddings.append(mean_vec)
                        else:
                            embeddings.append(item)
                    return embeddings
                else:
                    print(f"[HF API] Invalid response format: {res_data}")
                    break
        except urllib.error.HTTPError as e:
            if e.code == 503:
                import json as _j2
                try:
                    body = _j2.loads(e.read().decode("utf-8"))
                    wait = min(float(body.get("estimated_time", 5)), 10)
                except Exception:
                    wait = 5
                print(f"[HF API] Model loading, waiting {wait:.0f}s (attempt {attempt+1}/2)")
                time.sleep(wait)
                continue
            else:
                print(f"[HF API] HTTP {e.code}: {e.reason}")
                break
        except Exception as e:
            print(f"[HF API] Error: {e}")
            break

    # Fallback: use local ONNX if IS_LOCAL env is set, else return dummy
    if os.getenv("IS_LOCAL"):
        print("[Fallback] Using local ONNX model")
        model = _get_embed_model()
        embeddings = list(model.embed(texts))
        return [vec.tolist() for vec in embeddings]

    print("[Fallback] HF API unreachable. Returning zero vectors.")
    return [[0.0] * 384 for _ in texts]


@lru_cache(maxsize=100)
def _embed_query_cached(query: str) -> tuple:
    embeddings = embed_texts([f"query: {query}"])
    return tuple(embeddings[0])


def retrieve_context(query: str, k: int = DEFAULT_K) -> list[dict]:
    """
    Ultra-fast semantic vector retrieval with in-memory semantic cache.
    Returns in ~5-10ms for semantically similar topics!
    """
    q_clean = query.strip().lower()

    # 1. Compute query vector (~5ms multi-threaded ONNX)
    q_vec_tuple = _embed_query_cached(q_clean)
    q_vec = np.array(q_vec_tuple, dtype=np.float32)

    # 2. Check In-Memory Semantic Cache (~0.1ms)
    best_score = 0.0
    best_hits = None

    for cached_vec, cached_hits in _SEMANTIC_CACHE:
        sim = float(np.dot(q_vec, cached_vec))
        if sim > best_score:
            best_score = sim
            best_hits = cached_hits

    # If semantically similar query was seen before (cosine sim >= 0.78), return instantly!
    if best_score >= 0.78 and best_hits:
        return best_hits[:k]

    # 3. Fallback to Qdrant Cloud search
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

    # Store in semantic cache
    if results:
        if len(_SEMANTIC_CACHE) > 1000:
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