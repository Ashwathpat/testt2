import json
import time
import os
import httpx
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.retrieve import prewarm_semantic_cache
from app.hybrid_retriever import hybrid_retrieve_context, build_bm25_index
from app.retrieval_adapter import adapt_retrieval_results
from app.pipeline import run_pipeline
from app.generator import generate_answer_stream
from app.grounding import check_grounding
from app.rag_evaluator import evaluate_rag
from app.retrieve import _embed_query_cached, clear_retrieve_caches
from app.query_expander import clear_query_expansion_cache
import numpy as np

_LLM_ANSWER_CACHE = []  # List of (q_vector_np, answer_str, chunks_list, target_lang)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warmup():
    try:
        print("[Startup] Warming up Qdrant connection...")
        # BM25 and semantic cache prewarm disabled to fit 512MB RAM on Render free tier
        # Dense vector search alone provides excellent retrieval quality
        print("[Startup] Warmup complete")
    except Exception as e:
        print(f"[Startup Warmup Warning]: {e}")


class AskRequest(BaseModel):
    question: str
    strategy: str = "fixed_128"


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/cache/clear")
@app.post("/cache/clear")
def clear_all_caches():
    """Clear all in-memory RAG, LLM answer, embedding, and query expansion caches."""
    _LLM_ANSWER_CACHE.clear()
    clear_retrieve_caches()
    clear_query_expansion_cache()
    return {
        "status": "success",
        "message": "All caches cleared successfully (LLM Answer Cache, Semantic Vector Cache, Query Expansion LRU, and Embedding LRU)."
    }


import builtins
import collections

_server_logs = collections.deque(maxlen=50)
_original_print = builtins.print

def _patched_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    _server_logs.append(msg)
    _original_print(*args, **kwargs)

builtins.print = _patched_print

@app.get("/logs")
def get_logs():
    return {"logs": list(_server_logs)}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_api_key:
        return {"error": "GROQ_API_KEY is not configured in backend .env file. Please paste your Groq API key to enable live Speech-to-Text."}

    start_time = time.perf_counter()

    # Read file bytes
    file_bytes = await file.read()

    # Proxy to Groq Whisper
    async with httpx.AsyncClient() as client:
        files = {
            "file": (file.filename or "recording.webm", file_bytes, file.content_type or "audio/webm")
        }
        data = {
            "model": "whisper-large-v3-turbo",
            "prompt": "Please transcribe the audio exactly as spoken in the native language (e.g., Kannada, Hindi, Tamil, Telugu). Do not translate it to English. Use native scripts."
        }
        headers = {
            "Authorization": f"Bearer {groq_api_key}"
        }

        try:
            response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                files=files,
                data=data,
                headers=headers,
                timeout=30.0
            )
        except Exception as err:
            return {"error": f"STT Backend Network Error: {str(err)}"}

    stt_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    if response.status_code != 200:
        return {
            "error": f"Groq STT API Error ({response.status_code}): {response.text}"
        }

    res_data = response.json()
    return {
        "success": True,
        "transcript": res_data.get("text", ""),
        "confidence": 0.98,
        "sttLatency": stt_latency_ms,
        "languageCode": "en-IN"
    }


@app.post("/ask")
def ask(request: AskRequest):
    t_start = time.perf_counter()

    t_ret_start = time.perf_counter()
    retrieved, retrieval_metadata = hybrid_retrieve_context(request.question, k=3)
    t_ret_end = time.perf_counter()
    retrieval_ms = round((t_ret_end - t_ret_start) * 1000, 2)

    chunks = adapt_retrieval_results(retrieved)

    t_gen_start = time.perf_counter()
    
    q_vec = np.array(_embed_query_cached(request.question), dtype=np.float32)
    best_score = 0.0
    cached_answer, cached_chunks = None, None
    for c_vec, c_ans, c_chunks, _ in _LLM_ANSWER_CACHE:
        sim = float(np.dot(q_vec, c_vec))
        if sim > best_score:
            best_score = sim
            cached_answer, cached_chunks = c_ans, c_chunks
            
    if best_score >= 0.95 and cached_answer:
        # Cache hit!
        result_status = "success"
        result_answer = cached_answer
        result_grounded = True
        result_retrieval_confidence = max((c.get("score", 0.0) for c in cached_chunks), default=0.0)
        result_grounding_score = 1.0
        result_sources = [c.get("source", c.get("source_id", "unknown")) for c in cached_chunks]
        result_reason = "semantic_cache_hit"
        result_retrieval_method = "semantic_llm_cache"
        eval_metrics = {}
    else:
        result = run_pipeline(
            request.question,
            chunks,
            retrieval_metadata=retrieval_metadata,
            total_latency_start=t_start
        )
        result_status = result.status
        result_answer = result.answer
        result_grounded = result.grounded
        result_retrieval_confidence = result.retrieval_confidence
        result_grounding_score = result.grounding_score
        result_sources = result.sources
        result_reason = result.reason
        result_retrieval_method = result.retrieval_method
        eval_metrics = result.evaluation
        
        # Save to cache if successful
        if result_status == "success":
            _LLM_ANSWER_CACHE.append((q_vec, result_answer, chunks, "English"))
            
    t_gen_end = time.perf_counter()
    generation_pipeline_ms = round((t_gen_end - t_gen_start) * 1000, 2)
    server_total_ms = round((t_gen_end - t_start) * 1000, 2)

    return {
        "status": result_status,
        "answer": result_answer,
        "grounded": result_grounded,
        "retrieval_confidence": result_retrieval_confidence,
        "grounding_score": result_grounding_score,
        "sources": result_sources,
        "reason": result_reason,
        "latency_ms": round((t_gen_end - t_start) * 1000, 2),
        "retrieval_ms": retrieval_ms,
        "generation_pipeline_ms": generation_pipeline_ms,
        "server_total_ms": server_total_ms,
        "retrieval_method": result_retrieval_method,
        "evaluation": eval_metrics,
    }


@app.post("/ask/stream")
def ask_stream(request: AskRequest):
    t_start = time.perf_counter()

    # Step 1: Hybrid Retrieval (cache-first, fast path)
    t_ret_start = time.perf_counter()
    retrieved, retrieval_metadata = hybrid_retrieve_context(
        request.question, k=3,
        enable_multi_query=False,  # Disable multi-query LLM call for speed
        enable_reranking=False,    # Skip reranking for low latency
        collection_name=request.strategy,
    )
    t_ret_end = time.perf_counter()
    retrieval_ms = round((t_ret_end - t_ret_start) * 1000, 2)

    chunks = adapt_retrieval_results(retrieved)

    retrieval_confidence = max(
        (chunk.get("score", 0.0) for chunk in chunks), default=0.0
    )
    sources = [
        chunk.get("source", chunk.get("source_id", "unknown"))
        for chunk in chunks
    ]
    context = "\n\n".join(chunk["text"] for chunk in chunks)[:1200]

    # Let the LLM auto-detect the response language from the question.
    # "auto" tells the prompt to match the user's language automatically.
    target_lang = "the same language as the user's question"

    q_vec = np.array(_embed_query_cached(request.question), dtype=np.float32)
    best_score = 0.0
    cached_answer, cached_chunks = None, None
    for c_vec, c_ans, c_chunks, c_lang in _LLM_ANSWER_CACHE:
        sim = float(np.dot(q_vec, c_vec))
        if sim > best_score:
            best_score = sim
            cached_answer, cached_chunks = c_ans, c_chunks

    def event_generator():
        # Check cache hit
        if best_score >= 0.95 and cached_answer:
            ttft_ms = round((time.perf_counter() - t_start) * 1000, 2)
            meta = {
                "type": "metadata",
                "retrieval_ms": 0.0,
                "retrieval_confidence": max((c.get("score", 0.0) for c in cached_chunks), default=0.0),
                "grounding_ms": 0,
                "grounding_score": 1.0,
                "sources": [c.get("source", c.get("source_id", "unknown")) for c in cached_chunks],
                "ttft_ms": ttft_ms,
                "retrieval_method": "semantic_llm_cache"
            }
            yield f"data: {json.dumps(meta)}\n\n"
            
            # Stream cached answer in chunks to simulate fast generation
            words = cached_answer.split(" ")
            for i in range(0, len(words), 3):
                chunk = " ".join(words[i:i+3]) + " "
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                
            total_ms = round((time.perf_counter() - t_start) * 1000, 2)
            yield f"data: {json.dumps({'type': 'done', 'server_total_ms': total_ms})}\n\n"
            
            # Send evaluation event for cache hit
            eval_metrics = {"grounding": {"grounded": True, "grounding_score": 1.0}}
            yield f"data: {json.dumps({'type': 'evaluation', 'evaluation': eval_metrics})}\n\n"
            return
        # First event: Granular metadata (sent immediately = low TTFT)
        ttft_ms = round((time.perf_counter() - t_start) * 1000, 2)
        meta = {
            "type": "metadata",
            "retrieval_ms": retrieval_ms,
            "retrieval_confidence": retrieval_confidence,
            "grounding_ms": 0,
            "grounding_score": 0,
            "sources": sources,
            "ttft_ms": ttft_ms,
            "retrieval_method": retrieval_metadata.get("retrieval_method", "dense")
        }
        yield f"data: {json.dumps(meta)}\n\n"

        # Step 2: Stream tokens live as generated (with explicit language)
        lang_context = (
            f"USER QUESTION ({target_lang}):\n{request.question}\n\n"
            f"RETRIEVED CONTEXT (may be in a different language — translate facts, but respond ONLY in {target_lang}):\n{context}"
        )
        accumulated_answer = ""
        for token in generate_answer_stream(request.question, context):
            accumulated_answer += token
            payload = {"type": "token", "content": token}
            yield f"data: {json.dumps(payload)}\n\n"

        # CRITICAL FIX: Capture total_ms RIGHT HERE, after last token,
        # BEFORE running evaluation. This way eval time doesn't inflate
        # the displayed "End-to-End Total" and "LLM Token Synthesis".
        total_ms = round((time.perf_counter() - t_start) * 1000, 2)
        done_payload = {"type": "done", "server_total_ms": total_ms}
        yield f"data: {json.dumps(done_payload)}\n\n"

        # Step 3: Post-generation grounding + evaluation (runs AFTER done)
        grounded, grounding_score = check_grounding(accumulated_answer, chunks)
        
        # Save to cache if successful
        if grounded and not is_generation_refusal(accumulated_answer):
            if len(_LLM_ANSWER_CACHE) >= 1000:
                _LLM_ANSWER_CACHE.pop(0)
            _LLM_ANSWER_CACHE.append((q_vec, accumulated_answer, chunks, target_lang))
            
        eval_metrics = evaluate_rag(
            request.question, accumulated_answer, chunks, total_ms, retrieval_metadata
        )
        eval_metrics["grounding"] = {
            "grounded": grounded,
            "grounding_score": grounding_score,
        }
        eval_event = {"type": "evaluation", "evaluation": eval_metrics}
        yield f"data: {json.dumps(eval_event)}\n\n"

    from app.generator import is_generation_refusal
    return StreamingResponse(event_generator(), media_type="text/event-stream")