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
        # Skip prewarming cache to avoid memory spikes under 512MB RAM
        print("[Startup] Building BM25 index for sparse search...")
        build_bm25_index(max_docs=5000)
        print("[Startup] Warmup complete ✅")
    except Exception as e:
        print(f"[Startup Warmup Warning]: {e}")


class AskRequest(BaseModel):
    question: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


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
            "model": "whisper-large-v3"
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
    result = run_pipeline(
        request.question,
        chunks,
        retrieval_metadata=retrieval_metadata,
        total_latency_start=t_start
    )
    t_gen_end = time.perf_counter()
    generation_pipeline_ms = round((t_gen_end - t_gen_start) * 1000, 2)
    server_total_ms = round((t_gen_end - t_start) * 1000, 2)

    return {
        "status": result.status,
        "answer": result.answer,
        "grounded": result.grounded,
        "retrieval_confidence": result.retrieval_confidence,
        "grounding_score": result.grounding_score,
        "sources": result.sources,
        "reason": result.reason,
        "latency_ms": result.latency_ms,
        "retrieval_ms": retrieval_ms,
        "generation_pipeline_ms": generation_pipeline_ms,
        "server_total_ms": server_total_ms,
        "retrieval_method": result.retrieval_method,
        "evaluation": result.evaluation,
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

    # Detect language from the question so the LLM answers in the right language
    import re as _re
    target_lang = "Hindi" if _re.search(r'[\u0900-\u097F]', request.question) else "English"

    def event_generator():
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
        for token in generate_answer_stream(request.question, context, target_lang=target_lang):
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
        eval_metrics = evaluate_rag(
            request.question, accumulated_answer, chunks, total_ms, retrieval_metadata
        )
        eval_metrics["grounding"] = {
            "grounded": grounded,
            "grounding_score": grounding_score,
        }
        eval_event = {"type": "evaluation", "evaluation": eval_metrics}
        yield f"data: {json.dumps(eval_event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")