"""
query_expander.py - Multi-Query Expansion via LLM

Generates multiple similar queries from a user query to improve retrieval recall.
Architecture from diagram: User Query → Generate Similar Queries → [Query 1..4 + Original]
"""

import os
import re
import time
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


def _get_groq_client():
    """Lazy-init Groq client."""
    try:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY", "")
        if api_key:
            return Groq(api_key=api_key)
    except ImportError:
        pass
    return None


# Cache expanded queries to avoid repeated LLM calls for the same question
def clear_query_expansion_cache():
    """Clear query expansion LRU cache."""
    expand_query.cache_clear()


@lru_cache(maxsize=500)
def expand_query(query: str, n_variations: int = 4) -> list[str]:
    """
    Generate n_variations of similar queries from the original query using LLM.
    Returns: [original_query, variation_1, variation_2, ..., variation_n]

    Falls back to simple keyword expansion if LLM is unavailable.
    """
    variations = [query]  # Always include original

    client = _get_groq_client()
    if not client:
        # Fallback: simple keyword-based expansion
        return _simple_expand(query, n_variations)

    try:
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model="groq/compound-mini",  # Fast model for query expansion
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a search query expansion assistant. "
                        "Given a user query, generate exactly "
                        f"{n_variations} alternative search queries that capture "
                        "different aspects, synonyms, or phrasings of the same intent. "
                        "Output ONLY the queries, one per line, numbered 1-"
                        f"{n_variations}. No explanations."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=200,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        raw = response.choices[0].message.content or ""

        # Parse numbered lines: "1. ..." or "1) ..."
        lines = raw.strip().split("\n")
        for line in lines:
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
            if cleaned and cleaned != query and len(cleaned) > 3:
                variations.append(cleaned)
            if len(variations) >= n_variations + 1:
                break

        print(
            f"[QueryExpander] Generated {len(variations)-1} variations "
            f"in {elapsed_ms:.0f}ms"
        )

    except Exception as e:
        print(f"[QueryExpander] LLM expansion failed: {e}, using simple fallback")
        return _simple_expand(query, n_variations)

    return variations


def _simple_expand(query: str, n_variations: int = 4) -> list[str]:
    """
    Simple keyword-based query expansion fallback (no LLM needed).
    Generates variations by rephrasing with common patterns.
    """
    variations = [query]
    q_lower = query.lower().strip()

    # Variation 1: Add "what is" prefix if not already a question
    if not q_lower.startswith(("what", "how", "why", "when", "where", "who")):
        variations.append(f"What is {q_lower}?")

    # Variation 2: "Explain" prefix
    if not q_lower.startswith("explain"):
        variations.append(f"Explain {q_lower}")

    # Variation 3: "Tell me about" prefix
    variations.append(f"Tell me about {q_lower}")

    # Variation 4: Keywords only (remove stop words)
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on",
        "at", "to", "for", "of", "with", "by", "from", "and", "or",
        "but", "not", "this", "that", "it", "be", "have", "has",
        "do", "does", "did", "will", "would", "can", "could",
        "should", "may", "might", "what", "how", "why", "when",
        "where", "who", "which", "me", "about", "tell",
    }
    keywords = [
        w for w in q_lower.split() if w not in stop_words and len(w) > 2
    ]
    if keywords:
        variations.append(" ".join(keywords))

    return variations[: n_variations + 1]
