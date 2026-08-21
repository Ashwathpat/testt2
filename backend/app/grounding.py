import re
from app.retrieve import embed_model as embedding_model


def normalize_text(text: str) -> set[str]:
    # Support Unicode / Indic / Devanagari word extraction
    words = re.findall(
        r"\w+",
        text.lower(),
        flags=re.UNICODE
    )
    return set(words)


def lexical_overlap(
    answer: str,
    context: str
) -> float:

    answer_words = normalize_text(answer)
    context_words = normalize_text(context)

    if not answer_words:
        return 0.0

    overlap = answer_words.intersection(context_words)

    return len(overlap) / len(answer_words)


def split_sentences(text: str) -> list[str]:
    """
    Split generated answer into individual claims/sentences.
    """

    sentences = re.split(
        r"(?<=[.!?|।])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def check_grounding(
    answer: str,
    retrieved_chunks: list[dict],
    min_overlap: float = 0.25,
    min_semantic_similarity: float = 0.50
) -> tuple[bool, float]:

    if not answer.strip():
        return False, 0.0

    if not retrieved_chunks:
        return False, 0.0

    # --------------------------------
    # 1. Split answer into claims
    # --------------------------------

    claims = split_sentences(answer)

    if not claims:
        return False, 0.0

    valid_chunks = [c.get("text", "") for c in retrieved_chunks if c.get("text", "").strip()]
    if not valid_chunks:
        return False, 0.0

    # --------------------------------
    # 2. Pre-embed all claims and chunks at once
    # --------------------------------
    texts_to_embed = [f"query: {c}" for c in claims] + [f"passage: {c}" for c in valid_chunks]
    all_embs = list(embedding_model.embed(texts_to_embed))
    
    claim_embs = all_embs[:len(claims)]
    chunk_embs = all_embs[len(claims):]

    claim_scores = []
    
    import numpy as np

    # --------------------------------
    # 3. Check every claim
    # --------------------------------

    for i, claim in enumerate(claims):

        best_score = 0.0
        claim_supported = False

        for j, chunk_text in enumerate(valid_chunks):

            # ----------------------------
            # Lexical evidence
            # ----------------------------

            overlap_score = lexical_overlap(
                claim,
                chunk_text
            )

            if overlap_score >= min_overlap:
                claim_supported = True
                best_score = max(
                    best_score,
                    overlap_score
                )
                continue

            # ----------------------------
            # Multilingual semantic evidence
            # ----------------------------

            semantic_score = float(np.dot(claim_embs[i], chunk_embs[j]))

            best_score = max(
                best_score,
                semantic_score
            )

            if semantic_score >= min_semantic_similarity:
                claim_supported = True

        # --------------------------------
        # Claim has no supporting evidence
        # --------------------------------

        if not claim_supported:
            return False, best_score

        claim_scores.append(best_score)

    # --------------------------------
    # 4. Overall grounding score
    # --------------------------------

    grounding_score = min(claim_scores) if claim_scores else 0.0

    return True, grounding_score