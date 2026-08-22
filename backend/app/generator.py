import os
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=api_key) if api_key else None

# openai/gpt-oss-20b: fast reasoning model on Groq, requires reasoning_effort+reasoning_format params
MODEL = "openai/gpt-oss-20b"


def generate_answer(
    question: str,
    retrieved_context: str,
    target_lang: str = "English",
    retry: bool = False
) -> str:
    global client
    if not client:
        api_key_check = os.getenv("GROQ_API_KEY", "")
        if api_key_check:
            client = Groq(api_key=api_key_check)
        else:
            if retrieved_context.strip():
                return f"Based on retrieved context:\n{retrieved_context[:500]}"
            return "I do not have enough information to answer."

    system_prompt = f"""You are a grounded Multilingual Voice RAG assistant. Answer ONLY in {target_lang}.

CRITICAL INSTRUCTIONS (OBEY STRICTLY):
1. ONLY answer using facts found in the RETRIEVED CONTEXT below. Do NOT use your own knowledge.
2. If the user's question asks for something unsafe, unethical, illegal, or inappropriate (like weapons, violence, harm), respond EXACTLY with: "I cannot fulfill this request."
3. If the RETRIEVED CONTEXT does not contain information to answer the question, respond EXACTLY with: "I do not have enough information to answer this question based on the available context."
4. Output the direct answer immediately without preamble, scratchpads, or conversational fluff.
5. Complete every sentence cleanly. Never cut off mid-sentence.
6. You MUST respond ONLY in {target_lang}. If the retrieved context is in a different language, translate the facts into {target_lang}.
7. Keep the answer concise (2-4 sentences max).
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"USER QUESTION:\n{question}\n\n"
                        f"RETRIEVED CONTEXT:\n{retrieved_context}"
                    )
                }
            ],
            temperature=0.1,
            max_tokens=300,
            reasoning_effort="low",
            reasoning_format="hidden"
        )

        answer = response.choices[0].message.content
        return answer.strip() if answer else ""
    except Exception as e:
        print(f"[Groq Generator Error]: {e}")
        if retrieved_context.strip():
            return f"Retrieved context summary:\n{retrieved_context[:500]}"
        return "I do not have enough information to answer."


def generate_answer_stream(
    question: str,
    retrieved_context: str,
    target_lang: str = "English"
):
    global client
    if not client:
        api_key_check = os.getenv("GROQ_API_KEY", "")
        if api_key_check:
            client = Groq(api_key=api_key_check)
        else:
            if retrieved_context.strip():
                yield f"Based on retrieved context:\n{retrieved_context[:500]}"
            else:
                yield "I do not have enough information to answer."
            return

    system_prompt = f"""You are a grounded Multilingual Voice RAG assistant. Answer ONLY in {target_lang}.

CRITICAL INSTRUCTIONS (OBEY STRICTLY):
1. ONLY answer using facts found in the RETRIEVED CONTEXT below. Do NOT use your own knowledge.
2. If the user's question asks for something unsafe, unethical, illegal, or inappropriate (like weapons, violence, harm), respond EXACTLY with: "I cannot fulfill this request."
3. If the RETRIEVED CONTEXT does not contain information to answer the question, respond EXACTLY with: "I do not have enough information to answer this question based on the available context."
4. Output the direct answer immediately without preamble, scratchpads, or conversational fluff.
5. Complete every sentence cleanly. Never cut off mid-sentence.
6. You MUST respond ONLY in {target_lang}. If the retrieved context is in a different language, translate the facts into {target_lang}.
7. Keep the answer concise (2-4 sentences max)."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"USER QUESTION:\n{question}\n\n"
                        f"RETRIEVED CONTEXT:\n{retrieved_context}"
                    )
                }
            ],
            temperature=0.1,
            max_tokens=300,
            reasoning_effort="low",
            reasoning_format="hidden",
            stream=True
        )

        has_yielded = False
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                has_yielded = True

        if not has_yielded:
            if retrieved_context.strip():
                yield f"Based on retrieved context:\n{retrieved_context[:500]}"
            else:
                yield "I do not have enough information to answer."

    except Exception as e:
        print(f"[Groq Stream Error]: {e}")
        if retrieved_context.strip():
            yield f"Retrieved context summary:\n{retrieved_context[:500]}"
        else:
            yield "I do not have enough information to answer."


def is_generation_refusal(answer: str) -> bool:
    if not answer:
        return True

    normalized = answer.lower().strip()
    refusal_phrases = [
        "i don't have enough information",
        "i do not have enough information",
        "not enough information",
        "i can't provide",
        "i cannot provide",
        "i'm sorry",
        "unable to provide",
        "i cannot fulfill this request",
    ]
    return any(phrase in normalized for phrase in refusal_phrases)