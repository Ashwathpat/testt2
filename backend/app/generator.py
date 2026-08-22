import os
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=api_key) if api_key else None

MODEL = "groq/compound-mini"


def generate_answer(
    question: str,
    retrieved_context: str,
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

    system_prompt = """You are an intelligent Multilingual Voice RAG assistant.

CRITICAL INSTRUCTIONS (OBEY STRICTLY):
1. Output the direct answer immediately without preamble, scratchpads, or conversational fluff.
2. Complete every sentence cleanly. Never cut off mid-sentence.
3. Answer in the EXACT SAME LANGUAGE as the USER QUESTION (translate from context if needed).
4. State facts directly without apologies or refusals.
5. Keep the answer extremely concise and brief (1-2 short sentences max).
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
            max_tokens=25
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

    system_prompt = f"""You are an intelligent Voice RAG assistant. Answer ONLY in {target_lang}.

CRITICAL INSTRUCTIONS (OBEY STRICTLY):
1. Output the direct answer immediately without preamble, scratchpads, or conversational fluff.
2. Complete every sentence cleanly. Never cut off mid-sentence.
3. You MUST respond ONLY in {target_lang}. If the context is in another language, translate the facts into {target_lang}.
4. State facts directly without apologies or refusals.
5. Keep the answer extremely concise and brief (1-2 short sentences max)."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"USER QUESTION ({target_lang}):\n{question}\n\n"
                        f"RETRIEVED CONTEXT (may be in a different language — translate facts, respond ONLY in {target_lang}):\n{retrieved_context}"
                    )
                }
            ],
            temperature=0.1,
            max_tokens=25,
            stream=True
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
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
    ]
    return any(phrase in normalized for phrase in refusal_phrases)