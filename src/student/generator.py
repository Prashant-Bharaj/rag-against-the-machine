"""Answer generation with latency-oriented backend strategies."""

import json
import os
import urllib.error
import urllib.request
from typing import Any, List, Optional

from student.models import Chunk

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 200
MAX_CONTEXT_CHARS = 3000
MAX_LATENCY_SECONDS = float(os.getenv("STUDENT_MAX_LATENCY_SECONDS", "1.8"))

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the vLLM codebase. "
    "Answer concisely based only on the provided context. /no_think"
)

# Module-level singleton
_pipeline: Optional[Any] = None


def _normalize_backend(raw: Optional[str]) -> str:
    """Return normalized backend identifier."""
    backend = (raw or "").strip().lower()
    if backend in {"", "auto"}:
        return "auto"
    if backend in {"extractive", "hf_local", "openai_compatible"}:
        return backend
    return "auto"


def _get_pipeline() -> Any:
    """Load the generation pipeline lazily (once per process).

    Returns:
        Loaded transformers pipeline.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import torch
    from transformers import pipeline

    device_map = "auto"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model {MODEL_NAME}...")
    _pipeline = pipeline(
        "text-generation",
        model=MODEL_NAME,
        torch_dtype=dtype,
        device_map=device_map,
    )
    print("Model loaded.")
    return _pipeline


def _build_prompt(question: str, chunks: List[Chunk]) -> str:
    """Build the prompt string from question and context chunks.

    Args:
        question: The user's question.
        chunks: Retrieved context chunks.

    Returns:
        Formatted prompt string.
    """
    context_parts: List[str] = []
    total_chars = 0

    for chunk in chunks:
        snippet = f"# {chunk.file_path}\n{chunk.content}\n"
        if total_chars + len(snippet) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(snippet)
        total_chars += len(snippet)

    context = "\n---\n".join(context_parts)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def _extractive_answer(question: str, chunks: List[Chunk]) -> str:
    """Fast fallback that extracts concise evidence from retrieved chunks."""
    if not chunks:
        return "I could not find supporting context in the retrieved sources."

    q_tokens = {
        tok.lower()
        for tok in question.replace("?", " ").replace(",", " ").split()
        if len(tok) > 2
    }

    best_snippet = ""
    best_overlap = -1
    for chunk in chunks[:5]:
        for line in chunk.content.splitlines():
            text = line.strip()
            if not text:
                continue
            line_tokens = {
                tok.lower()
                for tok in text.replace("`", " ").replace("_", " ").split()
                if len(tok) > 2
            }
            overlap = len(q_tokens & line_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_snippet = text

    if not best_snippet:
        best_snippet = chunks[0].content.strip().splitlines()[0]

    if len(best_snippet) > 320:
        best_snippet = best_snippet[:317].rstrip() + "..."

    return best_snippet


def _generate_openai_compatible(
    question: str,
    chunks: List[Chunk],
) -> Optional[str]:
    """Generate via OpenAI-compatible serving endpoint (e.g., vLLM server)."""
    base_url = os.getenv("STUDENT_OPENAI_BASE_URL", "").rstrip("/")
    if not base_url:
        return None

    model_name = os.getenv("STUDENT_OPENAI_MODEL", MODEL_NAME)
    endpoint = f"{base_url}/chat/completions"
    context = _build_prompt(question, chunks)
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        "temperature": 0,
        "max_tokens": 120,
    }
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout_s = max(0.2, MAX_LATENCY_SECONDS)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
        body = json.loads(raw)
        content = body["choices"][0]["message"]["content"].strip()
        return content or None
    except (
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        urllib.error.URLError,
    ):
        return None


def _generate_hf_local(question: str, chunks: List[Chunk]) -> str:
    """Generate with local transformers pipeline (higher latency)."""
    pipe = _get_pipeline()
    prompt = _build_prompt(question, chunks)

    outputs = pipe(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=pipe.tokenizer.eos_token_id,
        return_full_text=False,
    )

    answer: str = outputs[0]["generated_text"].strip()

    if "<think>" in answer:
        end_think = answer.find("</think>")
        if end_think != -1:
            answer = answer[end_think + len("</think>"):].strip()
    return answer


def generate(question: str, chunks: List[Chunk]) -> str:
    """Generate an answer with latency-aware backend selection.

    Args:
        question: The user's question.
        chunks: Relevant chunks retrieved from the knowledge base.

    Returns:
        Generated answer string.
    """
    backend = _normalize_backend(os.getenv("STUDENT_GENERATION_BACKEND"))

    if backend == "extractive":
        return _extractive_answer(question, chunks)

    if backend == "openai_compatible":
        result = _generate_openai_compatible(question, chunks)
        return result if result else _extractive_answer(question, chunks)

    if backend == "hf_local":
        return _generate_hf_local(question, chunks)

    # auto: prefer low-latency endpoint, otherwise safe extractive fallback.
    if os.getenv("STUDENT_OPENAI_BASE_URL"):
        result = _generate_openai_compatible(question, chunks)
        if result:
            return result
    return _extractive_answer(question, chunks)
