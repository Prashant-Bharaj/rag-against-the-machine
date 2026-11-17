"""Answer generation with Qwen/Qwen3-0.6B grounded in retrieved chunks."""

import sys
from typing import Any, List, Optional, Tuple

from src.models import Chunk

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 128
MAX_CONTEXT_TOKENS = 1536
MAX_CONTEXT_CHUNKS = 5

NOT_FOUND = "The retrieved context does not contain the answer."
SYSTEM_PROMPT = (
    "You answer questions about the vLLM codebase using only the numbered "
    "context snippets. Rules:\n"
    "1. Answer in one to three sentences.\n"
    "2. Copy names, default values and identifiers exactly as they appear "
    "in the snippets; never invent a value.\n"
    "3. Prefer snippets from the library itself over tests or examples.\n"
    f'4. If no snippet contains the answer, reply exactly "{NOT_FOUND}" '
    "and nothing else."
)

# Module-level singletons: the model is loaded once per process.
_model: Optional[Any] = None
_tokenizer: Optional[Any] = None
_load_failed = False


def _load_model() -> Optional[Tuple[Any, Any]]:
    """Load the Qwen tokenizer and model lazily (once per process).

    Returns:
        ``(tokenizer, model)``, or None if the model cannot be loaded.
    """
    global _model, _tokenizer, _load_failed
    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model
    if _load_failed:
        return None

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        use_cuda = torch.cuda.is_available()
        print(
            f"Loading model {MODEL_NAME} "
            f"on {'GPU' if use_cuda else 'CPU'}...",
            file=sys.stderr,
        )
        tokenizer: Any = AutoTokenizer.from_pretrained(MODEL_NAME)
        model: Any = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16 if use_cuda else torch.float32,
        )
        model = model.to("cuda" if use_cuda else "cpu")
        model.eval()
        # Greedy decoding: drop the sampling defaults shipped with the model.
        for name in ("temperature", "top_p", "top_k"):
            setattr(model.generation_config, name, None)
        _tokenizer, _model = tokenizer, model
    except Exception as exc:
        print(
            f"Warning: could not load {MODEL_NAME} ({exc}); "
            "falling back to extractive answers.",
            file=sys.stderr,
        )
        _load_failed = True
        _model = _tokenizer = None
        return None
    return _tokenizer, _model


def _build_context(tokenizer: Any, chunks: List[Chunk]) -> str:
    """Format retrieved chunks as numbered snippets within a token budget.

    Args:
        tokenizer: Tokenizer used to measure the context length.
        chunks: Retrieved chunks, most relevant first.

    Returns:
        Context string that fits in ``MAX_CONTEXT_TOKENS``.
    """
    parts: List[str] = []
    used = 0
    for i, chunk in enumerate(chunks[:MAX_CONTEXT_CHUNKS], start=1):
        snippet = f"[{i}] {chunk.file_path}\n{chunk.content.strip()}\n"
        ids = tokenizer.encode(snippet, add_special_tokens=False)
        remaining = MAX_CONTEXT_TOKENS - used
        if remaining <= 0:
            break
        if len(ids) > remaining:
            snippet = tokenizer.decode(ids[:remaining])
        parts.append(snippet)
        used += min(len(ids), remaining)
    return "\n".join(parts)


def _strip_thinking(text: str) -> str:
    """Remove any ``<think>...</think>`` block emitted by Qwen3.

    Args:
        text: Raw model output.

    Returns:
        The answer text without the reasoning block.
    """
    end = text.find("</think>")
    if end != -1:
        text = text[end + len("</think>"):]
    return text.replace("<think>", "").strip()


def _generate_qwen(
    tokenizer: Any, model: Any, question: str, chunks: List[Chunk]
) -> str:
    """Generate an answer with the local Qwen model.

    Args:
        tokenizer: Loaded Qwen tokenizer.
        model: Loaded Qwen causal language model.
        question: The user's question.
        chunks: Retrieved context chunks.

    Returns:
        Generated answer text.
    """
    import torch

    context = _build_context(tokenizer, chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return _strip_thinking(text)


def _extractive_answer(question: str, chunks: List[Chunk]) -> str:
    """Return the context line that best overlaps the question.

    Used only when the Qwen model cannot be loaded or fails.

    Args:
        question: The user's question.
        chunks: Retrieved context chunks.

    Returns:
        A short extract from the retrieved context.
    """
    q_tokens = {
        tok.lower()
        for tok in question.replace("?", " ").replace(",", " ").split()
        if len(tok) > 2
    }

    best_snippet = ""
    best_overlap = -1
    for chunk in chunks[:MAX_CONTEXT_CHUNKS]:
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

    if len(best_snippet) > 320:
        best_snippet = best_snippet[:317].rstrip() + "..."
    return best_snippet


def generate(question: str, chunks: List[Chunk]) -> str:
    """Generate an answer to ``question`` grounded in ``chunks``.

    Args:
        question: The user's question.
        chunks: Relevant chunks retrieved from the knowledge base.

    Returns:
        Generated answer string (never raises).
    """
    if not question.strip():
        return "The question is empty."
    if not chunks:
        return "No relevant context was retrieved for this question."

    loaded = _load_model()
    if loaded is not None:
        tokenizer, model = loaded
        try:
            answer = _generate_qwen(tokenizer, model, question, chunks)
            if answer:
                return answer
        except Exception as exc:
            print(f"Warning: generation failed ({exc})", file=sys.stderr)
    return _extractive_answer(question, chunks) or NOT_FOUND
