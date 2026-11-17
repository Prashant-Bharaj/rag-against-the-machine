"""BM25 retrieval with dual-index (code + docs) support."""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import bm25s
import numpy as np
import numpy.typing as npt

from src.indexer import load_index
from src.models import Chunk, MinimalSource
from src.tokenizer import tokenize

CANDIDATE_FACTOR = 4
MIN_CANDIDATES = 50
SECONDARY_PATH_WEIGHT = 0.5
_SECONDARY_PATH = re.compile(r"vllm-[\d.]+/(tests|benchmarks)/")

# Module-level singletons — loaded once per process
_code_retriever: Optional[bm25s.BM25] = None
_docs_retriever: Optional[bm25s.BM25] = None
_code_chunks: Optional[List[Chunk]] = None
_docs_chunks: Optional[List[Chunk]] = None
_loaded_path: Optional[str] = None


def _ensure_loaded(processed_path: str) -> None:
    """Load indexes if not already loaded or if path changed.

    Args:
        processed_path: Directory containing BM25 indexes.
    """
    global _code_retriever, _docs_retriever
    global _code_chunks, _docs_chunks, _loaded_path

    resolved = str(Path(processed_path).resolve())
    if _code_retriever is not None and _loaded_path == resolved:
        return

    print(f"Loading indexes from {resolved}...")
    (
        _code_retriever,
        _code_chunks,
        _docs_retriever,
        _docs_chunks,
    ) = load_index(resolved)
    _loaded_path = resolved
    assert _code_chunks is not None
    assert _docs_chunks is not None
    print(
        f"Loaded {len(_code_chunks)} code chunks, "
        f"{len(_docs_chunks)} docs chunks"
    )


def _query_index(
    retriever: bm25s.BM25,
    chunks: List[Chunk],
    query: str,
    k: int,
) -> List[Tuple[Chunk, float]]:
    """Query a single BM25 index and return ranked (chunk, score) pairs.

    Args:
        retriever: Fitted BM25 retriever.
        chunks: Chunk list corresponding to the retriever corpus.
        query: Search query string.
        k: Number of results to return.

    Returns:
        List of (Chunk, score) tuples sorted by descending score.
    """
    if k <= 0 or not chunks:
        return []
    # Fetch extra candidates so re-weighting can promote lower-ranked ones.
    candidates = min(max(k * CANDIDATE_FACTOR, MIN_CANDIDATES), len(chunks))

    vocab = retriever.vocab_dict
    query_tokens = [t for t in tokenize(query) if t in vocab]
    if not query_tokens:
        return []

    results, scores = retriever.retrieve(
        [query_tokens], k=candidates, show_progress=False
    )

    indices: npt.NDArray[np.int64] = results[0]
    score_values: npt.NDArray[np.float32] = scores[0]

    pairs: List[Tuple[Chunk, float]] = []
    for idx, score in zip(indices, score_values):
        chunk = chunks[int(idx)]
        pairs.append((chunk, float(score) * path_weight(chunk.file_path)))

    pairs.sort(key=lambda pair: pair[1], reverse=True)
    return pairs[:k]


def path_weight(file_path: str) -> float:
    """Return the score multiplier for a file based on its location.

    Tests and benchmarks repeat the identifiers of the code they
    exercise, so they often outrank the implementation for questions
    about it. They are kept in the index but down-weighted.

    Args:
        file_path: Corpus-relative file path.

    Returns:
        Multiplier applied to the BM25 score.
    """
    if _SECONDARY_PATH.search(file_path):
        return SECONDARY_PATH_WEIGHT
    return 1.0


def _normalize_scores(
    pairs: List[Tuple[Chunk, float]],
) -> List[Tuple[Chunk, float]]:
    """Normalize scores to [0, 1] by dividing by the maximum score.

    Args:
        pairs: List of (chunk, score) tuples.

    Returns:
        List with scores normalized to [0, 1].
    """
    if not pairs:
        return pairs
    max_score = max(s for _, s in pairs)
    if max_score == 0:
        return pairs
    return [(ch, s / max_score) for ch, s in pairs]


def chunk_to_source(chunk: Chunk) -> MinimalSource:
    """Convert a Chunk to a MinimalSource output model.

    Args:
        chunk: Internal chunk model.

    Returns:
        MinimalSource with file_path and character indices.
    """
    return MinimalSource(
        file_path=chunk.file_path,
        first_character_index=chunk.start_char,
        last_character_index=chunk.end_char,
    )


def search(
    query: str,
    k: int,
    processed_path: str,
    index_type: str = "both",
) -> List[MinimalSource]:
    """Search the indexed knowledge base for relevant chunks.

    Args:
        query: Natural language search query.
        k: Number of top results to return.
        processed_path: Directory containing BM25 indexes.
        index_type: Which index to search — 'code', 'docs', or 'both'.

    Returns:
        List of up to k MinimalSource objects sorted by relevance.
    """
    _ensure_loaded(processed_path)

    assert _code_retriever is not None
    assert _docs_retriever is not None
    assert _code_chunks is not None
    assert _docs_chunks is not None

    all_pairs: List[Tuple[Chunk, float]] = []

    if index_type in ("code", "both"):
        code_pairs = _query_index(_code_retriever, _code_chunks, query, k)
        all_pairs.extend(_normalize_scores(code_pairs))

    if index_type in ("docs", "both"):
        docs_pairs = _query_index(_docs_retriever, _docs_chunks, query, k)
        all_pairs.extend(_normalize_scores(docs_pairs))

    # Sort by normalized score descending, take top k
    all_pairs.sort(key=lambda x: x[1], reverse=True)
    top = all_pairs[:k]

    return [chunk_to_source(ch) for ch, _ in top]


def get_chunk_content(
    source: MinimalSource,
    processed_path: str,
) -> Optional[str]:
    """Retrieve the text content for a MinimalSource from disk.

    Args:
        source: A MinimalSource referencing a file and character range.
        processed_path: Base directory — used to locate repo root if needed.

    Returns:
        The chunk text, or None if not found in the chunk registry.
    """
    matches = get_chunks_for_sources([source], processed_path)
    return matches[0].content if matches else None


def get_chunks_for_sources(
    sources: List[MinimalSource],
    processed_path: str,
) -> List[Chunk]:
    """Retrieve Chunk objects matching a list of MinimalSource references.

    Args:
        sources: List of MinimalSource objects.
        processed_path: Directory containing BM25 indexes.

    Returns:
        List of matching Chunk objects (in the same order as sources).
    """
    _ensure_loaded(processed_path)

    assert _code_chunks is not None
    assert _docs_chunks is not None

    # Build lookup: (file_path, start_char, end_char) -> Chunk
    lookup: dict[tuple[str, int, int], Chunk] = {}
    for chunk_list in (_code_chunks, _docs_chunks):
        for ch in chunk_list:
            key = (ch.file_path, ch.start_char, ch.end_char)
            lookup[key] = ch

    result: List[Chunk] = []
    for src in sources:
        key = (
            src.file_path,
            src.first_character_index,
            src.last_character_index,
        )
        if key in lookup:
            result.append(lookup[key])

    return result
