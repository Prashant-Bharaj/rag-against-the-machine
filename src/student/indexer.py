"""BM25 indexing of the vLLM repository."""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import bm25s
from tqdm import tqdm

from student.chunker import (
    DOCS_EXTENSIONS,
    PYTHON_EXTENSIONS,
    SKIP_EXTENSIONS,
    MAX_FILE_SIZE,
    chunk_file,
)
from student.models import Chunk

CHUNKS_DIR = "chunks"
BM25_DIR = "bm25_index"
CODE_CHUNKS_FILE = "code_chunks.jsonl"
DOCS_CHUNKS_FILE = "docs_chunks.jsonl"
CODE_INDEX_DIR = "code"
DOCS_INDEX_DIR = "docs"


def _collect_files(repo_path: str) -> List[str]:
    """Walk the repo and return all indexable file paths.

    Args:
        repo_path: Root directory of the repository.

    Returns:
        Sorted list of absolute file paths.
    """
    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(repo_path):
        # Skip hidden directories and common non-source dirs
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".")
            and d not in {"__pycache__", ".git", "node_modules", ".venv"}
        ]
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            suffix = Path(fpath).suffix.lower()
            if suffix in SKIP_EXTENSIONS:
                continue
            try:
                if os.path.getsize(fpath) > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue
            if suffix in PYTHON_EXTENSIONS or suffix in DOCS_EXTENSIONS:
                files.append(fpath)
    return sorted(files)


def _build_bm25(chunks: List[Chunk], stopwords: str = "en") -> bm25s.BM25:
    """Build a BM25 index from a list of chunks.

    Args:
        chunks: List of Chunk objects to index.
        stopwords: Stopword language string for bm25s.tokenize, or 'none'.

    Returns:
        A fitted BM25 retriever.
    """
    corpus = [c.content for c in chunks]
    sw: Optional[str] = None if stopwords == "none" else stopwords
    tokenized = bm25s.tokenize(corpus, stopwords=sw, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokenized)
    return retriever


def _save_chunks(chunks: List[Chunk], path: str) -> None:
    """Save chunks to a JSONL file.

    Args:
        chunks: List of Chunk objects.
        path: Output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")


def index(
    repo_path: str,
    processed_path: str,
    max_chunk_size: int = 2000,
) -> None:
    """Index all files in the repository and save BM25 indexes.

    Args:
        repo_path: Root directory of the vLLM repository.
        processed_path: Directory to store processed indexes and chunks.
        max_chunk_size: Maximum characters per chunk.
    """
    repo_path = str(Path(repo_path).resolve())
    processed_path = str(Path(processed_path).resolve())

    chunks_dir = Path(processed_path) / CHUNKS_DIR
    bm25_dir = Path(processed_path) / BM25_DIR
    chunks_dir.mkdir(parents=True, exist_ok=True)
    (bm25_dir / CODE_INDEX_DIR).mkdir(parents=True, exist_ok=True)
    (bm25_dir / DOCS_INDEX_DIR).mkdir(parents=True, exist_ok=True)

    print("Scanning repository files...")
    files = _collect_files(repo_path)
    print(f"Found {len(files)} indexable files")

    code_chunks: List[Chunk] = []
    docs_chunks: List[Chunk] = []
    code_id = 0
    docs_id = 0

    for fpath in tqdm(files, desc="Chunking files"):
        try:
            file_chunks = chunk_file(
                fpath,
                repo_path,
                max_chunk_size,
                chunk_id_start=0,
            )
        except Exception as exc:
            print(f"Warning: skipping {fpath}: {exc}", file=sys.stderr)
            continue

        for ch in file_chunks:
            if ch.index_type == "code":
                code_chunks.append(ch.model_copy(update={"chunk_id": code_id}))
                code_id += 1
            else:
                docs_chunks.append(ch.model_copy(update={"chunk_id": docs_id}))
                docs_id += 1

    print(f"Total chunks: {len(code_chunks)} code, {len(docs_chunks)} docs")

    # Save chunk registries
    code_chunks_path = str(chunks_dir / CODE_CHUNKS_FILE)
    docs_chunks_path = str(chunks_dir / DOCS_CHUNKS_FILE)
    print("Saving chunk registries...")
    _save_chunks(code_chunks, code_chunks_path)
    _save_chunks(docs_chunks, docs_chunks_path)

    # Build and save BM25 indexes
    if code_chunks:
        print("Building BM25 code index...")
        code_retriever = _build_bm25(code_chunks, stopwords="none")
        code_index_path = str(bm25_dir / CODE_INDEX_DIR)
        code_retriever.save(code_index_path, corpus=None)
        print(f"Code index saved to {code_index_path}")

    if docs_chunks:
        print("Building BM25 docs index...")
        docs_retriever = _build_bm25(docs_chunks, stopwords="en")
        docs_index_path = str(bm25_dir / DOCS_INDEX_DIR)
        docs_retriever.save(docs_index_path, corpus=None)
        print(f"Docs index saved to {docs_index_path}")

    print("Ingestion complete! Indices saved under", processed_path)


def load_chunks(chunks_path: str) -> List[Chunk]:
    """Load chunks from a JSONL file.

    Args:
        chunks_path: Path to the JSONL chunk file.

    Returns:
        List of Chunk objects in order.
    """
    chunks: List[Chunk] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk.model_validate_json(line))
    return chunks


def load_index(
    processed_path: str,
) -> Tuple[bm25s.BM25, List[Chunk], bm25s.BM25, List[Chunk]]:
    """Load both BM25 indexes and chunk registries from disk.

    Args:
        processed_path: Directory containing processed indexes.

    Returns:
        Tuple of (code_retriever, code_chunks, docs_retriever, docs_chunks).

    Raises:
        FileNotFoundError: If index files are missing.
    """
    processed = Path(processed_path)
    code_index_path = str(processed / BM25_DIR / CODE_INDEX_DIR)
    docs_index_path = str(processed / BM25_DIR / DOCS_INDEX_DIR)
    code_chunks_path = str(processed / CHUNKS_DIR / CODE_CHUNKS_FILE)
    docs_chunks_path = str(processed / CHUNKS_DIR / DOCS_CHUNKS_FILE)

    required_paths = [
        code_index_path,
        docs_index_path,
        code_chunks_path,
        docs_chunks_path,
    ]
    for p in required_paths:
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Index file not found: {p}. Run 'index' command first."
            )

    code_retriever = bm25s.BM25.load(code_index_path, load_corpus=False)
    docs_retriever = bm25s.BM25.load(docs_index_path, load_corpus=False)
    code_chunks = load_chunks(code_chunks_path)
    docs_chunks = load_chunks(docs_chunks_path)

    return code_retriever, code_chunks, docs_retriever, docs_chunks
