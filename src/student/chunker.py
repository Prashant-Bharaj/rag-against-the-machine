"""Chunking strategies for Python code and text/Markdown files."""

import ast
import re
from pathlib import Path
from typing import List, Optional

from student.models import Chunk

# File extensions to handle
PYTHON_EXTENSIONS = {".py", ".pyi"}
DOCS_EXTENSIONS = {".md", ".rst", ".txt"}
SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".png", ".jpg", ".jpeg", ".gif", ".ico",
    ".whl", ".zip", ".tar", ".gz", ".lock", ".bin", ".so",
    ".pth", ".npy", ".npz", ".pt", ".ckpt", ".h5", ".pkl",
    ".svg", ".pdf", ".ttf", ".woff", ".eot",
}

# Maximum file size to process (500 KB)
MAX_FILE_SIZE = 500 * 1024

# Sliding window overlap in characters
OVERLAP = 200


def build_line_offsets(source: str) -> List[int]:
    """Build a list mapping line index (0-based) to character offset.

    Args:
        source: The full source text.

    Returns:
        List where index i gives the character offset of line i.
    """
    offsets: List[int] = []
    pos = 0
    for line in source.splitlines(keepends=True):
        offsets.append(pos)
        pos += len(line)
    return offsets


def sliding_window(
    content: str,
    file_path: str,
    start_offset: int,
    max_size: int,
    overlap: int = OVERLAP,
) -> List[Chunk]:
    """Split text using a sliding window with overlap.

    Args:
        content: Text to split.
        file_path: Source file path for chunk metadata.
        start_offset: Character offset of ``content`` within the full file.
        max_size: Maximum chunk size in characters.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of Chunk objects.
    """
    chunks: List[Chunk] = []
    pos = 0
    length = len(content)

    while pos < length:
        end = min(pos + max_size, length)
        chunk_text = content[pos:end]
        if chunk_text.strip():
            chunks.append(
                Chunk(
                    chunk_id=0,
                    file_path=file_path,
                    start_char=start_offset + pos,
                    end_char=start_offset + end,
                    content=chunk_text,
                    index_type="",
                )
            )
        if end == length:
            break
        pos = end - overlap

    return chunks


def chunk_python(
    source: str, file_path: str, max_size: int
) -> List[Chunk]:
    """Chunk Python source using AST node boundaries.

    Extracts top-level functions and classes as separate chunks,
    keeping class methods together with their class. Falls back to
    sliding window when AST parsing fails or a node exceeds max_size.

    Args:
        source: Full Python source text.
        file_path: Path used in chunk metadata.
        max_size: Maximum chunk size in characters.

    Returns:
        List of Chunk objects.
    """
    chunks: List[Chunk] = []
    line_offsets = build_line_offsets(source)
    lines = source.splitlines(keepends=True)

    def line_to_char(lineno: int) -> int:
        """Convert 1-based line number to character offset."""
        idx = lineno - 1
        return line_offsets[idx] if idx < len(line_offsets) else len(source)

    def end_line_to_char(lineno: int) -> int:
        """Character offset right after the last char of ``lineno``."""
        idx = lineno - 1
        if idx < len(lines):
            return line_offsets[idx] + len(lines[idx])
        return len(source)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return sliding_window(source, file_path, 0, max_size)

    # Collect top-level node boundaries
    top_level: List[tuple[int, int]] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            if (
                hasattr(node, "lineno")
                and hasattr(node, "end_lineno")
                and node.end_lineno is not None
            ):
                start = line_to_char(node.lineno)
                end = end_line_to_char(node.end_lineno)
                top_level.append((start, end))

    if not top_level:
        return sliding_window(source, file_path, 0, max_size)

    # Sort by start position
    top_level.sort()

    # Collect leading module-level code (before first node)
    first_start = top_level[0][0]
    if first_start > 0:
        header = source[:first_start]
        if header.strip():
            chunks.extend(sliding_window(header, file_path, 0, max_size))

    prev_end = first_start
    for start, end in top_level:
        # Gap between nodes (e.g., module-level assignments)
        if start > prev_end:
            gap = source[prev_end:start]
            if gap.strip():
                chunks.extend(sliding_window(gap, file_path, prev_end, max_size))

        node_text = source[start:end]
        if len(node_text) <= max_size:
            if node_text.strip():
                chunks.append(
                    Chunk(
                        chunk_id=0,
                        file_path=file_path,
                        start_char=start,
                        end_char=end,
                        content=node_text,
                        index_type="",
                    )
                )
        else:
            chunks.extend(sliding_window(node_text, file_path, start, max_size))

        prev_end = end

    # Trailing code after last node
    if prev_end < len(source):
        tail = source[prev_end:]
        if tail.strip():
            chunks.extend(sliding_window(tail, file_path, prev_end, max_size))

    return chunks


def chunk_text(
    source: str, file_path: str, max_size: int
) -> List[Chunk]:
    """Chunk Markdown/text by headings then paragraphs.

    Splits first on heading lines (``# ``/``## ``/etc.), then on blank
    lines, then falls back to sliding window for oversized sections.

    Args:
        source: Full text content.
        file_path: Path used in chunk metadata.
        max_size: Maximum chunk size in characters.

    Returns:
        List of Chunk objects.
    """
    chunks: List[Chunk] = []

    heading_pattern = re.compile(r"(?=\n#{1,6} )", re.MULTILINE)
    sections = heading_pattern.split(source)

    current_pos = 0
    for section in sections:
        section_start = current_pos
        current_pos += len(section)
        if not section.strip():
            continue
        if len(section) <= max_size:
            chunks.append(
                Chunk(
                    chunk_id=0,
                    file_path=file_path,
                    start_char=section_start,
                    end_char=section_start + len(section),
                    content=section,
                    index_type="",
                )
            )
        else:
            section_offset = 0
            for part in re.split(r"(\n{2,})", section):
                if not part.strip():
                    section_offset += len(part)
                    continue
                actual_start = section_start + section_offset
                if len(part) <= max_size:
                    chunks.append(
                        Chunk(
                            chunk_id=0,
                            file_path=file_path,
                            start_char=actual_start,
                            end_char=actual_start + len(part),
                            content=part,
                            index_type="",
                        )
                    )
                else:
                    chunks.extend(sliding_window(part, file_path, actual_start, max_size))
                section_offset += len(part)

    if not chunks:
        return sliding_window(source, file_path, 0, max_size)

    return chunks


def chunk_file(
    file_path: str,
    repo_root: str,
    max_size: int,
    chunk_id_start: int = 0,
) -> List[Chunk]:
    """Read and chunk a single file.

    Args:
        file_path: Absolute path to the file.
        repo_root: Root path used to compute relative file paths.
        max_size: Maximum chunk size in characters.
        chunk_id_start: Starting chunk_id for this file's chunks.

    Returns:
        List of Chunk objects with assigned chunk_ids and index_type.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in SKIP_EXTENSIONS:
        return []

    if path.stat().st_size > MAX_FILE_SIZE:
        return []

    # Use path relative to cwd so it matches ground-truth format:
    # e.g. "data/raw/vllm-0.10.1/docs/features/lora.md"
    try:
        rel_path = str(path.relative_to(Path.cwd()))
    except ValueError:
        rel_path = str(path.relative_to(repo_root))

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    if not content.strip():
        return []

    if suffix in PYTHON_EXTENSIONS:
        raw_chunks = chunk_python(content, rel_path, max_size)
        index_type = "code"
    elif suffix in DOCS_EXTENSIONS:
        raw_chunks = chunk_text(content, rel_path, max_size)
        index_type = "docs"
    else:
        return []

    return [
        ch.model_copy(update={"chunk_id": chunk_id_start + i, "index_type": index_type})
        for i, ch in enumerate(raw_chunks)
    ]


def get_supported_extensions() -> dict:
    """Return extension sets for introspection.

    Returns:
        Dict with 'python', 'docs', and 'skip' extension sets.
    """
    return {
        "python": PYTHON_EXTENSIONS,
        "docs": DOCS_EXTENSIONS,
        "skip": SKIP_EXTENSIONS,
    }


def resolve_index_type(file_path: str) -> Optional[str]:
    """Determine which index type a file belongs to.

    Args:
        file_path: Path to the file.

    Returns:
        'code', 'docs', or None if not indexable.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix in PYTHON_EXTENSIONS:
        return "code"
    if suffix in DOCS_EXTENSIONS:
        return "docs"
    return None
