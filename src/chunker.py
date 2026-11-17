"""Chunking strategies for Python code and text/Markdown files."""

import ast
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from src.models import Chunk

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
    max_size = max(1, max_size)
    # The overlap must stay below the window size or the window never moves.
    overlap = min(overlap, max_size // 4)

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


_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _first_line(node: ast.stmt) -> int:
    """Return the first line of a definition, including its decorators.

    Args:
        node: A function or class definition node.

    Returns:
        1-based line number where the definition starts.
    """
    decorators: List[ast.expr] = getattr(node, "decorator_list", [])
    lines: List[int] = [node.lineno] + [d.lineno for d in decorators]
    return min(lines)


def _split_class(
    node: ast.ClassDef,
    source: str,
    file_path: str,
    start: int,
    end: int,
    max_size: int,
    line_to_char: Callable[[int], int],
    end_line_to_char: Callable[[int], int],
) -> List[Chunk]:
    """Split an oversized class into chunks along its member boundaries.

    The class body is cut into segments (header, attributes, methods),
    which are packed greedily into chunks of at most ``max_size``
    characters. Segments that are still too large are split recursively
    (nested classes) or with a sliding window. Every chunk records the
    class name as context so retrieval can match it.

    Args:
        node: The class definition node.
        source: Full file text.
        file_path: Path used in chunk metadata.
        start: Character offset where the class starts.
        end: Character offset where the class ends.
        max_size: Maximum chunk size in characters.
        line_to_char: Maps a 1-based line to its first character offset.
        end_line_to_char: Maps a 1-based line to the offset after it.

    Returns:
        List of Chunk objects covering the class.
    """
    bounds: List[int] = [start]
    members: dict[int, ast.stmt] = {}
    for child in node.body:
        if isinstance(child, _DEF_NODES) and child.end_lineno is not None:
            c_start = max(line_to_char(_first_line(child)), bounds[-1])
            c_end = end_line_to_char(child.end_lineno)
            if c_start > bounds[-1]:
                bounds.append(c_start)
            members[c_start] = child
            bounds.append(c_end)
    if bounds[-1] < end:
        bounds.append(end)

    chunks: List[Chunk] = []
    pack_start = pack_end = start

    def flush() -> None:
        """Emit the segments packed so far as one chunk."""
        if pack_end > pack_start and source[pack_start:pack_end].strip():
            chunks.append(
                Chunk(
                    chunk_id=0,
                    file_path=file_path,
                    start_char=pack_start,
                    end_char=pack_end,
                    content=source[pack_start:pack_end],
                    index_type="",
                )
            )

    for seg_start, seg_end in zip(bounds, bounds[1:]):
        if seg_end - pack_start <= max_size:
            pack_end = seg_end
            continue
        flush()
        if seg_end - seg_start <= max_size:
            pack_start, pack_end = seg_start, seg_end
            continue
        member = members.get(seg_start)
        if isinstance(member, ast.ClassDef):
            chunks.extend(
                _split_class(
                    member, source, file_path, seg_start, seg_end,
                    max_size, line_to_char, end_line_to_char,
                )
            )
        else:
            chunks.extend(
                sliding_window(
                    source[seg_start:seg_end], file_path, seg_start, max_size
                )
            )
        pack_start = pack_end = seg_end
    flush()

    context = f"class {node.name}"
    return [
        ch if ch.context else ch.model_copy(update={"context": context})
        for ch in chunks
    ]


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
    top_level: List[Tuple[int, int, ast.stmt]] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, _DEF_NODES) and node.end_lineno is not None:
            start = line_to_char(_first_line(node))
            end = end_line_to_char(node.end_lineno)
            top_level.append((start, end, node))

    if not top_level:
        return sliding_window(source, file_path, 0, max_size)

    # Sort by start position
    top_level.sort(key=lambda t: t[0])

    # Collect leading module-level code (before first node)
    first_start = top_level[0][0]
    if first_start > 0:
        header = source[:first_start]
        if header.strip():
            chunks.extend(sliding_window(header, file_path, 0, max_size))

    prev_end = first_start
    for start, end, node in top_level:
        # Gap between nodes (e.g., module-level assignments)
        if start > prev_end:
            gap = source[prev_end:start]
            if gap.strip():
                chunks.extend(
                    sliding_window(gap, file_path, prev_end, max_size)
                )

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
        elif isinstance(node, ast.ClassDef):
            chunks.extend(
                _split_class(
                    node, source, file_path, start, end, max_size,
                    line_to_char, end_line_to_char,
                )
            )
        else:
            chunks.extend(
                sliding_window(node_text, file_path, start, max_size)
            )

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
                    chunks.extend(
                        sliding_window(part, file_path, actual_start, max_size)
                    )
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
        ch.model_copy(update={
            "chunk_id": chunk_id_start + i,
            "index_type": index_type,
        })
        for i, ch in enumerate(raw_chunks)
    ]


def get_supported_extensions() -> dict[str, set[str]]:
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
