"""Code-aware tokenization shared by indexing and retrieval."""

import re
from functools import lru_cache
from typing import List

import Stemmer
from bm25s.stopwords import STOPWORDS_EN

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_STOPWORDS = frozenset(STOPWORDS_EN)
_stemmer = Stemmer.Stemmer("english")


@lru_cache(maxsize=200_000)
def _expand(word: str) -> tuple[str, ...]:
    """Expand one identifier into its normalised sub-tokens.

    ``GPUModelRunner`` yields ``gpumodelrunner``, ``gpu``, ``model`` and
    ``runner``; ``trust_remote_code`` yields the full identifier plus
    ``trust``, ``remote`` and ``code``. Every token is lowercased and
    stemmed, and English stopwords are dropped.

    Args:
        word: A single identifier or number.

    Returns:
        Tuple of tokens derived from the identifier.
    """
    parts: List[str] = []
    for piece in word.split("_"):
        if piece:
            parts.extend(_CAMEL.findall(piece))
    lowered = [p.lower() for p in parts]
    full = word.lower().strip("_")
    tokens: List[str] = []
    if len(lowered) > 1 and full:
        tokens.append(full)
    tokens.extend(p for p in lowered if p not in _STOPWORDS and len(p) > 1)
    return tuple(_stemmer.stemWords(tokens))


def tokenize(text: str) -> List[str]:
    """Split text into stemmed, identifier-aware tokens.

    Args:
        text: Source code, documentation or a natural-language query.

    Returns:
        List of tokens suitable for a BM25 index.
    """
    tokens: List[str] = []
    for word in _IDENTIFIER.findall(text):
        tokens.extend(_expand(word))
    return tokens


def path_tokens(file_path: str) -> List[str]:
    """Tokenize the meaningful part of a corpus file path.

    The ``data/raw/vllm-x.y.z/`` prefix is shared by every file, so only
    the path inside the repository is kept.

    Args:
        file_path: Corpus-relative path of a file.

    Returns:
        Tokens describing the file location and name.
    """
    inner = re.sub(r"^.*?vllm-[\d.]+/", "", file_path)
    inner = re.sub(r"\.(py|pyi|md|rst|txt)$", "", inner)
    return tokenize(inner.replace("/", " "))
