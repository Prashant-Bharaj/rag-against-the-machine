"""Command-Line Interface for the RAG pipeline using Python Fire."""

import json
import sys
from pathlib import Path
from typing import NoReturn, Optional

from tqdm import tqdm

MAX_CHUNK_SIZE_LIMIT = 2000
MIN_CHUNK_SIZE = 50
INDEX_TYPES = ("code", "docs", "both")


def _fail(message: str) -> NoReturn:
    """Print an error message and exit with status 1.

    Args:
        message: Explanation shown to the user.
    """
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def _check_k(k: object) -> int:
    """Validate the ``--k`` option.

    Args:
        k: Value parsed by Fire.

    Returns:
        ``k`` as a non-negative integer.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        _fail(f"--k must be a non-negative integer, got {k!r}")
    return k


def _check_query(query: object) -> str:
    """Normalise a query parsed by Fire into a string.

    Fire turns some inputs into numbers or tuples (``search 42``,
    ``search a,b``), so they are converted back to text.

    Args:
        query: Value parsed by Fire.

    Returns:
        The query as a stripped string.
    """
    if isinstance(query, (list, tuple)):
        return " ".join(str(q) for q in query).strip()
    return str(query).strip()


def _check_index_type(index_type: object) -> str:
    """Validate the ``--index_type`` option.

    Args:
        index_type: Value parsed by Fire.

    Returns:
        One of ``code``, ``docs`` or ``both``.
    """
    if index_type not in INDEX_TYPES:
        _fail(f"--index_type must be one of {INDEX_TYPES}, got {index_type!r}")
    return str(index_type)


class StudentCLI:
    """RAG system for the vLLM knowledge base.

    Commands:
        index           - Index the repository
        search          - Search for a single query
        search_dataset  - Process multiple questions and save results
        answer          - Answer a single question with context
        answer_dataset  - Generate answers from search results
        evaluate        - Evaluate search results against ground truth
    """

    REPO_PATH = "data/raw"
    PROCESSED_PATH = "data/processed"

    def index(
        self,
        repo_path: str = REPO_PATH,
        processed_path: str = PROCESSED_PATH,
        max_chunk_size: int = 2000,
    ) -> None:
        """Index every file under ``repo_path`` and save BM25 indexes.

        Args:
            repo_path: Corpus root (default ``data/raw``).
            processed_path: Directory to save processed indexes.
            max_chunk_size: Maximum chunk size in characters.
        """
        from src.indexer import index as run_index

        if (
            isinstance(max_chunk_size, bool)
            or not isinstance(max_chunk_size, int)
            or not MIN_CHUNK_SIZE <= max_chunk_size <= MAX_CHUNK_SIZE_LIMIT
        ):
            _fail(
                "--max_chunk_size must be an integer between "
                f"{MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE_LIMIT}, "
                f"got {max_chunk_size!r}"
            )
        if not Path(repo_path).exists():
            print(
                f"Error: Repository not found at '{repo_path}'. "
                "Please extract the vLLM archive to data/raw/",
                file=sys.stderr,
            )
            sys.exit(1)

        run_index(repo_path, processed_path, max_chunk_size)

    def search(
        self,
        query: str,
        k: int = 10,
        processed_path: str = PROCESSED_PATH,
        index_type: str = "both",
    ) -> None:
        """Search for a single query and print results as JSON.

        Args:
            query: Search query string.
            k: Number of results to return.
            processed_path: Directory containing BM25 indexes.
            index_type: Index to search — 'code', 'docs', or 'both'.
        """
        from src.retriever import search as run_search
        from src.models import StudentSearchResults, MinimalSearchResults

        query = _check_query(query)
        k = _check_k(k)
        index_type = _check_index_type(index_type)
        try:
            sources = run_search(query, k, processed_path, index_type)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        output = StudentSearchResults(
            search_results=[
                MinimalSearchResults(
                    question_id="single",
                    question=query,
                    retrieved_sources=sources,
                )
            ],
            k=k,
        )
        print(output.model_dump_json(indent=2))

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        processed_path: str = PROCESSED_PATH,
        index_type: str = "both",
    ) -> None:
        """Search all questions in a dataset file and save results.

        Args:
            dataset_path: Path to the dataset JSON file.
            k: Number of results per question.
            save_directory: Directory to save search results.
            processed_path: Directory containing BM25 indexes.
            index_type: Index to search — 'code', 'docs', or 'both'.
        """
        from src.retriever import search as run_search
        from src.models import (
            RagDataset,
            StudentSearchResults,
            MinimalSearchResults,
        )

        k = _check_k(k)
        index_type = _check_index_type(index_type)
        dataset_path_obj = Path(str(dataset_path))
        if not dataset_path_obj.is_file():
            print(
                f"Error: Dataset file not found: {dataset_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            raw = json.loads(dataset_path_obj.read_text(encoding="utf-8"))
            dataset = RagDataset.model_validate(raw)
        except Exception as e:
            print(f"Error loading dataset: {e}", file=sys.stderr)
            sys.exit(1)

        # Auto-detect index type from dataset filename
        fname_lower = dataset_path_obj.name.lower()
        if index_type == "both":
            if "code" in fname_lower:
                index_type = "code"
            elif "docs" in fname_lower or "doc" in fname_lower:
                index_type = "docs"

        # Pre-load indexes once
        try:
            run_search("warmup", 1, processed_path, index_type)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        results = []
        for question in tqdm(dataset.rag_questions, desc="Searching"):
            sources = run_search(
                question.question, k, processed_path, index_type
            )
            results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=sources,
                )
            )

        output = StudentSearchResults(search_results=results, k=k)

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_file = save_dir / dataset_path_obj.name

        out_file.write_text(output.model_dump_json(indent=2), encoding="utf-8")
        print(f"Saved student_search_results to {out_file}")

    def answer(
        self,
        query: str,
        k: int = 10,
        processed_path: str = PROCESSED_PATH,
        index_type: str = "both",
    ) -> None:
        """Answer a single question using retrieved context.

        Args:
            query: The question to answer.
            k: Number of context chunks to retrieve.
            processed_path: Directory containing BM25 indexes.
            index_type: Index to search — 'code', 'docs', or 'both'.
        """
        from src.retriever import (
            search as run_search,
            get_chunks_for_sources,
        )
        from src.generator import generate
        from src.models import MinimalAnswer, StudentSearchResultsAndAnswer

        query = _check_query(query)
        k = _check_k(k)
        index_type = _check_index_type(index_type)
        try:
            sources = run_search(query, k, processed_path, index_type)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        chunks = get_chunks_for_sources(sources, processed_path)
        answer_text = generate(query, chunks)

        output = StudentSearchResultsAndAnswer(
            search_results=[
                MinimalAnswer(
                    question_id="single",
                    question=query,
                    retrieved_sources=sources,
                    answer=answer_text,
                )
            ],
            k=k,
        )
        print(output.model_dump_json(indent=2))

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        processed_path: str = PROCESSED_PATH,
    ) -> None:
        """Generate answers for all search results in a file.

        Args:
            student_search_results_path: Path to StudentSearchResults JSON.
            save_directory: Directory to save answered results.
            processed_path: Directory containing BM25 indexes.
        """
        from src.retriever import get_chunks_for_sources
        from src.generator import generate
        from src.models import (
            StudentSearchResults,
            StudentSearchResultsAndAnswer,
            MinimalAnswer,
        )

        src_path = Path(str(student_search_results_path))
        if not src_path.is_file():
            print(
                "Error: Search results file not found: "
                f"{student_search_results_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            raw = json.loads(src_path.read_text(encoding="utf-8"))
            search_results = StudentSearchResults.model_validate(raw)
        except Exception as e:
            print(f"Error loading search results: {e}", file=sys.stderr)
            sys.exit(1)

        total = len(search_results.search_results)
        print(f"Loaded {total} questions from {student_search_results_path}")

        # Pre-load indexes for chunk content lookup
        try:
            get_chunks_for_sources([], processed_path)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        answered: list[MinimalAnswer] = []
        for result in tqdm(
            search_results.search_results, desc="Generating answers"
        ):
            chunks = get_chunks_for_sources(
                result.retrieved_sources, processed_path
            )
            answer_text = generate(result.question, chunks)
            answered.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer_text,
                )
            )
        output = StudentSearchResultsAndAnswer(
            search_results=answered,
            k=search_results.k,
        )

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_file = save_dir / src_path.name
        out_file.write_text(output.model_dump_json(indent=2), encoding="utf-8")
        print(f"Saved student_search_results_and_answer to {out_file}")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int = 10,
        max_context_length: Optional[int] = None,
    ) -> None:
        """Evaluate retrieval recall@k against ground truth.

        Args:
            student_search_results_path: Path to StudentSearchResults or
                StudentSearchResultsAndAnswer JSON.
            dataset_path: Path to the ground truth AnsweredQuestions dataset.
            k: Number of retrieved sources to evaluate at.
            max_context_length: Unused parameter kept for CLI compatibility.
        """
        from src.models import RagDataset, StudentSearchResults
        from src.evaluator import print_evaluation

        k = _check_k(k)
        student_answer_path = str(student_search_results_path)
        dataset_path = str(dataset_path)
        for p, name in [
            (student_answer_path, "Student search results"),
            (dataset_path, "Ground truth dataset"),
        ]:
            if not Path(p).is_file():
                print(f"Error: {name} file not found: {p}", file=sys.stderr)
                sys.exit(1)

        try:
            raw_student = json.loads(
                Path(student_answer_path).read_text(encoding="utf-8")
            )
            # Accept StudentSearchResults and
            # StudentSearchResultsAndAnswer.
            student_results = StudentSearchResults.model_validate(raw_student)
        except Exception as e:
            print(f"Error loading student answers: {e}", file=sys.stderr)
            sys.exit(1)

        try:
            raw_gt = json.loads(
                Path(dataset_path).read_text(encoding="utf-8")
            )
            dataset = RagDataset.model_validate(raw_gt)
        except Exception as e:
            print(f"Error loading ground truth dataset: {e}", file=sys.stderr)
            sys.exit(1)

        print_evaluation(student_results, dataset, k)
