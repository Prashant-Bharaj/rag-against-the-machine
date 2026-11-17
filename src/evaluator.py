"""Recall@k evaluation for retrieval quality."""

from typing import Dict

from src.models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

DEFAULT_OVERLAP_THRESHOLD = 0.05


def sources_overlap(
    retrieved: MinimalSource,
    ground_truth: MinimalSource,
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> bool:
    """Check whether retrieved source overlaps with ground truth.

    A source is considered found if the character-level overlap between the
    retrieved range and the ground truth range is at least ``threshold``
    fraction of the ground truth range length.

    Args:
        retrieved: The retrieved source reference.
        ground_truth: The ground truth source reference.
        threshold: Minimum overlap fraction (default 5%).

    Returns:
        True if overlap >= threshold of ground truth length, False otherwise.
    """
    if retrieved.file_path != ground_truth.file_path:
        return False

    overlap_start = max(
        retrieved.first_character_index,
        ground_truth.first_character_index,
    )
    overlap_end = min(
        retrieved.last_character_index,
        ground_truth.last_character_index,
    )

    if overlap_end <= overlap_start:
        return False

    overlap_len = overlap_end - overlap_start
    gt_len = (
        ground_truth.last_character_index
        - ground_truth.first_character_index
    )

    if gt_len <= 0:
        return False

    return (overlap_len / gt_len) >= threshold


def recall_at_k(
    student_results: StudentSearchResults,
    dataset: RagDataset,
    k: int,
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> float:
    """Compute mean recall@k over all evaluable questions.

    For each question, recall is ``found_sources / total_gt_sources``
    (fractional, not binary). The final score is the mean over all questions
    that have ground truth sources.

    Args:
        student_results: Search results from the student system.
        dataset: Ground truth dataset with answered questions.
        k: Number of retrieved sources to consider.
        threshold: Minimum character overlap fraction to count as found.

    Returns:
        Mean recall@k across all questions with ground truth annotations.
    """
    gt_map: Dict[str, AnsweredQuestion] = {}
    for q in dataset.rag_questions:
        if isinstance(q, AnsweredQuestion):
            gt_map[q.question_id] = q

    total_recall = 0.0
    evaluated = 0

    for result in student_results.search_results:
        if result.question_id not in gt_map:
            continue

        gt_question = gt_map[result.question_id]
        gt_sources = gt_question.sources
        retrieved = result.retrieved_sources[:k]

        if not gt_sources:
            continue

        found = 0
        for gt_src in gt_sources:
            for ret_src in retrieved:
                if sources_overlap(ret_src, gt_src, threshold):
                    found += 1
                    break  # count each GT source at most once

        total_recall += found / len(gt_sources)
        evaluated += 1

    return total_recall / evaluated if evaluated > 0 else 0.0


def evaluate_all(
    student_results: StudentSearchResults,
    dataset: RagDataset,
    k: int,
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> Dict[str, float]:
    """Compute recall at multiple k values and summary statistics.

    Args:
        student_results: Search results from the student system.
        dataset: Ground truth dataset.
        k: Maximum k value (computes recall@1, @3, @5, @10, and @k).
        threshold: Minimum overlap fraction.

    Returns:
        Dict mapping 'recall@N' keys to float values.
    """
    ks = sorted({1, 3, 5, 10, k})
    results: Dict[str, float] = {}
    for ki in ks:
        results[f"recall@{ki}"] = recall_at_k(
            student_results, dataset, ki, threshold
        )
    return results


def print_evaluation(
    student_results: StudentSearchResults,
    dataset: RagDataset,
    k: int,
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> None:
    """Print evaluation results in the standard format.

    Args:
        student_results: Search results to evaluate.
        dataset: Ground truth dataset.
        k: Maximum k for evaluation.
        threshold: Minimum overlap fraction.
    """
    gt_count = sum(
        1 for q in dataset.rag_questions if isinstance(q, AnsweredQuestion)
    )
    total = len(student_results.search_results)

    print("Student data is valid: True")
    print(f"Total number of questions: {total}")
    print(f"Total number of questions with sources: {gt_count}")
    print(f"Total number of questions with student sources: {total}")
    print()
    print("Evaluation Results")
    print("=" * 40)
    print(f"Questions evaluated: {min(total, gt_count)}")

    metrics = evaluate_all(student_results, dataset, k, threshold)
    for key in sorted(metrics, key=lambda x: int(x.split("@")[1])):
        ki = int(key.split("@")[1])
        print(f"Recall@{ki}: {metrics[key]:.3f}")
