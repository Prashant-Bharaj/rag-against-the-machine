"""Pydantic data models for the RAG pipeline."""

import uuid
from typing import List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class MinimalSource(BaseModel):
    """A reference to a character range within a file."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without a ground-truth answer."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with a ground-truth answer and source references."""

    model_config = ConfigDict(extra="ignore")

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions (answered or unanswered)."""

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """Search results for a single question."""

    question_id: str
    question_str: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results with a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Search results for all questions in a dataset."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results with generated answers for all questions."""

    search_results: List[MinimalAnswer]
    k: int


class Chunk(BaseModel):
    """Internal model representing a chunk of a source file."""

    chunk_id: int
    file_path: str
    start_char: int
    end_char: int
    content: str
    index_type: Literal["code", "docs", ""]
