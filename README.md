*This project has been created as part of the 42 curriculum by prasingh.*

# RAG Against the Machine

## Description

A **Retrieval-Augmented Generation (RAG)** system that answers questions about the [vLLM](https://github.com/vllm-project/vllm) codebase. The system:

1. **Ingests** the vLLM repository (Python source + Markdown docs) into a searchable BM25 index
2. **Retrieves** the most relevant code/documentation chunks for any query
3. **Generates** natural language answers using `Qwen/Qwen3-0.6B`
4. **Evaluates** retrieval quality using recall@k metrics

## Instructions

### Requirements

- Python ≥ 3.10
- [`uv`](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Install dependencies
make install
# or: uv sync
```

### Setup — vLLM Repository

Extract the vLLM archive to `data/raw/`:

```bash
mkdir -p data/raw
unzip vllm-0.10.1.zip -d data/raw/
# Result: data/raw/vllm-0.10.1/
```

### Index the Repository

```bash
uv run python -m student index --max_chunk_size 2000
```

Output: indexes saved to `data/processed/`.

### Search

```bash
# Single query
uv run python -m student search "How to configure OpenAI server?" --k 10

# Dataset of questions
uv run python -m student search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results
```

### Answer

```bash
# Single question
uv run python -m student answer "How to configure OpenAI server?" --k 10

# Full dataset
uv run python -m student answer_dataset \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer
```

### Evaluate

```bash
uv run python -m student evaluate \
  --student_answer_path data/output/search_results/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10
```

### Other Commands

```bash
make lint        # flake8 + mypy
make clean       # remove __pycache__, .mypy_cache
make debug       # run with pdb
```

---

## System Architecture

```
┌────────────────────────────────────────────────┐
│                  CLI (Fire)                    │
│  index | search | answer | evaluate | ...      │
└────────┬───────────────────────┬───────────────┘
         │                       │
    ┌────▼─────┐          ┌──────▼──────┐
    │ Indexer  │          │  Retriever  │
    │ (BM25)   │          │  (BM25 x2) │
    └────┬─────┘          └──────┬──────┘
         │                       │
    ┌────▼────────────────────────▼──────┐
    │         data/processed/            │
    │  bm25_index/code/  bm25_index/docs/│
    │  chunks/code_chunks.jsonl          │
    │  chunks/docs_chunks.jsonl          │
    └───────────────────┬────────────────┘
                        │
                  ┌─────▼──────┐
                  │ Generator  │
                  │ Qwen3-0.6B │
                  └─────┬──────┘
                        │
                  ┌─────▼──────┐
                  │ Evaluator  │
                  │ Recall@k   │
                  └────────────┘
```

**Pipeline flow:**
1. **Chunker** splits `.py` files (AST-based) and `.md` files (heading-based) into overlapping chunks
2. **Indexer** builds two BM25 indexes: one for code, one for docs
3. **Retriever** queries both indexes, normalizes scores, merges, returns top-k
4. **Generator** formats retrieved chunks as context and queries Qwen3-0.6B
5. **Evaluator** computes recall@k with 5% character-overlap threshold

---

## Chunking Strategy

Two dedicated strategies, dispatched by file extension:

### Python Code (`.py`, `.pyi`)

Uses Python's `ast` module to extract **function and class definitions** as atomic chunks:

- Parse AST, collect `FunctionDef`, `AsyncFunctionDef`, `ClassDef` nodes
- Convert `node.lineno` / `node.end_lineno` to character offsets via a precomputed line-offset array
- Class methods stay together with their class body (no splitting)
- Gap code (imports, module-level assignments) becomes its own chunk
- Files with syntax errors fall back to sliding window

### Markdown/Text (`.md`, `.rst`, `.txt`)

Splits on **heading boundaries** first, then on **paragraph breaks**:

1. Split on regex `(?=\n#{1,6} )` — each heading + body is a candidate section
2. If a section exceeds `max_chunk_size`, split further on `\n\n`
3. Oversized paragraphs fall back to sliding window

### Sliding Window Fallback

Used whenever a chunk would exceed `max_chunk_size` (default 2000):
- Window stride = `max_size - 200` (200 char overlap between windows)
- Ensures no content falls between window boundaries

### Configurable Limit

```bash
uv run python -m student index --max_chunk_size 1500
```

---

## Retrieval Method

**BM25** (via [`bm25s`](https://github.com/xhluca/bm25s)) — a fast, pure-Python BM25 implementation.

### Dual-Index Architecture

Two separate BM25 indexes to optimize for different token distributions:

| Index | Files | Stopwords |
|-------|-------|-----------|
| `code` | `.py`, `.pyi` | None (identifiers matter) |
| `docs` | `.md`, `.rst`, `.txt` | English stopwords |

### Search Algorithm

For a query with `k` results:
1. Query the **code** index for top-k results with scores
2. Query the **docs** index for top-k results with scores
3. Normalize each set of scores to [0, 1] by dividing by the maximum
4. Merge both lists, sort by normalized score descending
5. Return top-k `MinimalSource` objects

Querying each index for the full k (not k/2) ensures cross-domain questions (e.g., a question about a `.md` file) still get full coverage.

---

## Performance Analysis

### Target Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Recall@5 (docs) | ≥ 55% | BM25 on heading-chunked docs |
| Recall@5 (code) | ≥ 45% | BM25 on AST-chunked code |
| Indexing time | ≤ 5 min | One-time cost |
| Cold start | ≤ 60 s | Model loading + first query |
| Warm retrieval (1000 q) | ≤ 90 s | BM25 is fast; ~90ms/query |

### Latency Benchmark (Generation Backend)

Measured on this repository with:

```bash
STUDENT_GENERATION_BACKEND=<backend> \
uv run python -m student answer "What method needs to be overridden in BaseProcessingInfo?" --k 5 --index_type docs
```

| Backend | Measured latency | Notes |
|---------|------------------|-------|
| `auto` | ~1.35s | Uses low-latency path (OpenAI-compatible if configured, otherwise extractive fallback) |
| `hf_local` | ~86.49s | Loads local `Qwen/Qwen3-0.6B` via transformers pipeline; high cold-start cost |

For defense constraints, `auto` is the default to keep latency low. For full local LLM generation, set `STUDENT_GENERATION_BACKEND=hf_local`.

### BM25 Strengths for This Task

- Exact term matching is very effective for code queries (function names, class names, API parameters)
- No GPU required for retrieval
- Retrieval of 1000 queries completes in seconds once indexes are loaded

---

## Design Decisions

1. **BM25 over TF-IDF** — BM25 provides better recall on variable-length documents due to document length normalization. `bm25s` is significantly faster than scikit-learn's TF-IDF at query time.

2. **Dual index, not single** — Keeping code and docs separate avoids vocabulary pollution. Code identifiers (e.g., `vllm_engine`) are treated as meaningful tokens without English stopword removal.

3. **AST chunking** — Chunking at function/class boundaries preserves semantic units. A retriever finding a function body returns the entire function, which is the correct context for a code question.

4. **Module-level singletons** — Indexes and the generation model are loaded once per process. This is critical for meeting the 90s/1000q throughput requirement.

5. **`/no_think` in prompt** — Qwen3 models have a "thinking" mode that adds significant latency. Disabling it is necessary to meet the 2s/answer target.

6. **Pydantic v2** — Used throughout for data validation, JSON serialization, and type safety in the pipeline.

---

## Challenges Faced

1. **AST line-to-character offset conversion** — `ast` gives 1-based line numbers; converting to character offsets requires a precomputed line-offset array. Off-by-one errors here silently corrupt chunk boundaries.

2. **Encoding errors** — vLLM contains files with non-UTF-8 characters. Using `errors='replace'` prevents crashes but means chunk character indices are computed on the replaced string. This is acceptable since the evaluator uses the same character positions.

3. **bm25s k clamping** — If `k > corpus_size`, `bm25s.retrieve()` raises an error. All retrieval calls clamp k to `min(k, corpus_size)`.

4. **Transformers `do_sample=False` API** — In transformers ≥ 4.45, greedy decoding (`do_sample=False`) requires explicitly setting `temperature=None` and `top_p=None` to avoid warnings/errors.

5. **Pydantic Union deserialization** — `RagDataset.rag_questions` contains both `AnsweredQuestion` and `UnansweredQuestion`. Pydantic v2 tries each Union member in order; putting `AnsweredQuestion` first ensures the richer model is attempted first.

---

## Example Usage

```bash
# Full workflow
make install

# Extract vLLM repo
unzip vllm-0.10.1.zip -d data/raw/

# Index
uv run python -m student index --max_chunk_size 2000

# Search
uv run python -m student search "What is the sampling parameter for temperature?" --k 5

# Search dataset
uv run python -m student search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results

# Evaluate
uv run python -m student evaluate \
  --student_answer_path data/output/search_results/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10

# Answer a question
uv run python -m student answer "How to use beam search in vLLM?" --k 10

# Generate answers for dataset
uv run python -m student answer_dataset \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer
```

---

## Known Deviations from the Subject Spec

### `question_str` instead of `question` in `MinimalSearchResults`

The subject PDF specifies a field named `question` in the `MinimalSearchResults` model. This implementation uses `question_str` instead:

```python
class MinimalSearchResults(BaseModel):
    question_id: str
    question_str: str   # spec says 'question', moulinette requires 'question_str'
    retrieved_sources: List[MinimalSource]
```

The moulinette binary validates output JSON against `question_str`, not `question`. Using `question` causes the moulinette to reject the submission. This was verified empirically — the moulinette README uses `question_str` while the PDF uses `question`. The implementation follows what the moulinette actually requires.

### Answer generation latency with local model

The subject requires answers in ≤ 2 seconds per question and specifies `Qwen/Qwen3-0.6B` as the generation model. Running Qwen locally via the `transformers` pipeline takes significantly longer than 2 seconds per answer and has a cold-start cost exceeding the 60-second limit.

The default backend (`auto`) resolves this by preferring a low-latency OpenAI-compatible endpoint (e.g., a running vLLM server) and falling back to an extractive answer if none is configured. The extractive fallback meets the latency constraint but does not use Qwen for generation.

To use Qwen/Qwen3-0.6B locally (at the cost of latency):

```bash
STUDENT_GENERATION_BACKEND=hf_local uv run python -m student answer "your question"
```

To use a vLLM-served Qwen instance (meets both constraints):

```bash
STUDENT_OPENAI_BASE_URL=http://localhost:8000/v1 \
STUDENT_OPENAI_MODEL=Qwen/Qwen3-0.6B \
uv run python -m student answer_dataset ...
```

---

## Resources

### Documentation
- [vLLM Documentation](https://docs.vllm.ai/)
- [BM25 — Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25)
- [bm25s library](https://github.com/xhluca/bm25s)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/)
- [Python Fire](https://github.com/google/python-fire)

### Papers
- [Dense Passage Retrieval (DPR)](https://arxiv.org/abs/2004.04906)
- [RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

### AI Usage

AI (Claude) was used in this project for:
- **Architectural planning**: designing the dual-index strategy, discussing trade-offs between BM25 and dense retrieval
- **Boilerplate generation**: generating Pydantic model stubs and docstrings
- **Debugging assistance**: identifying the `do_sample=False` + `temperature=None` issue in transformers
- **Code review**: checking edge cases in the AST chunker (syntax error fallback, line offset conversion)

All AI-generated content was reviewed, tested, and understood before inclusion.
