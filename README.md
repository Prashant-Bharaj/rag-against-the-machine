*This project has been created as part of the 42 curriculum by prasingh.*

# RAG Against the Machine

## Description

A **Retrieval-Augmented Generation (RAG)** system that answers questions about the [vLLM](https://github.com/vllm-project/vllm) codebase. The system:

1. **Indexes** the vLLM repository (Python source + Markdown docs) into two BM25 indexes
2. **Retrieves** the most relevant code/documentation snippets for any question, as exact source locations (file path + character range)
3. **Generates** a grounded natural-language answer with `Qwen/Qwen3-0.6B`
4. **Evaluates** retrieval quality with recall@k

## Instructions

### Requirements

- Python ≥ 3.10
- [`uv`](https://docs.astral.sh/uv/) package manager

### Installation

```bash
make install        # runs: uv sync (installs runtime + dev tools)
```

### Expected layout

```
src/                         Python package, run with: uv run python -m src <command>
data/raw/vllm-0.10.1/        corpus (unzip vllm-0.10.1.zip -d data/raw/)
data/processed/              index written by the `index` command
data/datasets/{UnansweredQuestions,AnsweredQuestions}/
data/output/search_results/<DatasetScope>/
data/output/search_results_and_answer/<DatasetScope>/
```

### Commands

Every command is `uv run python -m src <command> [options]`. All input and output paths are CLI options.

| Command | Purpose |
|---------|---------|
| `index --max_chunk_size 2000` | Chunk `data/raw/` and build the indexes under `data/processed/` |
| `search "<query>" --k 10` | Top-k sources for one query |
| `search_dataset --dataset_path <file> --k 10 --save_directory <dir>` | Search every question of a dataset, write `StudentSearchResults` JSON |
| `answer "<query>" --k 10` | Retrieve and answer one question |
| `answer_dataset --student_search_results_path <file> --save_directory <dir>` | Answer every question of a search-results file, write `StudentSearchResultsAndAnswer` JSON |
| `evaluate --student_search_results_path <file> --dataset_path <file> --k 10` | Report recall@1/3/5/10 against a ground-truth dataset (for development; the official score comes from the moulinette) |

Optional on search commands: `--processed_path` (index location) and `--index_type code|docs|both`. `search_dataset` picks the code or docs index from the dataset file name when `--index_type` is `both`.

### Other Makefile rules

```bash
make run          # show the CLI help
make lint         # flake8 . + mypy . with the subject's flags
make lint-strict  # flake8 . + mypy . --strict
make debug        # run under pdb
make clean        # remove __pycache__, .mypy_cache, .pytest_cache
```

---

## System Architecture

```
            ┌──────────────────────── CLI (Python Fire, src/cli.py) ───────────────────────┐
            │  index │ search │ search_dataset │ answer │ answer_dataset │ evaluate        │
            └────┬────────────────────┬─────────────────────────┬─────────────────┬───────┘
                 │                    │                         │                 │
          ┌──────▼──────┐      ┌──────▼──────┐           ┌──────▼──────┐   ┌──────▼──────┐
          │  Chunker    │      │  Retriever  │──chunks──▶│  Generator  │   │  Evaluator  │
          │ AST / Markdown│    │  BM25 x 2   │           │ Qwen3-0.6B  │   │  recall@k   │
          └──────┬──────┘      └──────▲──────┘           └─────────────┘   └─────────────┘
                 │                    │
          ┌──────▼──────┐      ┌──────┴───────────────────────────────┐
          │  Indexer    │─────▶│ data/processed/                      │
          │ + Tokenizer │      │   bm25_index/{code,docs}/            │
          └─────────────┘      │   chunks/{code,docs}_chunks.jsonl    │
                               └──────────────────────────────────────┘
```

| Module | Role |
|--------|------|
| `src/chunker.py` | Two chunking strategies (Python AST, Markdown headings) + sliding-window fallback |
| `src/tokenizer.py` | Identifier-aware tokenizer shared by indexing and querying |
| `src/indexer.py` | Walks `data/raw/`, chunks every file, builds and saves both BM25 indexes |
| `src/retriever.py` | Loads indexes once per process, scores a query, returns `MinimalSource`s |
| `src/generator.py` | Builds a grounded prompt from the top chunks and generates with Qwen3-0.6B |
| `src/evaluator.py` | recall@k for the `evaluate` command |
| `src/models.py` | Pydantic models exchanged between stages (subject's models + internal `Chunk`) |

---

## Chunking Strategy

Every chunk is at most `--max_chunk_size` characters (default and maximum 2000, minimum 50), because the moulinette rejects any source longer than 2000 characters. Chunk offsets are exact character positions in the original file.

### Python code (`.py`, `.pyi`)

- Parse the file with `ast` and cut at **top-level function and class boundaries** (decorators included).
- A definition that fits in the limit becomes one chunk.
- An **oversized class** is split along its members: header/docstring/attributes and each method become segments, which are packed greedily into chunks of up to `max_chunk_size`. Nested classes are split recursively; a single oversized method falls back to the sliding window.
- Every chunk produced from a class remembers the class name (`context`), which is added to its index text, so "the `usage` property of the KV cache manager" can match a method chunk that never names its class.
- Module-level code between definitions (imports, constants) gets its own chunks.
- Files that fail to parse fall back to the sliding window.

### Markdown / text (`.md`, `.rst`, `.txt`)

1. Split on headings (`(?=\n#{1,6} )`), so each heading + body is a candidate section.
2. Sections over the limit are split on blank lines (paragraphs).
3. Paragraphs still over the limit use the sliding window.

### Sliding window fallback

Windows of `max_chunk_size` characters with a 200-character overlap (capped at a quarter of the window for small chunk sizes, so the window always advances).

---

## Retrieval Method

**BM25** via [`bm25s`](https://github.com/xhluca/bm25s), with two separate indexes (code and docs).

### Tokenization

Questions name things the way people talk ("the fused batched MoE layer", "`GPUModelRunner`", "the keye model"), while code writes `fused_batched_moe.py`, `class GPUModelRunner`, `_MAX_IMAGE_SIZE`. A plain word tokenizer cannot connect these, so `src/tokenizer.py` does, for both chunks and queries:

1. Extract identifiers and numbers.
2. Keep each compound identifier whole **and** split it on `_` and camelCase: `GPUModelRunner` → `gpumodelrunner`, `gpu`, `model`, `runner`.
3. Lowercase, drop English stopwords, stem with PyStemmer.

### Index text

Each chunk is indexed as **file path + enclosing class + content**. The path (without the shared `data/raw/vllm-0.10.1/` prefix) contributes tokens like `model_executor`, `models`, `keye`, which match questions that name a module or model rather than a function.

### Search

1. Tokenize the query; drop tokens not in the index vocabulary.
2. Query the code index, the docs index, or both (`--index_type`; `search_dataset` chooses from the dataset file name).
3. Fetch extra candidates (4 × k, at least 50) and halve the score of chunks under `tests/` or `benchmarks/`: they repeat the identifiers of the code they exercise and would otherwise outrank the implementation (e.g. `tests/conftest.py` sets `trust_remote_code=True`, the library default is `False`). They stay in the index, so questions about tests still work.
4. With both indexes, scores are normalised to [0, 1] per index, merged and sorted.
5. Return the top-k chunks as `MinimalSource` (file path + character range).

---

## Performance Analysis

Measured with the provided moulinette on the public datasets (`--k 10 --max_context_length 2000`), CPU only:

### Recall (moulinette)

| Dataset | Recall@1 | Recall@3 | **Recall@5** | Recall@10 | Target @5 |
|---------|---------:|---------:|-------------:|----------:|----------:|
| Docs (100 q) | 62.0% | 82.0% | **87.0%** | 92.0% | ≥ 80% |
| Code (99 q)  | 62.6% | 82.8% | **87.9%** | 91.9% | ≥ 50% |

### Effect of chunk size

| `--max_chunk_size` | Chunks (code / docs) | Docs R@5 | Code R@5 | Docs R@1 | Code R@1 |
|-------------------:|---------------------:|---------:|---------:|---------:|---------:|
| 2000 (default) | 19,007 / 2,948 | 86.0% | 86.9% | 62.0% | 61.6% |
| 500 | 55,888 / 5,871 | 84.0% | 89.9% | 49.0% | 55.6% |

Smaller chunks help code at k=5 (narrower spans around a specific constant or argument) but hurt the top-1 result, especially for docs, where a whole section is the natural unit. 2000 is kept as the default.

### Effect of the tokenizer (code dataset, chunk size 2000)

| Tokenization | Code R@5 | Docs R@5 |
|--------------|---------:|---------:|
| bm25s default tokenizer, content only | 48.5% | 85.0% |
| Identifier splitting + stemming + path and class context | 86.9% | 86.0% |
| + `tests/` and `benchmarks/` scores × 0.5 | **87.9%** | **87.0%** |

### Speed

| Step | Measured | Limit |
|------|---------:|------:|
| Indexing the whole corpus (`index`) | ~10 s | 5 min |
| `search_dataset`, 100 questions, including index load | ~1.6 s | 90 s for 200 questions |
| Qwen3-0.6B model load (once per process) | ~10–30 s | — |
| Answer generation, Qwen3-0.6B on a 6-core laptop CPU | ~15–35 s per question | — |

Generation time on CPU is dominated by reading the prompt (~100 tokens/s for the prefill), so the context is capped at 1,536 tokens from the top 5 chunks and answers at 128 new tokens. When a CUDA GPU is available the model runs on it automatically in float16; a full single `answer` command (index load + model load + generation) took ~17 s on an RTX 2060.

### Where it still misses

The remaining code misses are mostly questions whose wording shares few tokens with the code (e.g. "what dimension handling is applied to 2D vision embeddings") or where the right file is found but a different part of a very large file is ranked first.

---

## Design Decisions

1. **BM25 over TF-IDF**: length normalisation suits chunks of very different sizes, and `bm25s` scores hundreds of queries per second on CPU.
2. **Separate code and docs indexes**: the two corpora have very different vocabularies and document lengths; keeping them apart keeps IDF statistics meaningful for each.
3. **Identifier-aware tokenizer instead of embeddings**: most code questions quote or paraphrase identifiers. Splitting identifiers and indexing file paths raised code recall@5 from 48.5% to over 85% while keeping retrieval purely lexical, CPU-only and fast.
4. **AST chunking with member-level splitting of large classes**: returns whole functions/methods as context, and avoids blind 2000-character windows that cut a method away from its class.
5. **Module-level singletons**: indexes and the Qwen model are loaded once per process, so dataset commands pay the load cost once.
6. **Grounded prompt for Qwen3-0.6B**: numbered snippets with their file paths, a token budget measured with the model's own tokenizer, greedy decoding, thinking disabled through the chat template (`enable_thinking=False`), and numbered rules: copy values exactly, prefer library code over tests, and reply with one fixed sentence when the context lacks the answer. A first version phrased that last rule loosely; the 0.6B model then appended "the context does not contain the answer" to correct answers and invented values when the right file was not retrieved. The strict wording fixed both on the test questions.
7. **Never crash**: CLI arguments are validated (`k`, `max_chunk_size`, `index_type`, paths, JSON), every command reports errors on one line, and the generator falls back to an extractive answer if the model cannot be loaded.
8. **Pydantic v2** for every structure exchanged between stages.

---

## Challenges Faced

1. **Code recall below target**: the default BM25 tokenizer treats `trust_remote_code` or `GPUModelRunner` as single opaque words and ignores file names, so many code questions retrieved the wrong file. Analysing the misses led to the identifier-splitting tokenizer and path tokens.
2. **Large classes**: classes such as `GPUModelRunner` are tens of thousands of characters; slicing them into fixed windows separated methods from their class name. Member-level splitting with class context fixed most "right file, wrong span" misses.
3. **AST line-to-character conversion**: `ast` gives 1-based line numbers; converting them to character offsets needs a precomputed line-offset table, and off-by-one errors silently corrupt spans.
4. **Sliding-window overlap**: with a chunk size smaller than the overlap, the window never advanced. The overlap is now capped relative to the window size.
5. **Output schema change**: an earlier moulinette expected `question_str`; the current subject and moulinette use `question`, which the models now follow.
6. **Encoding errors**: some files contain non-UTF-8 bytes; they are read with `errors="replace"` so indexing never crashes.

---

## Example Usage

```bash
make install
mkdir -p data/raw && unzip vllm-0.10.1.zip -d data/raw/

# 1. Index
uv run python -m src index --max_chunk_size 2000

# 2. Search a dataset (scope the output directory by dataset)
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

# 3. Score with the moulinette (official) or with the built-in evaluator
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10

# 4. Generate answers
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions

# Single questions
uv run python -m src search "How to configure the OpenAI compatible server?" --k 5
uv run python -m src answer "What is the default value of trust_remote_code in the LLM class?" --k 5
```

---

## Resources

### Documentation
- [vLLM Documentation](https://docs.vllm.ai/)
- [Okapi BM25 (Wikipedia)](https://en.wikipedia.org/wiki/Okapi_BM25)
- [bm25s library](https://github.com/xhluca/bm25s)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Transformers documentation](https://huggingface.co/docs/transformers)
- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/)
- [Python Fire](https://github.com/google/python-fire)
- [Python `ast` module](https://docs.python.org/3/library/ast.html)

### Papers
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [The Probabilistic Relevance Framework: BM25 and Beyond (Robertson & Zaragoza, 2009)](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
- [Dense Passage Retrieval (Karpukhin et al., 2020)](https://arxiv.org/abs/2004.04906)

### AI Usage

AI (Claude) was used in this project for:
- **Architectural planning**: the dual-index design and BM25 vs dense retrieval trade-offs
- **Retrieval analysis**: classifying recall misses (wrong file vs wrong span), which motivated the identifier-aware tokenizer, path tokens and member-level class splitting
- **Spec compliance**: checking the project against the v2.0 subject (CLI entry point, output schema, Makefile lint rule, edge cases)
- **Boilerplate**: Pydantic model stubs and docstrings
- **Debugging**: the transformers greedy-decoding arguments, the sliding-window overlap bug, mypy errors

All AI-generated content was reviewed, tested, and understood before inclusion.
