.PHONY: install run debug clean lint lint-strict

PYTHON = uv run python
MODULE = student

SOURCES = \
	src/student/__init__.py \
	src/student/__main__.py \
	src/student/chunker.py \
	src/student/cli.py \
	src/student/evaluator.py \
	src/student/generator.py \
	src/student/indexer.py \
	src/student/models.py \
	src/student/retriever.py

install:
	uv sync
	uv pip install -e .

run:
	$(PYTHON) -m $(MODULE)

debug:
	$(PYTHON) -m pdb -m $(MODULE)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; true

lint:
	uv run flake8 $(SOURCES)
	uv run mypy $(SOURCES) --warn-return-any --warn-unused-ignores --ignore-missing-imports \
	       --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 $(SOURCES)
	uv run mypy $(SOURCES) --strict
