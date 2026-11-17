.PHONY: install run debug clean lint lint-strict

PYTHON = uv run python
MODULE = src

install:
	uv sync

run:
	$(PYTHON) -m $(MODULE)

debug:
	$(PYTHON) -m pdb -m $(MODULE)

clean:
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	rm -rf .mypy_cache .ruff_cache .pytest_cache; \
	find . -path ./.venv -prune -o -name "*.pyc" -delete 2>/dev/null; true

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports \
	       --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
