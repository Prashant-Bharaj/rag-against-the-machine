"""Entry point for ``python -m src``."""

import sys

import fire

from src.cli import StudentCLI


def main() -> None:
    """Launch the RAG system CLI.

    Any unexpected exception is reported as a one-line error instead of
    a traceback, and the process exits with a non-zero status.
    """
    try:
        fire.Fire(StudentCLI)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
