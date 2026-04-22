"""Entry point for ``python -m student``."""

import fire

from student.cli import StudentCLI


def main() -> None:
    """Launch the RAG system CLI."""
    fire.Fire(StudentCLI)


if __name__ == "__main__":
    main()
