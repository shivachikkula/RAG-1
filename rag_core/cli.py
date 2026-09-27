"""Tiny CLI shared by each pipeline's ``python pipeline.py "question"`` entry point."""

from __future__ import annotations

import argparse

from .documents import DEFAULT_DATA_DIR, load_directory


def run_cli(rag_cls, default_question: str) -> None:
    parser = argparse.ArgumentParser(description=f"{rag_cls.name}: {(rag_cls.__doc__ or '').strip().splitlines()[0]}")
    parser.add_argument("question", nargs="?", default=default_question)
    parser.add_argument("--data", default=str(DEFAULT_DATA_DIR), help="directory of .md/.txt files to index")
    args = parser.parse_args()

    rag = rag_cls()
    print(f"# {rag_cls.name}  (llm: {rag.llm.model})")
    rag.index(load_directory(args.data))
    print(f"Q: {args.question}\n")
    print(rag.query(args.question).pretty())
