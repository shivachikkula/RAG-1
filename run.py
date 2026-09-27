"""Run any RAG type from one entry point.

    python run.py --list
    python run.py hybrid "Which ISO standard do the robots comply with?"
    python run.py --compare naive hybrid graph -- "Who designed Cirrus?"
"""

import argparse

from rag_core import get_llm, load_directory
from rag_core.documents import DEFAULT_DATA_DIR
from rag_types import RAG_TYPES, load_rag_class


def main():
    parser = argparse.ArgumentParser(description="Run a RAG pipeline over a folder of documents.")
    parser.add_argument("types", nargs="*", help=f"one or more of: {', '.join(RAG_TYPES)}")
    parser.add_argument("-q", "--question", default="How long does the Stratus S1 battery last?")
    parser.add_argument("--data", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--list", action="store_true", help="list available RAG types")
    parser.add_argument("--mock", action="store_true", help="force the offline mock LLM")
    args = parser.parse_args()

    if args.list or not args.types:
        for name in RAG_TYPES:
            doc = (load_rag_class(name).__doc__ or "").strip().splitlines()[0]
            print(f"{name:22} {doc}")
        return

    docs = load_directory(args.data)
    llm = get_llm(force_mock=args.mock or None)
    for name in args.types:
        rag = load_rag_class(name)(llm=llm)
        rag.index(docs)
        print(f"\n{'=' * 70}\n# {name}  (llm: {llm.model})\nQ: {args.question}\n")
        print(rag.query(args.question).pretty())


if __name__ == "__main__":
    main()
