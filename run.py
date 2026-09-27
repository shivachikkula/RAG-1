"""Run any RAG type from one entry point.

    python run.py --list
    python run.py hybrid -q "Which ISO standard do the robots comply with?"
    python run.py naive hybrid graph -q "Who designed Cirrus?"
    python run.py all --mock --report report.md
"""

import argparse
import sys
import time

from rag_core import ClaudeLLM, get_llm, load_directory
from rag_core.documents import DEFAULT_DATA_DIR
from rag_types import RAG_TYPES, load_rag_class


def write_report(path, question, llm, rows):
    lines = [f"## RAG run ({llm.model})", "", f"**Question:** {question}", "",
             "| Type | Status | Time (s) | Sources |", "|---|---|---|---|"]
    for name, status, seconds, result in rows:
        sources = ", ".join(d.id for d in result.sources) if result else ""
        lines.append(f"| {name} | {status} | {seconds:.2f} | {sources} |")
    for name, status, _, result in rows:
        if result:
            answer = result.answer.replace("\n", "\n> ")
            lines += ["", f"### {name}", "", f"> {answer}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run a RAG pipeline over a folder of documents.")
    parser.add_argument("types", nargs="*", help=f"'all' or one or more of: {', '.join(RAG_TYPES)}")
    parser.add_argument("-q", "--question", default="How long does the Stratus S1 battery last?")
    parser.add_argument("--data", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--list", action="store_true", help="list available RAG types")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true", help="force the offline mock LLM")
    mode.add_argument("--live", action="store_true", help="require Claude; fail if no credentials are set")
    parser.add_argument("--report", help="write a Markdown summary to this path")
    args = parser.parse_args()

    if args.list or not args.types:
        for name in RAG_TYPES:
            doc = (load_rag_class(name).__doc__ or "").strip().splitlines()[0]
            print(f"{name:22} {doc}")
        return

    types = list(RAG_TYPES) if args.types == ["all"] else args.types
    unknown = [t for t in types if t not in RAG_TYPES]
    if unknown:
        parser.error(f"unknown RAG type(s): {', '.join(unknown)}")

    llm = get_llm(force_mock=args.mock or None)
    if args.live and not isinstance(llm, ClaudeLLM):
        parser.error("--live needs ANTHROPIC_API_KEY (or another Anthropic credential) and RAG_MOCK unset")

    docs = load_directory(args.data)
    rows, failed = [], False
    for name in types:
        start = time.perf_counter()
        print(f"\n{'=' * 70}\n# {name}  (llm: {llm.model})\nQ: {args.question}\n")
        try:
            rag = load_rag_class(name)(llm=llm)
            rag.index(docs)
            result = rag.query(args.question)
            print(result.pretty())
            rows.append((name, "ok", time.perf_counter() - start, result))
        except Exception as exc:  # keep running the other types, report failure at the end
            failed = True
            print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            rows.append((name, f"failed: {type(exc).__name__}", time.perf_counter() - start, None))

    if args.report:
        write_report(args.report, args.question, llm, rows)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
