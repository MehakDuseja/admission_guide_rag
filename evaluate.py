# """
# EVALUATION SCRIPT
# ----------------
# Runs a lightweight evaluation over a set of sample questions.
# It reports:
# - whether the retrieved chunks included the expected source/document
# - whether the answer contains expected keywords
# - a simple pass/fail summary
# """

# import json
# import signal
# from pathlib import Path
# from collections import Counter

# from src.embedder import load_embedder
# from src.vectorstore import load_vectorstore
# from src.retriever import retrieve
# from src.generator import build_chain, generate_answer
# import config


# class TimeoutException(Exception):
#     pass


# def _raise_timeout(signum, frame):
#     raise TimeoutException("evaluation timed out")


# def run_evaluation(dataset_path: Path | None = None) -> dict:
#     dataset_path = dataset_path or config.EVAL_DATASET_PATH
#     with open(dataset_path, "r", encoding="utf-8") as fh:
#         dataset = json.load(fh)

#     if config.EVAL_LIMIT:
#         dataset = dataset[: config.EVAL_LIMIT]

#     embedder = load_embedder()
#     vectorstore = load_vectorstore(embedder)
#     chain = build_chain()

#     signal.signal(signal.SIGALRM, _raise_timeout)
#     results = []
#     for item in dataset:
#         question = item["question"]
#         expected_keywords = item.get("expected_keywords", [])

#         retrieved = retrieve(question, vectorstore)
#         signal.alarm(config.EVAL_TIMEOUT_SECONDS)
#         try:
#             answer = generate_answer(question, retrieved, chain)
#             error = None
#         except TimeoutException as exc:
#             answer = f"[ERROR] {exc}"
#             error = str(exc)
#         except Exception as exc:
#             answer = f"[ERROR] {exc}"
#             error = str(exc)
#         finally:
#             signal.alarm(0)

#         sources = [doc.metadata.get("source", "unknown") for doc, _ in retrieved]
#         source_counter = Counter(sources)
#         answer_lower = answer.lower()
#         matched_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]

#         supported_by_pdf = bool(retrieved) and any(
#             doc.metadata.get("source", "").lower().endswith(".pdf") for doc, _ in retrieved
#         )

#         results.append({
#             "id": item.get("id", "unknown"),
#             "question": question,
#             "sources": sources,
#             "source_summary": dict(source_counter),
#             "answer": answer,
#             "expected_keywords": expected_keywords,
#             "matched_keywords": matched_keywords,
#             "supported_by_pdf": supported_by_pdf,
#             "error": error,
#             "passed": len(matched_keywords) >= 1 and supported_by_pdf and error is None,
#         })

#     return {
#         "dataset_path": str(dataset_path),
#         "total_questions": len(results),
#         "passed": sum(1 for r in results if r["passed"]),
#         "results": results,
#     }


# def _print_report(summary: dict) -> None:
#     print("\n╔══════════════════════════════════════╗")
#     print("║      RAG EVALUATION SUMMARY         ║")
#     print("╚══════════════════════════════════════╝")
#     print(f"Questions evaluated: {summary['total_questions']}")
#     print(f"Passed: {summary['passed']}")
#     print()

#     for item in summary["results"]:
#         status = "PASS" if item["passed"] else "FAIL"
#         status_color = "\033[92m" if item["passed"] else "\033[91m"
#         reset = "\033[0m"
#         print(f"{status_color}[{item['id']}] {status}{reset}")
#         print(f"Question: {item['question']}")
#         print("\nAnswer:")
#         print(item["answer"].strip())
#         print("\nRetrieved sources:")
#         for source_name, count in item["source_summary"].items():
#             print(f"  • {source_name} ({count})")
#         print("\nExpected keywords:")
#         print(f"  • {', '.join(item['expected_keywords']) or 'none'}")
#         print("Matched keywords:")
#         print(f"  • {', '.join(item['matched_keywords']) or 'none'}")
#         print(f"\nVerified from PDF: {'yes' if item['supported_by_pdf'] else 'no'}")
#         if item.get("error"):
#             print("Model error:")
#             print(f"  • {item['error']}")
#         print("─" * 40)


# if __name__ == "__main__":
#     summary = run_evaluation()
#     _print_report(summary)
"""
EVALUATION SCRIPT (IMPROVED)
----------------------------
Runs a lightweight evaluation over a set of sample questions.
It reports:
- whether the retrieved chunks included the expected source/document (PDF check, unchanged)
- keyword COVERAGE ratio (instead of just "at least one match")
- whether the answer leaks raw citations/source names (violates "no source clutter" rule)
- a pass/fail summary based on a configurable keyword coverage threshold
"""

import json
import re
import signal
from pathlib import Path
from collections import Counter

from src.embedder import load_embedder
from src.vectorstore import load_vectorstore
from src.retriever import retrieve
from src.generator import build_chain, generate_answer
import config


class TimeoutException(Exception):
    pass


def _raise_timeout(signum, frame):
    raise TimeoutException("evaluation timed out")


# Tune this: how much of the expected_keywords list must appear for a "pass"
KEYWORD_COVERAGE_THRESHOLD = 0.7  # 70% of expected keywords must be present

# Patterns that suggest the model leaked raw source/citation info into the answer
CITATION_LEAK_PATTERNS = [
    r"\.pdf\b",
    r"\bsource[:\s]",
    r"\[\d+\]",
    r"according to (the )?(document|batch|pdf)",
]


def _check_citation_leak(answer: str) -> bool:
    """Return True if the answer appears to leak raw source/citation references."""
    answer_lower = answer.lower()
    return any(re.search(pat, answer_lower) for pat in CITATION_LEAK_PATTERNS)


def _keyword_coverage(expected_keywords: list[str], answer_lower: str) -> tuple[list[str], float]:
    if not expected_keywords:
        return [], 1.0  # nothing expected => trivially full coverage
    matched = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    coverage = len(matched) / len(expected_keywords)
    return matched, coverage


def run_evaluation(dataset_path: Path | None = None) -> dict:
    dataset_path = dataset_path or config.EVAL_DATASET_PATH
    with open(dataset_path, "r", encoding="utf-8") as fh:
        dataset = json.load(fh)

    if config.EVAL_LIMIT:
        dataset = dataset[: config.EVAL_LIMIT]

    embedder = load_embedder()
    vectorstore = load_vectorstore(embedder)
    chain = build_chain()

    signal.signal(signal.SIGALRM, _raise_timeout)
    results = []
    for item in dataset:
        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])

        retrieved = retrieve(question, vectorstore)
        signal.alarm(config.EVAL_TIMEOUT_SECONDS)
        try:
            answer = generate_answer(question, retrieved, chain)
            error = None
        except TimeoutException as exc:
            answer = f"[ERROR] {exc}"
            error = str(exc)
        except Exception as exc:
            answer = f"[ERROR] {exc}"
            error = str(exc)
        finally:
            signal.alarm(0)

        sources = [doc.metadata.get("source", "unknown") for doc, _ in retrieved]
        source_counter = Counter(sources)
        answer_lower = answer.lower()

        # unchanged from original script
        supported_by_pdf = bool(retrieved) and any(
            doc.metadata.get("source", "").lower().endswith(".pdf") for doc, _ in retrieved
        )

        matched_keywords, keyword_coverage = _keyword_coverage(expected_keywords, answer_lower)
        citation_leak = _check_citation_leak(answer) if error is None else False

        passed = (
            error is None
            and supported_by_pdf
            and keyword_coverage >= KEYWORD_COVERAGE_THRESHOLD
            and not citation_leak
        )

        results.append({
            "id": item.get("id", "unknown"),
            "question": question,
            "sources": sources,
            "source_summary": dict(source_counter),
            "answer": answer,
            "expected_keywords": expected_keywords,
            "matched_keywords": matched_keywords,
            "keyword_coverage": round(keyword_coverage, 2),
            "supported_by_pdf": supported_by_pdf,
            "citation_leak": citation_leak,
            "error": error,
            "passed": passed,
        })

    return {
        "dataset_path": str(dataset_path),
        "total_questions": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "avg_keyword_coverage": round(
            sum(r["keyword_coverage"] for r in results) / len(results), 2
        ) if results else 0,
        "citation_leaks": sum(1 for r in results if r["citation_leak"]),
        "results": results,
    }


def _print_report(summary: dict) -> None:
    print("\n╔══════════════════════════════════════╗")
    print("║      RAG EVALUATION SUMMARY         ║")
    print("╚══════════════════════════════════════╝")
    print(f"Questions evaluated: {summary['total_questions']}")
    print(f"Passed: {summary['passed']}")
    print(f"Average keyword coverage: {summary['avg_keyword_coverage'] * 100:.0f}%")
    print(f"Citation leaks detected: {summary['citation_leaks']}")
    print()

    for item in summary["results"]:
        status = "PASS" if item["passed"] else "FAIL"
        status_color = "\033[92m" if item["passed"] else "\033[91m"
        reset = "\033[0m"
        print(f"{status_color}[{item['id']}] {status}{reset}")
        print(f"Question: {item['question']}")
        print("\nAnswer:")
        print(item["answer"].strip())
        print("\nRetrieved sources:")
        for source_name, count in item["source_summary"].items():
            print(f"  • {source_name} ({count})")
        print("\nExpected keywords:")
        print(f"  • {', '.join(item['expected_keywords']) or 'none'}")
        print("Matched keywords:")
        print(f"  • {', '.join(item['matched_keywords']) or 'none'}")
        print(f"Keyword coverage: {item['keyword_coverage'] * 100:.0f}%")
        print(f"\nVerified from PDF: {'yes' if item['supported_by_pdf'] else 'no'}")
        print(f"Citation leak detected: {'yes' if item['citation_leak'] else 'no'}")
        if item.get("error"):
            print("Model error:")
            print(f"  • {item['error']}")
        print("─" * 40)


if __name__ == "__main__":
    summary = run_evaluation()
    _print_report(summary)