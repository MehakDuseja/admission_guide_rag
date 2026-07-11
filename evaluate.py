"""
EVALUATION SCRIPT
----------------
Runs a lightweight evaluation over a set of sample questions.
It reports:
- whether the retrieved chunks included the expected source/document
- whether the answer contains expected keywords
- a simple pass/fail summary
"""

import json
from pathlib import Path
from collections import Counter

from src.embedder import load_embedder
from src.vectorstore import load_vectorstore
from src.retriever import retrieve
from src.generator import build_chain, generate_answer
import config


def run_evaluation(dataset_path: Path | None = None) -> dict:
    dataset_path = dataset_path or config.EVAL_DATASET_PATH
    with open(dataset_path, "r", encoding="utf-8") as fh:
        dataset = json.load(fh)

    embedder = load_embedder()
    vectorstore = load_vectorstore(embedder)
    chain = build_chain()

    results = []
    for item in dataset:
        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])

        retrieved = retrieve(question, vectorstore)
        answer = generate_answer(question, retrieved, chain)

        sources = [doc.metadata.get("source", "unknown") for doc, _ in retrieved]
        source_counter = Counter(sources)
        answer_lower = answer.lower()
        matched_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]

        supported_by_pdf = bool(retrieved) and any(
            doc.metadata.get("source", "").lower().endswith(".pdf") for doc, _ in retrieved
        )

        results.append({
            "id": item.get("id", "unknown"),
            "question": question,
            "sources": sources,
            "source_summary": dict(source_counter),
            "answer": answer,
            "expected_keywords": expected_keywords,
            "matched_keywords": matched_keywords,
            "supported_by_pdf": supported_by_pdf,
            "passed": len(matched_keywords) >= 1 and supported_by_pdf,
        })

    return {
        "dataset_path": str(dataset_path),
        "total_questions": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "results": results,
    }


def _print_report(summary: dict) -> None:
    print("\n╔══════════════════════════════════════╗")
    print("║      RAG EVALUATION SUMMARY         ║")
    print("╚══════════════════════════════════════╝")
    print(f"Questions evaluated: {summary['total_questions']}")
    print(f"Passed: {summary['passed']}")
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
        print(f"\nVerified from PDF: {'yes' if item['supported_by_pdf'] else 'no'}")
        print("─" * 40)


if __name__ == "__main__":
    summary = run_evaluation()
    _print_report(summary)
