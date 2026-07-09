"""
QUERY PIPELINE
--------------
Ask questions about the course outlines.

Usage:
  python query.py                              # interactive mode
  python query.py "What topics are in AI?"     # single question
  python query.py --debug                      # show retrieved chunk contents
  python query.py "Your question" --debug      # single question with debug

Steps:
  1. Load the embedding model
  2. Load the existing ChromaDB vector store
  3. Build the Gemini LangChain chain
  4. For each question: retrieve relevant chunks → generate answer
"""

import argparse
import sys
from src.logger import get_logger, enable_debug
from src.embedder import load_embedder
from src.vectorstore import load_vectorstore
from src.retriever import retrieve
from src.generator import build_chain, generate_answer
import config

logger = get_logger("QUERY")


def run_query(question: str, vectorstore, chain) -> None:
    logger.info("─" * 55)

    results = retrieve(question, vectorstore)
    answer  = generate_answer(question, results, chain)

    print(f"\n{'═' * 55}")
    print(f"Answer:\n{answer}")
    print(f"{'═' * 55}\n")


def main(debug: bool = False, question: str = None):
    if debug:
        enable_debug()

    vectorstore_path = config.VECTORSTORE_DIR
    if not vectorstore_path.exists():
        logger.error(
            "Vector store not found. Run 'python ingest.py' first."
        )
        sys.exit(1)

    logger.info("Loading models and vector store...")
    embedder    = load_embedder()
    vectorstore = load_vectorstore(embedder)
    chain       = build_chain()

    if question:
        run_query(question, vectorstore, chain)
    else:
        print("\nCourse Outline QA — type 'exit' to quit\n")
        while True:
            try:
                q = input("Your question: ").strip()
                if not q or q.lower() in ("exit", "quit"):
                    break
                run_query(q, vectorstore, chain)
            except KeyboardInterrupt:
                print("\nExiting.")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the course outline knowledge base")
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask (omit for interactive mode)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show retrieved chunk contents alongside the answer",
    )
    args = parser.parse_args()
    main(debug=args.debug, question=args.question)
