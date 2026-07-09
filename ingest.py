"""
INGEST PIPELINE
---------------
Run this once to process your PDFs and build the vector store.

Usage:
  python ingest.py            # normal run
  python ingest.py --debug    # shows chunk contents for inspection

Steps:
  1. Load all PDFs from docs/
  2. Split them into chunks
  3. Load the embedding model
  4. Embed and store everything in ChromaDB

After this runs successfully, use query.py to ask questions.
"""

import argparse
from src.logger import get_logger, enable_debug
from src.loader import load_documents
from src.chunker import chunk_documents
from src.embedder import load_embedder
from src.vectorstore import create_vectorstore
import config

logger = get_logger("INGEST")


def main(debug: bool = False):
    if debug:
        enable_debug()

    logger.info("=" * 55)
    logger.info("INGESTION PIPELINE STARTED")
    logger.info("=" * 55)

    # Step 1: Load PDFs
    logger.info("STEP 1/4 — Loading PDFs")
    documents = load_documents(config.DOCS_DIR)

    # Step 2: Chunk documents
    logger.info("STEP 2/4 — Chunking documents")
    chunks = chunk_documents(documents)

    # Step 3: Load embedding model
    logger.info("STEP 3/4 — Loading embedding model")
    embedder = load_embedder()

    # Step 4: Store in ChromaDB
    logger.info("STEP 4/4 — Storing in ChromaDB")
    create_vectorstore(chunks, embedder)

    logger.info("=" * 55)
    logger.info("Ingestion complete. Run query.py to ask questions.")
    logger.info("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into the vector store")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show detailed chunk contents during ingestion",
    )
    args = parser.parse_args()
    main(debug=args.debug)
