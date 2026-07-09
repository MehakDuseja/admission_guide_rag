"""
VECTORSTORE
-----------
Stores and loads chunks in ChromaDB using LangChain's Chroma integration.

What is a vector store?
  After embedding each chunk into a vector, we need somewhere to store them
  so we can search through them efficiently. ChromaDB is a local vector
  database — it saves everything to disk so you don't have to re-embed
  your documents every time you run a query.

Two operations:
  create_vectorstore — called once during ingestion (ingest.py)
  load_vectorstore   — called every time you query (query.py)
"""

import shutil
from langchain_chroma import Chroma
from src.logger import get_logger
import config

logger = get_logger("VECTORSTORE")


def create_vectorstore(chunks: list, embedder) -> Chroma:
    logger.info(
        f"Creating ChromaDB collection '{config.COLLECTION_NAME}' "
        f"at {config.VECTORSTORE_DIR} ..."
    )

    if config.VECTORSTORE_DIR.exists():
        shutil.rmtree(config.VECTORSTORE_DIR)
        logger.info("Removed existing vector store directory to rebuild it from the current PDFs.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.VECTORSTORE_DIR),
    )

    logger.info(f"Stored {len(chunks)} chunk(s) in ChromaDB successfully.")
    return vectorstore


def load_vectorstore(embedder) -> Chroma:
    logger.info(f"Loading ChromaDB from {config.VECTORSTORE_DIR} ...")

    vectorstore = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embedder,
        persist_directory=str(config.VECTORSTORE_DIR),
    )

    logger.info("ChromaDB loaded and ready.")
    return vectorstore
