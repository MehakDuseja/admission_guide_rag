"""
EMBEDDER
--------
Loads an open-source embedding model using LangChain's HuggingFaceEmbeddings.

What is an embedding?
  An embedding converts text into a list of numbers (a vector) that captures
  its meaning. Similar texts produce vectors that are close together in space.
  This is what makes semantic search possible — we don't match keywords,
  we match meaning.

Why all-MiniLM-L6-v2?
  - Very small (~22 MB), runs on CPU with no GPU required
  - Trained specifically for semantic similarity tasks
  - One of the most used models in RAG tutorials — easy to find help for
  - normalize_embeddings=True ensures vectors are unit length, which makes
    cosine similarity scores consistent and comparable
"""

from langchain_huggingface import HuggingFaceEmbeddings
from src.logger import get_logger
import config

logger = get_logger("EMBEDDER")


def load_embedder() -> HuggingFaceEmbeddings:
    logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    logger.info("  (first run will download the model — this may take a moment)")

    embedder = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Embedding model ready.")
    return embedder
