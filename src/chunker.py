"""
CHUNKER
-------
Splits large pages into smaller overlapping chunks using LangChain's
RecursiveCharacterTextSplitter.

Why do we chunk?
  An entire page may be too long to embed meaningfully as one unit.
  Smaller chunks also mean more precise retrieval — instead of returning
  a whole page, we return just the relevant section.

How RecursiveCharacterTextSplitter works:
  It tries to split on paragraph breaks (\n\n) first, then line breaks (\n),
  then spaces, then individual characters — whatever keeps chunks under
  chunk_size while respecting natural text boundaries.

chunk_overlap ensures consecutive chunks share some text so context
isn't lost at the boundary between two chunks.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.logger import get_logger
import config

logger = get_logger("CHUNKER")


def chunk_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    logger.info(
        f"Chunking {len(documents)} page(s) "
        f"(chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})..."
    )

    chunks = splitter.split_documents(documents)

    # Show per-source breakdown
    source_counts: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    for source, count in source_counts.items():
        logger.info(f"  {source} → {count} chunk(s)")

    logger.info(f"Total chunks created: {len(chunks)}")

    logger.debug("Chunk previews:")
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata
        logger.debug(
            f"  Chunk #{i + 1} | {meta.get('source')} | "
            f"page {meta.get('page', '?')} | batch {meta.get('batch_year')}\n"
            f"  {chunk.page_content[:300].strip()}\n"
            f"  {'─' * 60}"
        )

    return chunks
