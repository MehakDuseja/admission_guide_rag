# """
# RETRIEVER
# ---------
# Searches the vector store for the chunks most relevant to the user's query.

# How retrieval works:
#   1. The query is embedded into a vector using the same embedding model
#   2. ChromaDB compares that vector against all stored chunk vectors
#   3. It returns the top-k closest matches along with a similarity score

# similarity_search_with_score returns:
#   A list of (Document, score) tuples.
#   Lower score = more similar (it's a distance, not a percentage).
#   We log the score so you can see which chunks were confidently matched
#   and which were borderline — useful for debugging poor answers.

# Batch-aware filtering:
#   If the query mentions a specific batch year (e.g. "batch 2025"), we tell
#   ChromaDB to only search within that batch's documents using a metadata
#   filter. This prevents chunks from other batches from outranking the
#   correct ones in similarity search.

#   If the filtered search returns nothing (the info genuinely isn't in that
#   batch's documents), we fall back to searching all batches and log a
#   warning so you know the answer may not be batch-specific.
# """

# import re
# from src.logger import get_logger
# import config

# logger = get_logger("RETRIEVER")

# KNOWN_BATCHES = ["2014", "2018", "2025"]
# COURSE_ABBREVIATIONS = {
#     "CA": "Computer Architecture",
#     "OS": "Operating Systems",
#     "DS": "Data Structures and Algorithms",
#     "DLD": "Digital Logic Design",
#     "DBMS": "Database Management Systems",
#     "OOP": "Object Oriented Programming",
#     "AI": "Artificial Intelligence",
#     "ML": "Machine Learning",
#     "CCN": "Computer Communication Networks",
#     "SE": "Software Engineering",
#     "CV": "Computer Vision",
#     "DSP": "Digital Signal Processing",
#     "COD":" computer organization and design",
#     "CN":" computer networks",
#     "DB":" database systems",
#     "DSA":" data structures and algorithms",
#     "DLD":" digital logic design",
#     "OOP":" object oriented programming",
#     "OS":" operating systems",
    
    
#     # add more as you find gaps
# }

# def expand_query(query: str) -> str:
#     expanded = query
#     for abbr, full in COURSE_ABBREVIATIONS.items():
#         # word-boundary match so "CA" doesn't hit inside other words
#         expanded = re.sub(rf"\b{abbr}\b", f"{abbr} ({full})", expanded, flags=re.IGNORECASE)
#     return expanded

# # then: rag_retriever.retrieve(expand_query(query), top_k=8)

# # ABBREVIATION_MAP = {
# #     "ds": "discrete structures",
# #     "oop": "object oriented programming",
# #     "os": "operating systems",
# #     "db": "database",
# # }


# def _normalize_query(query: str) -> str:
#     """Expand common course shorthand so the retriever can match the PDF text better."""
#     normalized = query.strip()
#     for short, full in COURSE_ABBREVIATIONS.items():
#         normalized = re.sub(rf"\b{short}\b", full, normalized, flags=re.IGNORECASE)
#     return normalized


# def _detect_batch_year(query: str) -> str | None:
#     """Return the first batch year found in the query, or None."""
#     for year in KNOWN_BATCHES:
#         if year in query:
#             return year
#     return None


# def retrieve(query: str, vectorstore) -> list:
#     normalized_query = _normalize_query(query)
#     logger.info(f'Searching for: "{query}"')
#     if normalized_query != query:
#         logger.info(f"Normalized query to: \"{normalized_query}\"")

#     batch_year = _detect_batch_year(normalized_query)

#     if batch_year:
#         logger.info(f"Batch year '{batch_year}' detected — filtering search to that batch only.")
#         results = vectorstore.similarity_search_with_score(
#             normalized_query,
#             k=config.TOP_K,
#             filter={"batch_year": batch_year},
#         )

#         if not results:
#             logger.warning(
#                 f"No chunks found for batch {batch_year}. "
#                 f"Falling back to search across all batches."
#             )
#             results = vectorstore.similarity_search_with_score(normalized_query, k=config.TOP_K)
#     else:
#         logger.info("No specific batch year detected — searching across all batches with a wider context window.")
#         results = vectorstore.similarity_search_with_score(normalized_query, k=config.TOP_K)

#         # If the first pass only pulls from one source, broaden the search so
#         # general questions can still use information from the other PDFs.
#         sources = {doc.metadata.get("source") for doc, _ in results}
#         if len(sources) < 2:
#             logger.info("Only one source matched in the first pass; broadening the search.")
#             results = vectorstore.similarity_search_with_score(normalized_query, k=config.TOP_K * 2)

#     logger.info(f"Retrieved {len(results)} chunk(s):")
#     for i, (doc, score) in enumerate(results):
#         meta = doc.metadata
#         logger.info(
#             f"  #{i + 1} | score: {score:.4f} | {meta.get('source')} | "
#             f"page {meta.get('page', '?')} | batch {meta.get('batch_year')}"
#         )
#         logger.debug(
#             f"       Content:\n"
#             f"       {doc.page_content[:400].strip()}\n"
#             f"       {'─' * 60}"
#         )

#     return results

#1
"""
RETRIEVER
---------
Searches the vector store for the chunks most relevant to the user's query.

How retrieval works:
  1. The query is embedded into a vector using the same embedding model
  2. ChromaDB compares that vector against all stored chunk vectors
  3. It returns the top-k closest matches along with a similarity score

similarity_search_with_score returns:
  A list of (Document, score) tuples.
  Lower score = more similar (it's a distance, not a percentage).
  We log the score so you can see which chunks were confidently matched
  and which were borderline — useful for debugging poor answers.

Batch-aware filtering:
  If the query mentions a specific batch year (e.g. "batch 2025"), we tell
  ChromaDB to only search within that batch's documents using a metadata
  filter. This prevents chunks from other batches from outranking the
  correct ones in similarity search.

  If the filtered search returns nothing (the info genuinely isn't in that
  batch's documents), we fall back to searching all batches and log a
  warning so you know the answer may not be batch-specific.
"""

import re
from src.logger import get_logger
import config

logger = get_logger("RETRIEVER")

KNOWN_BATCHES = ["2014", "2018", "2025"]
COURSE_ABBREVIATIONS = {
    "CA": "Computer Architecture",
    "OS": "Operating Systems",
    "DS": "Data Structures and Algorithms",
    "DSA": "Data Structures and Algorithms",
    "DLD": "Digital Logic Design",
    "DBMS": "Database Management Systems",
    "DB": "Database Systems",
    "OOP": "Object Oriented Programming",
    "AI": "Artificial Intelligence",
    "ML": "Machine Learning",
    "CCN": "Computer Communication Networks",
    "CN": "Computer Networks",
    "SE": "Software Engineering",
    "CV": "Computer Vision",
    "DSP": "Digital Signal Processing",
    "COD": "Computer Organization and Design",
}

def expand_query(query: str) -> str:
    expanded = query
    for abbr, full in COURSE_ABBREVIATIONS.items():
        # word-boundary match so "CA" doesn't hit inside other words
        expanded = re.sub(rf"\b{abbr}\b", f"{abbr} ({full})", expanded, flags=re.IGNORECASE)
    return expanded

# then: rag_retriever.retrieve(expand_query(query), top_k=8)

# ABBREVIATION_MAP = {
#     "ds": "discrete structures",
#     "oop": "object oriented programming",
#     "os": "operating systems",
#     "db": "database",
# }


def _normalize_query(query: str) -> str:
    """Expand common course shorthand so the retriever can match the PDF text better."""
    normalized = query.strip()
    for short, full in COURSE_ABBREVIATIONS.items():
        normalized = re.sub(rf"\b{short}\b", full, normalized, flags=re.IGNORECASE)
    return normalized


def _detect_batch_year(query: str) -> str | None:
    """Return the first batch year found in the query, or None."""
    for year in KNOWN_BATCHES:
        if year in query:
            return year
    return None


def retrieve(query: str, vectorstore) -> list:
    normalized_query = _normalize_query(query)
    logger.info(f'Searching for: "{query}"')
    if normalized_query != query:
        logger.info(f"Normalized query to: \"{normalized_query}\"")

    batch_year = _detect_batch_year(normalized_query)

    if batch_year:
        logger.info(f"Batch year '{batch_year}' detected — filtering search to that batch only.")
        results = vectorstore.similarity_search_with_score(
            normalized_query,
            k=config.TOP_K,
            filter={"batch_year": batch_year},
        )

        if not results:
            logger.warning(
                f"No chunks found for batch {batch_year}. "
                f"Falling back to search across all batches."
            )
            results = vectorstore.similarity_search_with_score(normalized_query, k=config.TOP_K)
    else:
        logger.info("No specific batch year detected — performing round-robin batch retrieval.")
        results = []
        
        # Explicitly query each batch so no single batch dominates the results
        for batch in KNOWN_BATCHES:
            batch_results = vectorstore.similarity_search_with_score(
                normalized_query,
                k=max(3, config.TOP_K // len(KNOWN_BATCHES)),
                filter={"batch_year": batch}
            )
            results.extend(batch_results)
            logger.info(f"  Pulled {len(batch_results)} chunks for batch {batch}")

        # Fallback if the metadata filters returned completely empty results
        if not results:
            logger.info("Round-robin returned empty; pulling general top_k.")
            results = vectorstore.similarity_search_with_score(normalized_query, k=config.TOP_K * 2)

    logger.info(f"Retrieved {len(results)} chunk(s):")
    for i, (doc, score) in enumerate(results):
        meta = doc.metadata
        logger.info(
            f"  #{i + 1} | score: {score:.4f} | {meta.get('source')} | "
            f"page {meta.get('page', '?')} | batch {meta.get('batch_year')}"
        )
        logger.debug(
            f"       Content:\n"
            f"       {doc.page_content[:400].strip()}\n"
            f"       {'─' * 60}"
        )

    return results