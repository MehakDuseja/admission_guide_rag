import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger that prints: [NAME] message
    All modules use this so the output style is consistent.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(name)-12s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


def enable_debug():
    """
    Call this when --debug flag is passed.
    Drops all loggers to DEBUG level so verbose output is shown.
    """
    for name in ["LOADER", "CHUNKER", "EMBEDDER", "VECTORSTORE", "RETRIEVER", "GENERATOR", "INGEST", "QUERY"]:
        logging.getLogger(name).setLevel(logging.DEBUG)
