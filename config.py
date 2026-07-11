import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

DOCS_DIR        = BASE_DIR / "docs"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Chunking — how big each text chunk is and how much they overlap
CHUNK_SIZE    = 700
CHUNK_OVERLAP = 100

# Embedding model — runs locally, no API key needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB collection name
COLLECTION_NAME = "course_outlines"

# How many chunks to retrieve per query
# A slightly larger value helps general questions pull context from multiple PDFs.
TOP_K = 8

# Evaluation dataset for RAG benchmarking
EVAL_DATASET_PATH = BASE_DIR / "eval_questions.json"

# Gemini LLM
GEMINI_MODEL   = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
