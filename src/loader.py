"""
LOADER
------
Reads every PDF from the docs/ folder using LangChain's PyMuPDFLoader.

What PyMuPDFLoader does:
  - Opens a PDF file page by page
  - Returns a list of LangChain Document objects
  - Each Document has:
      .page_content  → the raw text of that page
      .metadata      → dict with source filename, page number, etc.

We also attach the batch year (e.g. "2018") to each page's metadata
so we can later filter or reference which curriculum the chunk came from.
"""

from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from src.logger import get_logger

logger = get_logger("LOADER")


def load_documents(docs_dir: Path) -> list:
    docs_dir = Path(docs_dir)
    pdf_files = sorted(docs_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {docs_dir}")

    logger.info(f"Found {len(pdf_files)} PDF(s): {[f.name for f in pdf_files]}")

    all_documents = []

    for pdf_file in pdf_files:
        loader = PyMuPDFLoader(str(pdf_file))
        pages  = loader.load()

        # Attach batch year derived from the filename (e.g. "2018.pdf" → "2018")
        batch_year = pdf_file.stem
        for page in pages:
            page.metadata["batch_year"] = batch_year
            page.metadata["source"]     = pdf_file.name

        logger.info(f"  {pdf_file.name} → {len(pages)} page(s) loaded")

        logger.debug("  Page previews:")
        for page in pages:
            logger.debug(
                f"    [page {page.metadata.get('page', '?')}] "
                f"{page.page_content[:150].strip().replace(chr(10), ' ')}"
            )

        all_documents.extend(pages)

    logger.info(f"Total pages loaded: {len(all_documents)}")
    return all_documents
