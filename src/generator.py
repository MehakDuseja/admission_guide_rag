"""
GENERATOR
---------
Builds the LangChain chain and generates an answer using Gemini.

LangChain Expression Language (LCEL) chain:
  prompt | llm | output_parser

  prompt        → fills in the template with {context} and {question}
  llm           → sends the filled prompt to Gemini and gets a response
  output_parser → extracts the plain string from Gemini's response object

Why temperature=0.2?
  Lower temperature = more focused, factual answers.
  We're doing document QA, not creative writing, so we want the model
  to stick closely to the retrieved context rather than improvise.

The prompt explicitly tells the model to answer ONLY from the context.
This is the key grounding instruction that prevents hallucination.
"""

import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.logger import get_logger
import config

logger = get_logger("GENERATOR")

PROMPT_TEMPLATE =("""
You are an academic advisor chatbot for a Computer Systems Engineering program. You answer questions about the curriculum, admissions, and FAQs using information provided in the CONTEXT below, which may include multiple sources — different curriculum batches (e.g., 2014, 2018, 2025) and PDF documents (e.g., FAQ PDFs).

Each chunk of CONTEXT is tagged with its source (batch/year or document name). Use these tags to determine which year or document a fact belongs to.

Rules you must follow:

1. COMPLETENESS: If the question spans multiple years, sections, or documents (e.g., "course outline," "all years," "full program," "FAQ," "all documents"), cover every relevant year or section present in the CONTEXT. Do not stop partway through if more relevant information exists.

2. ABBREVIATIONS & CODES: Students often use short forms or course codes (e.g., "CA" for Computer Architecture, "OS" for Operating Systems). Resolve these using context clues before answering. If unsure which course/topic an abbreviation refers to, state your interpretation explicitly.

3. CROSS-BATCH / CROSS-DOCUMENT CHANGES: If the same course or topic appears under different codes, credit hours, semesters, or documents across batches, explicitly point out the difference. Never silently pick one version when multiple exist.

4. NO GUESSING: Only state facts found in the CONTEXT. If the question concerns a batch/year/document not covered by the CONTEXT, say so plainly instead of inferring from another source.

5. UNSPECIFIED SCOPE: If the question doesn't specify a batch, year, or document, and multiple sources are present in the CONTEXT, either ask which one they mean, or briefly summarize all relevant ones — whichever is more helpful.

6. FULL, FOCUSED ANSWERS: Give complete, non-truncated answers when the CONTEXT supports it, but stay strictly on-topic. Do not add information beyond what answers the question.

7. EMPTY CONTEXT: If the CONTEXT is empty or irrelevant to the question, say so explicitly rather than fabricating an answer.

8. NO SOURCE CLUTTER: Do not display citations, source tags, or document names as inline references in the final answer (use them internally only to organize/attribute facts, e.g., "In the 2018 batch... / In the 2025 batch...").
9. DEFINATION : Define whenever user ask about the terms you can define it dont say you do not know use your knowledege to define it

CONTEXT:
{context}

QUESTION:
{question}

Answer clearly and completely using only the information present in the CONTEXT, following all rules above.`
""")


def build_chain():
    logger.info(f"Initializing Gemini: {config.GEMINI_MODEL}")

    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0.2,
    )

    prompt        = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    output_parser = StrOutputParser()

    # LCEL chain: prompt → llm → output_parser
    chain = prompt | llm | output_parser

    logger.info("Chain ready: Prompt → Gemini → Output Parser")
    return chain


def _resolve_course_name(query: str) -> str | None:
    """Figure out which course the student is actually asking about, using the
    same abbreviation map the retriever uses, plus a fallback to any full
    course name typed directly in the query."""
    from src.retriever import COURSE_ABBREVIATIONS

    lowered_query = query.lower()

    # 1. Direct abbreviation match (whole word), e.g. "OS", "CA", "DBMS"
    for abbr, full in COURSE_ABBREVIATIONS.items():
        if re.search(rf"\b{re.escape(abbr)}\b", query, re.IGNORECASE):
            return full.strip().lower()

    # 2. Full course name typed out directly, e.g. "operating systems"
    for full in COURSE_ABBREVIATIONS.values():
        if full.strip().lower() in lowered_query:
            return full.strip().lower()

    return None


def _extract_course_code(query: str, retrieved_results: list) -> str | None:
    """Try a simple rule-based extraction for course-code questions before using the LLM.

    IMPORTANT: this used to return as soon as it found ONE match anywhere in
    the retrieved chunks — so if the course's code differs across batches
    (e.g. CS-317 in 2014 vs CS-329 in 2018), it would silently report only
    whichever batch's chunk happened to come first and ignore the rest.
    Now it collects a match per batch and reports all of them if they differ.
    """
    lowered_query = query.lower()
    if "course code" not in lowered_query:
        return None

    target_course = _resolve_course_name(query)
    if not target_course:
        # We don't know which course they mean well enough to shortcut safely —
        # let it fall through to Gemini with full context instead of guessing.
        return None

    # batch_year -> code, so we don't just keep the first hit we see
    codes_by_batch: dict[str, str] = {}

    for doc, _ in retrieved_results:
        batch = doc.metadata.get("batch_year", "unknown")
        page_text = doc.page_content
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]

        for idx, line in enumerate(lines):
            if target_course in line.lower():
                code = None
                if idx > 0 and re.match(r"^[A-Za-z]{2,4}-\d{3,4}$", lines[idx - 1].strip()):
                    code = lines[idx - 1].strip()
                elif idx + 1 < len(lines) and re.match(r"^[A-Za-z]{2,4}-\d{3,4}$", lines[idx + 1].strip()):
                    code = lines[idx + 1].strip()

                if code and batch not in codes_by_batch:
                    codes_by_batch[batch] = code

    if not codes_by_batch:
        return None

    if len(codes_by_batch) == 1:
        # only one batch matched (or all batches agree) — simple answer
        ((_, only_code),) = codes_by_batch.items()
        return only_code

    # multiple batches, and their codes differ from one another (or we simply
    # have several distinct batch matches) — report all of them explicitly
    lines_out = [f"{batch}: {code}" for batch, code in sorted(codes_by_batch.items())]
    return "\n".join(lines_out)


def generate_answer(query: str, retrieved_results: list, chain) -> str:
    # Combine all retrieved chunks into one context block
    context_parts = []
    for doc, score in retrieved_results:
        meta = doc.metadata
        header = (
            f"[Source: {meta.get('source')} | "
            f"Batch: {meta.get('batch_year')} | "
            f"Page: {meta.get('page', '?')}]"
        )
        context_parts.append(f"{header}\n{doc.page_content.strip()}")

    context = "\n\n".join(context_parts)

    logger.debug(f"Full context being sent to Gemini:\n{'─'*60}\n{context}\n{'─'*60}")
    logger.info("Sending context + question to Gemini...")

    answer = chain.invoke({"context": context, "question": query})

    logger.info("Answer received.")
    return answer
