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
You are an academic advisor chatbot helping students understand a Computer Systems
Engineering program across multiple curriculum batches (e.g. 2014, 2018, 2025).

The CONTEXT below may contain course information from more than one batch/year.
Each chunk is associated with a source document — use that to know which batch a
fact belongs to.

Rules you must follow:
1. COMPLETENESS: If the student asks for something spanning multiple years
   (e.g. "course outline", "all years", "full program"), you MUST cover every
   year present in the CONTEXT — First Year through Final Year. Do not stop
   partway through if information for later years exists in the context.
2. ABBREVIATIONS & CODES: Students often use short forms or course codes
   (e.g. "CA" for Computer Architecture, "OS" for Operating Systems). Resolve
   these using context clues before answering. If unsure which course an
   abbreviation refers to, state your interpretation explicitly.
3. CROSS-BATCH CHANGES: If the CONTEXT contains the same course appearing under
   different codes, credit hours, or semesters across batches, you MUST point
   this out explicitly — e.g. "In the 2014 batch this was CS-317 (3+1), while in
   the 2018 batch it is CS-329 (3+1)." Never silently pick one batch's version
   when multiple are present.
4. NO GUESSING: Only state facts that appear in the CONTEXT. If the student asks
   about a batch/year not covered by the CONTEXT, say so plainly rather than
   inferring it from a different batch.
5. If the student's question doesn't specify a batch and multiple batches are
   present in the CONTEXT, ask which batch they mean OR briefly summarize all of
   them, whichever is more helpful given the question.
6. dont give me half responses try to give full responses dont cut off 
7. Now that i have added the FAQ section to the context, you can use it to answer questions about the program.

CONTEXT:
{context}

QUESTION:
{question}

Answer clearly, and mention the batch/year for every course fact you state.
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

    # direct_answer = _extract_course_code(query, retrieved_results)
    # if direct_answer:
    #     logger.info(f"Using direct course-code extraction:\n{direct_answer}")
    #     if "\n" in direct_answer:
    #         # multiple batches matched with (possibly) different codes
    #         return "The course code has changed across batches:\n" + direct_answer
    #     return f"The course code is {direct_answer}."

    logger.debug(f"Full context being sent to Gemini:\n{'─'*60}\n{context}\n{'─'*60}")
    logger.info("Sending context + question to Gemini...")

    answer = chain.invoke({"context": context, "question": query})

    logger.info("Answer received.")
    return answer
#jds