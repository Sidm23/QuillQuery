
import re
import json

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from ingestion import load_documents, split_documents


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# --------------------------------------------------
# SEMESTER DETECTION
# --------------------------------------------------

def get_semester_roman(query):

    semester_map = {
        "1": "I", "2": "II", "3": "III", "4": "IV",
        "5": "V", "6": "VI", "7": "VII", "8": "VIII"
    }

    ordinal_map = {
        "first": "I", "second": "II", "third": "III",
        "fourth": "IV", "fifth": "V", "sixth": "VI",
        "seventh": "VII", "eighth": "VIII"
    }

    query = query.lower()

    # Convert shorthand: sem 2 -> semester 2
    query = re.sub(r"\bsem\.?\s*(\d+)", r"semester \1", query)
    query = re.sub(r"\bsem\.?\b", "semester", query)

    for number, roman in semester_map.items():

        if re.search(
            rf"\b{number}(?:st|nd|rd|th)?\s+semester\b",
            query
        ) or re.search(rf"\bsemester\s+{number}\b", query):

            return roman

    for ordinal, roman in ordinal_map.items():

        if re.search(rf"\b{ordinal}\s+semester\b", query):
            return roman

    for roman in semester_map.values():

        if re.search(rf"\bsemester\s+{roman.lower()}\b", query):
            return roman

    return None


# --------------------------------------------------
# QUERY EXPANSION
# --------------------------------------------------

def expand_query(query):

    roman = get_semester_roman(query)

    if roman:

        return [
            query,
            f"B.Tech CSE Semester {roman} course titles and course codes"
        ]

    return [query]


# --------------------------------------------------
# BUILD RETRIEVAL SYSTEMS
# --------------------------------------------------

def create_vector_store(documents):

    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No text available to create vector database")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    return vector_store, bm25_retriever, len(chunks)


# --------------------------------------------------
# HYBRID RAG QUESTION ANSWERING
# --------------------------------------------------

def answer_question(vector_store, bm25_retriever, query):

    roman = get_semester_roman(query)

    results = []
    seen = set()

    # Special retrieval for semester questions
    if roman:

        all_chunks = bm25_retriever.docs
        matching_pages = set()

        # Find pages containing the semester heading.
        # The PDF uses "Semester II", not necessarily
        # "B.Tech CSE Semester II".
        heading_pattern = rf"\bSemester\s+{roman}\b"

        for doc in all_chunks:

            if re.search(
                heading_pattern,
                doc.page_content,
                re.IGNORECASE
            ):

                matching_pages.add((
                    doc.metadata["source"],
                    doc.metadata["page"]
                ))

        print("\n===== SEMESTER RETRIEVAL =====")
        print("Detected semester:", roman)
        print("Matching pages:", matching_pages)

        # Collect every chunk from the matching page(s)
        for doc in all_chunks:

            page_key = (
                doc.metadata["source"],
                doc.metadata["page"]
            )

            if page_key in matching_pages:

                key = (
                    doc.metadata["source"],
                    doc.metadata["page"],
                    doc.page_content
                )

                if key not in seen:
                    seen.add(key)
                    results.append(doc)

    # Normal hybrid retrieval if no semester page was found
    if not results:

        expanded_queries = expand_query(query)

        for search_query in expanded_queries:

            semantic_results = vector_store.similarity_search(
                search_query,
                k=4
            )

            keyword_results = bm25_retriever.invoke(search_query)

            for doc in semantic_results + keyword_results:

                key = (
                    doc.metadata["source"],
                    doc.metadata["page"],
                    doc.page_content
                )

                if key not in seen:
                    seen.add(key)
                    results.append(doc)

    # Debug retrieval results
    print("\n===== HYBRID RETRIEVAL RESULTS =====")

    for i, doc in enumerate(results):

        print(f"\nRESULT {i + 1}")
        print("Source:", doc.metadata["source"])
        print("Page:", doc.metadata["page"])
        print("-" * 40)
        print(doc.page_content[:1000])

    if not results:
        return "No relevant information found in the documents.", []

    # Prepare context with source references
    context = "\n\n".join(
        f"[Source: {doc.metadata['source']} | "
        f"Page: {doc.metadata['page']}]\n"
        f"{doc.page_content}"
        for doc in results
    )

    context = context[:7000]

    # Source-grounded prompt
    prompt = f"""
You are QuillQuery, an AI-powered document intelligence assistant.

Answer using ONLY the provided document context.

STRICT RULES:

1. Never invent or assume missing information.
2. If the answer is not supported by the context, say:
   "The provided documents do not contain this information."
3. Preserve course codes, subject names, numbers and dates exactly.
4. For semester questions, list all subjects and course codes
   belonging to the requested semester.
5. Do not mix subjects from different semesters.
6. Mention total credits if available.
7. Include source and page references for factual claims.
8. Use only source references explicitly provided in the context.
9. Never create fake source names or page numbers.
10. Format answers clearly using headings, tables or bullet points.

CITATION FORMAT:

[Source: filename, Page: number]

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{query}

ANSWER:
"""

    response = llm.invoke(prompt)

    # Collect source references
    sources = []
    seen_sources = set()

    for doc in results:

        source = (
            doc.metadata["source"],
            doc.metadata["page"]
        )

        if source not in seen_sources:
            seen_sources.add(source)
            sources.append(source)

    return response.content, sources


# --------------------------------------------------
# MCQ GENERATION
# --------------------------------------------------

def generate_mcqs(vector_store, topic, number_of_questions=5):

    docs = vector_store.similarity_search(
        topic,
        k=5
    )

    if not docs:
        return []

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    context = context[:6000]

    prompt = f"""
You are QuillQuery, an educational quiz generator.

Generate exactly {number_of_questions} multiple-choice questions
based ONLY on the provided document context.

Topic: {topic}

RULES:

- Each question must have exactly four options.
- Only one option must be correct.
- Include a short explanation.
- Do not use outside knowledge.
- Avoid duplicate questions.
- If the context does not contain enough information,
  generate fewer questions rather than inventing content.

Return ONLY valid JSON in this format:

{{
  "questions": [
    {{
      "question": "Question text",
      "options": {{
        "A": "Option A",
        "B": "Option B",
        "C": "Option C",
        "D": "Option D"
      }},
      "answer": "A",
      "explanation": "Short explanation"
    }}
  ]
}}

DOCUMENT CONTEXT:

{context}
"""

    response = llm.invoke(prompt)

    try:

        content = response.content.strip()

        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        result = json.loads(content)
        questions = result.get("questions", [])

        valid_questions = []

        for q in questions:

            if (
                isinstance(q, dict)
                and isinstance(q.get("question"), str)
                and isinstance(q.get("options"), dict)
                and set(q["options"].keys()) == {"A", "B", "C", "D"}
                and q.get("answer") in ["A", "B", "C", "D"]
                and isinstance(q.get("explanation"), str)
            ):

                valid_questions.append(q)

        return valid_questions

    except (json.JSONDecodeError, AttributeError, TypeError):

        raise ValueError(
            "Could not generate valid MCQs. Please try again."
        )


# --------------------------------------------------
# LOCAL BACKEND TEST
# --------------------------------------------------

if __name__ == "__main__":

    documents = load_documents("documents")

    vector_store, bm25_retriever, total_chunks = create_vector_store(
        documents
    )

    print("Documents loaded:", len(documents))
    print("Chunks created:", total_chunks)

    query = input("\nAsk QuillQuery: ")

    answer, sources = answer_question(
        vector_store,
        bm25_retriever,
        query
    )

    print("\n===== ANSWER =====")
    print(answer)

    print("\n===== SOURCES =====")

    for source, page in sources:
        print(f"{source} | Page {page}")
