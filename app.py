from ingestion import load_documents, split_documents
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv


# STEP 1: Load all documents
documents = load_documents("documents")

# STEP 2: Split documents
chunks = split_documents(documents)

print("Total documents:", len(documents))
print("Total chunks:", len(chunks))


# STEP 3: Generate embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# STEP 4: Create FAISS vector store
vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

print("\nFAISS vector store created successfully!")


# STEP 5: Ask a question
query = "What is RAG?"

results = vector_store.similarity_search_with_score(
    query,
    k=3
)


# STEP 6: Prepare context
context = "\n\n".join(
    doc.page_content for doc, score in results
)


# STEP 7: Connect Groq
load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

prompt = f"""
Answer the question using ONLY the context below.

If the answer is not available in the context,
clearly say that the information is not available.

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print("\n===== FINAL ANSWER =====")
print(response.content)


# STEP 8: Display sources
print("\n===== SOURCE REFERENCES =====")

seen_sources = set()

for doc, score in results:
    source = (
        doc.metadata["source"],
        doc.metadata["page"]
    )

    if source not in seen_sources:
        seen_sources.add(source)

        print("Source:", source[0])
        print("Page:", source[1])
        print("-" * 40)