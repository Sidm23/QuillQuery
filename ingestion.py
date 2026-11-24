
import pymupdf

from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load one PDF or TXT file
def load_document(file_path):
    file_path = Path(file_path)
    documents = []

    if file_path.suffix.lower() == ".pdf":

        with pymupdf.open(file_path) as pdf:

            for page_number, page in enumerate(pdf):

                text = page.get_text("text").strip()

                if not text:
                    print(
                        f"Skipping scanned/empty page: "
                        f"{file_path.name}, page {page_number + 1}"
                    )
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path.name,
                            "page": page_number + 1
                        }
                    )
                )

    elif file_path.suffix.lower() == ".txt":

        text = file_path.read_text(encoding="utf-8").strip()

        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "page": 1
                    }
                )
            )

    else:
        raise ValueError("Unsupported file format")

    return documents


# Load all supported files from a folder
def load_documents(folder_path):

    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    documents = []

    for file_path in sorted(folder_path.iterdir()):

        if file_path.suffix.lower() in [".pdf", ".txt"]:
            documents.extend(load_document(file_path))

    if not documents:
        raise ValueError("No readable text found in the documents folder")

    return documents


# Split documents into smaller chunks
def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    return text_splitter.split_documents(documents)


# Test ingestion
if __name__ == "__main__":

    documents = load_documents("documents")
    chunks = split_documents(documents)

    print("\nTotal pages/documents:", len(documents))
    print("Total chunks:", len(chunks))

    for doc in documents:
        print(
            doc.metadata,
            "| Characters:",
            len(doc.page_content)
        )
