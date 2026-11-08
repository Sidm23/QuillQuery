import pymupdf
import pytesseract
import io

from pathlib import Path
from PIL import Image

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# OCR for scanned or handwritten PDF pages
def extract_with_ocr(page):
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(3, 3),
        alpha=False
    )

    image = Image.open(io.BytesIO(pix.tobytes("png")))

    text = pytesseract.image_to_string(
        image,
        lang="eng",
        config="--psm 6"
    )

    return text.strip()


# Load one PDF or TXT file
def load_document(file_path):
    file_path = Path(file_path)
    documents = []

    if file_path.suffix.lower() == ".pdf":

        pdf = pymupdf.open(file_path)

        for page_number, page in enumerate(pdf):

            # First try normal text extraction
            text = page.get_text("text").strip()

            # If no text exists, use OCR
            if not text:
                print(f"Running OCR: {file_path.name}, page {page_number + 1}")
                text = extract_with_ocr(page)

            if not text:
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

        pdf.close()

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


# Load all PDFs and TXT files from a folder
def load_documents(folder_path):
    folder_path = Path(folder_path)
    documents = []

    for file_path in folder_path.iterdir():

        if file_path.suffix.lower() in [".pdf", ".txt"]:
            documents.extend(load_document(file_path))

    return documents


# Split documents into chunks
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