
# 📚 QuillQuery
### Every document has an answer.

QuillQuery is an AI-powered document intelligence platform that allows users to upload documents, ask natural-language questions, retrieve context-aware answers, and generate interactive quizzes from their study material.

Built using Retrieval-Augmented Generation (RAG), hybrid search, local embeddings, and Large Language Models.

---

## 🚀 Features

### 1. Document Intelligence
- Upload PDF and TXT documents.
- Extract text and divide it into manageable chunks.
- Generate semantic embeddings locally.
- Retrieve relevant information using FAISS and BM25.
- Ask questions using natural language.
- Generate answers grounded in uploaded documents.
- Display source documents and page references.

### 2. AI-Powered Quiz Generator
- Generate multiple-choice questions from uploaded documents.
- Choose between 5 and 10 questions.
- Attempt interactive quizzes.
- Get automatic scoring.
- View correct answers and explanations.

### 3. Learning Analytics
- Track quiz attempts using SQLite.
- Calculate average quiz scores.
- Identify topics requiring improvement.
- Visualize performance trends.
- Compare topic-wise performance.
- Export quiz history as CSV.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| PDF Extraction | PyMuPDF |
| Text Chunking | LangChain |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Keyword Retrieval | BM25 |
| LLM | Groq API |
| Database | SQLite |
| Data Visualization | Streamlit Charts |

---

## 🏗️ Architecture

```text
             USER
               |
               v
        STREAMLIT UI
               |
       +-------+-------+
       |               |
       v               v
 DOCUMENT UPLOAD    USER QUERY
       |               |
       v               v
  PDF / TXT       QUERY PROCESSING
       |
       v
  TEXT EXTRACTION
       |
       v
  TEXT CHUNKING
       |
       v
 LOCAL EMBEDDINGS
       |
       v
   FAISS + BM25
       |
       +-------+-------+
               |
               v
        HYBRID RETRIEVAL
               |
               v
         GROQ LLM
               |
               v
       GROUNDED ANSWER
               |
               v
       SOURCE REFERENCES


       QUIZ GENERATION
               |
               v
           GROQ LLM
               |
               v
         MCQ QUIZ
               |
               v
       SCORE EVALUATION
               |
               v
        SQLITE DATABASE
               |
               v
     LEARNING ANALYTICS
```

---

## 📂 Project Structure

```text
document-rag/
│
├── documents/              # Sample documents
├── temp_documents/         # Uploaded documents
│
├── app.py                  # RAG backend and MCQ generation
├── ingestion.py            # Document loading and chunking
├── streamlit_app.py        # Streamlit application
├── database.py             # SQLite database operations
│
├── .env                    # API credentials
├── .gitignore
├── requirements.txt
├── README.md
│
└── quillquery.db           # Local quiz history database
```

---

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd document-rag
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Obtain an API key from:

https://console.groq.com/

Never commit your API keys to GitHub.

### 6. Run the application

```bash
streamlit run streamlit_app.py
```

Open the local URL displayed in your terminal.

---

## 🧠 How It Works

### Retrieval-Augmented Generation (RAG)

QuillQuery uses a hybrid retrieval approach:

1. Documents are extracted and split into smaller chunks.
2. Sentence Transformers generate local vector embeddings.
3. FAISS retrieves semantically similar chunks.
4. BM25 retrieves keyword-relevant chunks.
5. Retrieved context is passed to the Groq LLM.
6. The LLM generates an answer grounded in the retrieved context.

### Quiz Generation

1. The user enters a topic.
2. Relevant document chunks are retrieved.
3. The LLM generates MCQs using the retrieved content.
4. Users attempt the quiz.
5. Results are evaluated and saved in SQLite.

### Learning Analytics

Quiz attempts are stored locally in SQLite and used to calculate performance metrics, identify weaker topics, and visualize learning progress.

---

## ⚠️ Current Limitations

- Scanned and handwritten PDF OCR is not currently supported.
- PDF extraction works with readable digital text.
- Embedding generation runs locally and requires an initial model download.
- LLM generation requires a valid Groq API key.
- Quiz history is stored locally and is not synchronized across devices.
- Source references indicate retrieved pages; they do not guarantee that every generated claim has been independently verified.

---

## 🔮 Future Improvements

- OCR support for scanned and handwritten documents.
- Multimodal document extraction.
- More advanced competency-gap analysis.
- Personalized revision recommendations.
- Improved retrieval evaluation and citation verification.
- User accounts and cloud-based quiz history.

---

## 📄 License

This project is licensed under the MIT License.

---

### Built with Python, RAG, and a focus on making documents easier to understand.
