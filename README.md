# RAG PDF AI Assistant

A beginner-friendly Retrieval-Augmented Generation (RAG) application that lets users upload a PDF, retrieve semantically relevant passages, and ask questions about the document.

## Features
- PDF text extraction
- Text chunking with overlap
- Sentence-transformer embeddings
- FAISS vector similarity search
- Interactive Streamlit interface
- Retrieved passages with similarity scores

## Architecture

```text
PDF → text extraction → chunking → embeddings → FAISS search → relevant context
```

## Tech Stack

- Python
- Streamlit
- PyPDF2
- Sentence Transformers
- FAISS
- NumPy

## Run Locally

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works

1. Upload a text-based PDF.
2. The application extracts its text.
3. Text is split into overlapping chunks.
4. Chunks are converted into embeddings.
5. FAISS indexes the embeddings.
6. A question is converted into an embedding.
7. The closest passages are returned as context.

## Important Scope Note

This implementation demonstrates the **retrieval** part of RAG. It intentionally does not claim to be a full production LLM question-answering system. An LLM generation layer can be added as a future enhancement.

## Future Improvements

- Add an LLM answer-generation layer
- Add conversation history
- Support DOCX files
- Add source/page citations
- Add automated tests
- Add Docker configuration
- Add document management for multiple PDFs

## Portfolio Skills

RAG fundamentals • embeddings • vector search • Python • document processing • Streamlit • semantic retrieval

## Author

**Prashant Kumar**

GitHub: https://github.com/prashantkumar1401
