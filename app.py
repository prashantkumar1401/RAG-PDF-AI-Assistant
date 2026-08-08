import io

import faiss
import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="RAG PDF AI Assistant", page_icon="📄")
st.title("📄 RAG PDF AI Assistant")
st.caption("Upload a text-based PDF and retrieve semantically relevant passages.")


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


def chunk_text(text: str, size: int = 700, overlap: int = 100):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_index(chunks):
    model = load_model()
    vectors = model.encode(chunks, normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded:
    try:
        text = extract_text(uploaded.getvalue())
    except Exception as exc:
        st.error(f"Unable to read the PDF: {exc}")
        st.stop()

    if not text:
        st.error("No extractable text was found in this PDF.")
        st.stop()

    chunks = chunk_text(text)
    index = build_index(chunks)
    model = load_model()
    st.success(f"Indexed {len(chunks)} text chunks.")

    question = st.text_input("Ask a question about the PDF")
    k = st.slider("Retrieved passages", 1, min(5, len(chunks)), min(3, len(chunks)))

    if question.strip():
        query = model.encode([question], normalize_embeddings=True).astype("float32")
        scores, ids = index.search(query, k)

        st.subheader("Retrieved context")
        for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), 1):
            st.markdown(f"**Passage {rank} — similarity {score:.3f}**")
            st.write(chunks[int(idx)])
