import os
import tempfile
from typing import List
import requests
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# =========================
# CONFIG
# =========================
OPENROUTER_API_KEY = os.getenv("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")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "deepseek/deepseek-r1-0528:free"

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set in environment")


# =========================
# STREAMLIT SETUP
# =========================
st.set_page_config(
    page_title="Resume RAG Bot",
    page_icon="🤖",
    layout="wide"
)

st.title("📄 Resume RAG Chatbot")
st.markdown(
    "Upload bulk **PDF resumes** and ask questions like:\n\n"
    "**Explain this resume** or **Who has Python experience?**"
)


# =========================
# HELPERS
# =========================
def save_uploaded_file(uploaded_file) -> str:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tf.write(uploaded_file.getbuffer())
    tf.close()
    return tf.name


def build_vectorstore_from_pdf_paths(pdf_paths: List[str]):
    all_docs: List[Document] = []

    for p in pdf_paths:
        loader = PyPDFLoader(p)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = os.path.basename(p)

        all_docs.extend(docs)

    if not all_docs:
        raise ValueError("No documents loaded")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)


def call_openrouter(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Resume RAG Bot",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an expert resume analyst."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# =========================
# FILE UPLOAD
# =========================
uploaded_files = st.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True,
)

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


if uploaded_files and st.button("Index uploaded resumes ✅"):
    with st.spinner("Indexing resumes..."):
        try:
            paths = [save_uploaded_file(f) for f in uploaded_files]
            st.session_state.vector_db = build_vectorstore_from_pdf_paths(paths)
            st.success(f"Indexed {len(paths)} resumes successfully!")
        except Exception as e:
            st.error(f"Indexing failed: {e}")


# =========================
# SEARCH / RAG
# =========================
if st.session_state.vector_db:
    query = st.text_input("Ask a question about the resumes")

    if st.button("Search 🔎"):
        if not query.strip():
            st.warning("Please enter a question")
        else:
            with st.spinner("Analyzing resumes..."):
                try:
                    docs_and_scores = (
                        st.session_state.vector_db
                        .similarity_search_with_score(query, k=4)
                    )

                    if not docs_and_scores:
                        st.warning("No relevant resumes found.")
                    else:
                        docs = [doc for doc, _ in docs_and_scores]
                        context = "\n\n".join(d.page_content for d in docs)

                        final_prompt = f"""
Use the resume information below to answer the question.
If the answer is not present, clearly say so.

RESUME DATA:
{context}

QUESTION:
{query}
"""

                        answer = call_openrouter(final_prompt)

                        st.subheader("✅ Answer")
                        st.write(answer)

                except requests.HTTPError as e:
                    st.error(f"OpenRouter API error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
