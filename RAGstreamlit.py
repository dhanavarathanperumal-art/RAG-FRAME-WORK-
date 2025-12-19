# RAGstreamlit_OpenRouter_Fixed.py

import os
import tempfile
from typing import List

import streamlit as st
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# -------------------------
# Streamlit config
# -------------------------
st.set_page_config(
    page_title="Resume RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# -------------------------
# Sidebar / API key
# -------------------------
st.sidebar.header("🔑 OpenRouter API Key")
openrouter_key = st.sidebar.text_input(
    "Enter OpenRouter API key",
    type="password",
    help="Starts with sk-or-v1-"
)

# -------------------------
# Helper: save uploaded file
# -------------------------
def save_uploaded_file(uploaded_file) -> str:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tf.write(uploaded_file.getbuffer())
    tf.close()
    return tf.name

# -------------------------
# Build vectorstore
# -------------------------
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

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# -------------------------
# OpenRouter LLM wrapper (FIXED)
# -------------------------
class OpenRouterLLM:
    def __init__(self, api_key, model="deepseek/deepseek-r1-0528:free"):
        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def __call__(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Resume RAG Chatbot"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0
        }

        response = requests.post(self.url, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter error: {response.text}")

        return response.json()["choices"][0]["message"]["content"]

# -------------------------
# UI
# -------------------------
st.title("📄 Resume RAG Chatbot")
st.markdown(
    "Upload bulk **PDF resumes** and ask questions like:\n\n"
    "**Who has 5+ years of Python experience?**"
)

uploaded_files = st.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True,
)

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# -------------------------
# Index resumes
# -------------------------
if uploaded_files and st.button("Index uploaded resumes ✅"):
    with st.spinner("Indexing resumes..."):
        try:
            paths = [save_uploaded_file(f) for f in uploaded_files]
            st.session_state.vector_db = build_vectorstore_from_pdf_paths(paths)
            st.success(f"Indexed {len(paths)} resumes successfully!")
        except Exception as e:
            st.error(f"Indexing failed: {e}")

# -------------------------
# RAG Search
# -------------------------
if st.session_state.vector_db:
    query = st.text_input("Ask a question about candidates")

    if st.button("Search 🔎"):
        if not openrouter_key:
            st.warning("Please enter your OpenRouter API key")
        elif not query.strip():
            st.warning("Please enter a query")
        else:
            with st.spinner("Searching..."):
                try:
                    docs_and_scores = st.session_state.vector_db.similarity_search_with_score(
                        query, k=4
                    )

                    if not docs_and_scores:
                        st.warning("No relevant documents found.")
                    else:
                        docs = [d for d, _ in docs_and_scores]
                        context_text = "\n\n".join(d.page_content for d in docs)

                        prompt_text = f"""
Use the following resume context to answer the question.
If the answer is not found, say so clearly.

Context:
{context_text}

Question:
{query}
"""

                        llm = OpenRouterLLM(openrouter_key)
                        answer = llm(prompt_text)

                        st.subheader("✅ Answer")
                        st.write(answer)

                except Exception as e:
                    st.error(str(e))
