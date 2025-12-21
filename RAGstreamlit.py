# RAGstreamlit_offline.py
import os
import tempfile
from typing import List

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

st.set_page_config(
    page_title="Offline Resume RAG (No AI)",
    page_icon="📄",
    layout="wide"
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
# Build vectorstore (OFFLINE)
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

    # ✅ LOCAL embeddings (NO INTERNET AFTER FIRST DOWNLOAD)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# -------------------------
# UI
# -------------------------
st.title("📄 Offline Resume RAG (No Internet)")
st.markdown(
    """
    ✅ Works **fully offline**  
    ✅ No API keys  
    ✅ No LLMs  
    🔍 Shows the most relevant resume sections
    """
)

uploaded_files = st.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True,
)

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


# Index button
if uploaded_files and st.button("Index resumes"):
    with st.spinner("Indexing resumes (offline)..."):
        try:
            paths = [save_uploaded_file(f) for f in uploaded_files]
            st.session_state.vector_db = build_vectorstore_from_pdf_paths(paths)
            st.success(f"Indexed {len(paths)} resumes successfully!")
        except Exception as e:
            st.error(f"Indexing failed: {e}")


# -------------------------
# Search (NO LLM)
# -------------------------
if st.session_state.vector_db:
    query = st.text_input("Search resumes (e.g. Python, Django, 5 years)")

    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a search query")
        else:
            with st.spinner("Searching resumes..."):
                docs = st.session_state.vector_db.similarity_search(
                    query,
                    k=5
                )

                st.subheader("🔍 Top Matching Resume Sections")

                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get("source", "Unknown")
                    st.markdown(f"### {i}. 📄 {source}")
                    st.code(doc.page_content[:800])
