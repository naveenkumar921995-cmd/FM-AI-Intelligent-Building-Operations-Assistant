import streamlit as st
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Page Config
st.set_page_config(page_title="FM AI Assistant", layout="wide")

# Sidebar Branding
st.sidebar.title("🏢 FM AI System")
st.sidebar.markdown("Facility Management Assistant")

# System Filter
system_filter = st.sidebar.selectbox(
    "Select System",
    ["All", "hvac", "fire", "electrical"]
)

# Load Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

DB_PATH = "vectorstore/"

# Load or Create Vector DB
@st.cache_resource
def load_vectorstore():
    if not os.path.exists(DB_PATH):
        st.warning("⚠️ Vector DB not found. Creating now...")

        from src.ingestion.ingest import load_documents, split_documents, create_vectorstore

        docs = load_documents()
        chunks = split_documents(docs)
        create_vectorstore(chunks)

    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)


vectorstore = load_vectorstore()

# Main UI
st.title("🏢 Intelligent Building Operations Assistant")
st.write("Ask questions related to HVAC, Fire, Electrical systems")

# User Input
query = st.text_input("🔍 Enter your question:")

if query:
    docs = vectorstore.similarity_search(query, k=5)

    # Apply System Filter
    if system_filter != "All":
        docs = [doc for doc in docs if doc.metadata.get("system") == system_filter]

    if not docs:
        st.error("❌ No relevant data found.")
    else:
        # AI Answer Section
        st.subheader("🧠 AI Diagnosis")

        points = []
        for doc in docs:
            text = doc.page_content.strip()
            if len(text) > 50:
                points.append(text)

        for i, point in enumerate(points[:3], 1):
            st.markdown(f"**{i}. {point}**")

        # Source Section
        st.subheader("📌 Sources")

        shown = set()
        for doc in docs:
            source = f"{doc.metadata.get('system')} | {doc.metadata.get('source_file')}"
            if source not in shown:
                st.write(f"- {source}")
                shown.add(source)
