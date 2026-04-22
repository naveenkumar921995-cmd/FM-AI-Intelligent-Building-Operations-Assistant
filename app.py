import streamlit as st
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="FM AI Assistant", layout="wide")

# ---------------- CORPORATE STYLE ---------------- #
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
    color: #1f2937;
}

section[data-testid="stSidebar"] {
    background-color: #eef2f7;
}

section[data-testid="stSidebar"] * {
    color: #1f2937 !important;
}

.card {
    padding: 15px;
    border-radius: 8px;
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    margin-bottom: 10px;
    color: #1f2937;
}

h1, h2, h3 {
    color: #1e3a8a;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("🏢 FM AI System")
st.sidebar.markdown("Facility Management Assistant")

# Department Mapping (NEW)
system_map = {
    "All": "All",
    "HVAC": "hvac",
    "Fire System": "fire",
    "Electrical": "electrical",
    "DG System": "dg",
    "STP Plant": "stp",
    "WTP Plant": "wtp",
    "Lift / Elevators": "lift",
    "CCTV": "cctv",
    "Access Control": "access_control",
    "Soft Services": "soft_services",
    "IBMS": "ibms",
    "Plumbing": "plumbing"
}

selected_label = st.sidebar.selectbox("Select System", list(system_map.keys()))
system_filter = system_map[selected_label]

# ---------------- EMBEDDINGS ---------------- #
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---------------- LLM ---------------- #
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

DB_PATH = "vectorstore/"

# ---------------- LOAD VECTOR DB ---------------- #
@st.cache_resource
def load_vectorstore():
    if not os.path.exists(DB_PATH):
        st.warning("⚠️ Creating knowledge base...")

        from src.ingestion.ingest import load_documents, split_documents, create_vectorstore

        docs = load_documents()
        chunks = split_documents(docs)
        create_vectorstore(chunks)

    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()

# ---------------- HEADER ---------------- #
st.title("🏢 Intelligent Building Operations Assistant")
st.markdown("### AI-powered Facility Management System")

# ---------------- INPUT ---------------- #
query = st.text_input("🔍 Ask your question")

# ---------------- AI FUNCTION ---------------- #
def generate_ai_response(docs, query):
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a senior Facility Management Engineer working in a corporate building like DLF Cyber Park.

Use the given maintenance manuals to answer professionally.

Context:
{context}

Question:
{query}

Give response in this format:

1. Problem Diagnosis  
2. Top Causes (max 3)  
3. Recommended Actions (step-wise)

Keep answer clear, practical and technical.
"""

    response = llm.invoke(prompt)
    return response.content

# ---------------- MAIN ---------------- #
if query:
    docs = vectorstore.similarity_search(query, k=5)

    # Apply system filter
    if system_filter != "All":
        docs = [doc for doc in docs if doc.metadata.get("system") == system_filter]

    if not docs:
        st.error("❌ No relevant data found")
    else:
        st.subheader("🧠 AI Diagnosis")

        with st.spinner("Analyzing issue..."):
            answer = generate_ai_response(docs, query)

        st.markdown(answer)

        # ---------------- SOURCES ---------------- #
        st.subheader("📌 Source Documents")

        shown = set()
        for doc in docs:
            source = f"{doc.metadata.get('system')} | {doc.metadata.get('source_file')}"
            if source not in shown:
                st.write(f"• {source}")
                shown.add(source)
