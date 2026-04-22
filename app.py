import streamlit as st
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------- UI CONFIG ---------------- #
st.set_page_config(page_title="FM AI Assistant", layout="wide")

# ---------------- CUSTOM STYLE ---------------- #
st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: white;
}

/* Card style FIXED */
.card {
    padding: 15px;
    border-radius: 10px;
    background-color: #1c1f26;
    margin-bottom: 10px;
    color: #ffffff;   /* 👈 FIX: force white text */
    font-size: 16px;
}

/* Improve headings */
h1, h2, h3 {
    color: #ffffff;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111;
}
</style>
""", unsafe_allow_html=True)
# ---------------- SIDEBAR ---------------- #
st.sidebar.title("🏢 FM AI System")
st.sidebar.markdown("Intelligent Facility Assistant")

system_filter = st.sidebar.selectbox(
    "Select System",
    ["All", "hvac", "fire", "electrical"]
)

# ---------------- EMBEDDINGS ---------------- #
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
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

query = st.text_input("🔍 Ask your question")

# ---------------- SMART ANSWER FUNCTION ---------------- #
def generate_smart_answer(docs):
    combined = " ".join([doc.page_content for doc in docs])

    sentences = combined.split(".")
    insights = []

    for s in sentences:
        s = s.strip()
        if len(s) > 40:
            insights.append(s)
        if len(insights) >= 3:
            break

    return insights

# ---------------- MAIN LOGIC ---------------- #
if query:
    docs = vectorstore.similarity_search(query, k=5)

    if system_filter != "All":
        docs = [doc for doc in docs if doc.metadata.get("system") == system_filter]

    if not docs:
        st.error("❌ No relevant data found")
    else:
        st.subheader("🧠 AI Diagnosis")

        insights = generate_smart_answer(docs)

        for i, point in enumerate(insights, 1):
            st.markdown(f"""
            <div class="card">
                <b>{i}. {point}</b>
            </div>
            """, unsafe_allow_html=True)

        # ---------------- ACTION SECTION ---------------- #
        st.subheader("⚙️ Recommended Action")

        st.info("""
        - Check system parameters immediately  
        - Inspect sensors / controllers  
        - Verify mechanical condition  
        """)

        # ---------------- SOURCE SECTION ---------------- #
        st.subheader("📌 Source Documents")

        shown = set()
        for doc in docs:
            source = f"{doc.metadata.get('system')} | {doc.metadata.get('source_file')}"
            if source not in shown:
                st.write(f"• {source}")
                shown.add(source)
