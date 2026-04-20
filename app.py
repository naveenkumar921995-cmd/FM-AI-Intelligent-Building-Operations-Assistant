import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Page config
st.set_page_config(page_title="FM AI Assistant", layout="wide")

st.title("🏢 FM AI – Building Operations Assistant")
st.write("Ask questions about HVAC, Fire, Electrical systems")

st.sidebar.title("System Filter")
system = st.sidebar.selectbox("Select System", ["All", "HVAC", "Fire", "Electrical"])

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

DB_PATH = "vectorstore/"

@st.cache_resource
def load_vectorstore():
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()

# Input
query = st.text_input("🔍 Ask your question:")

if query:
    docs = vectorstore.similarity_search(query, k=5)

    st.subheader("🧠 AI Answer")

    combined_text = "\n".join([doc.page_content for doc in docs])
    lines = combined_text.split(".")

    count = 0
    for line in lines:
        line = line.strip()
        if len(line) > 40:
            st.write(f"• {line}")
            count += 1
        if count == 3:
            break

    st.subheader("📌 Sources")

    for doc in docs:
        st.write(f"- {doc.metadata.get('system')} | {doc.metadata.get('source_file')}")