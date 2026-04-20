import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load same embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

DB_PATH = "vectorstore/"


def load_vectorstore():
    print("📦 Loading vector database...")
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)


def ask_question(vectorstore, query):
    print(f"\n❓ Question: {query}")

    docs = vectorstore.similarity_search(query, k=5)

    if not docs:
        print("❌ No relevant data found.")
        return

    # Combine content
    combined_text = "\n".join([doc.page_content for doc in docs])

    # Simple smart formatting
    print("\n🧠 AI Answer:\n")

    lines = combined_text.split(".")
    
    # Take top meaningful lines
    answer_points = []
    for line in lines:
        line = line.strip()
        if len(line) > 40:
            answer_points.append(line)
        if len(answer_points) == 3:
            break

    for i, point in enumerate(answer_points, 1):
        print(f"{i}. {point}\n")

    print("📌 Sources:")
    for doc in docs:
        print(f"- {doc.metadata.get('system')} | {doc.metadata.get('source_file')}")