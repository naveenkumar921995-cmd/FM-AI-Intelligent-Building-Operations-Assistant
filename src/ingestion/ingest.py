import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH = "data/"
DB_PATH = "vectorstore/"


def load_documents():
    documents = []

    print("📂 Scanning data folder...")

    for system in os.listdir(DATA_PATH):
        system_path = os.path.join(DATA_PATH, system)

        if os.path.isdir(system_path):
            print(f"➡️ Processing system: {system}")

            for file in os.listdir(system_path):
                if file.lower().endswith(".pdf"):
                    file_path = os.path.join(system_path, file)

                    print(f"   📄 Loading file: {file}")

                    loader = PyPDFLoader(file_path)
                    pages = loader.load()

                    for page in pages:
                        page.metadata["system"] = system
                        page.metadata["source_file"] = file

                    documents.extend(pages)

    return documents


def split_documents(documents):
    print("✂️ Splitting documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    print("🧠 Creating embeddings (FREE model)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    print("📦 Creating vector database...")

    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(DB_PATH)

    print("✅ Vector DB created successfully!")


if __name__ == "__main__":

    docs = load_documents()
    print(f"📊 Total documents loaded: {len(docs)}")

    if not docs:
        print("❌ No documents found in data folder.")
        exit()

    chunks = split_documents(docs)
    print(f"📊 Total chunks created: {len(chunks)}")

    if not chunks:
        print("❌ No chunks created.")
        exit()

    create_vectorstore(chunks)