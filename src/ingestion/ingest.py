import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

DATA_PATH = "data/"
DB_PATH = "vectorstore/"

def load_documents():
    documents = []

    for system in os.listdir(DATA_PATH):
        system_path = os.path.join(DATA_PATH, system)

        if os.path.isdir(system_path):
            for file in os.listdir(system_path):
                if file.endswith(".pdf"):
                    file_path = os.path.join(system_path, file)
                    loader = PyPDFLoader(file_path)
                    pages = loader.load()

                    for page in pages:
                        page.metadata["system"] = system
                        page.metadata["source"] = file

                    documents.extend(pages)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(DB_PATH)
    print("Vector DB created")


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    create_vectorstore(chunks)