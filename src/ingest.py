from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from data_loader import load_documents


CHROMA_DIR = "./chroma_db"


def ingest():

    docs = load_documents()

    print(f"Loaded {len(docs)} documents")

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print("ChromaDB ingestion completed")


if __name__ == "__main__":
    ingest()