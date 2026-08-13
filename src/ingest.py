import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()


def ingest():
    with open("data/faqs.json") as f:
        faqs = json.load(f)

    docs = [
        Document(
            page_content=f"Question: {faq['question']}",
            metadata={"answer": faq["answer"]}
        )
        for faq in faqs[50:100]
    ]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="./chroma_db"
    )


if __name__ == "__main__":
    ingest()