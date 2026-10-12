from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from data_loader import load_documents

from retrieval.dense_retriever import DenseRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.rrf import rrf_fusion
from retrieval.reranker import CrossEncoderReranker


CHROMA_DIR = "./chroma_db"


class HybridRetriever:

    def __init__(
        self,
        dense_k=20,
        bm25_k=20,
        hybrid_k=20,
        final_k=5
    ):

        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.hybrid_k = hybrid_k
        self.final_k = final_k

       

        self.docs = load_documents()

      

        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )

      

        chroma = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )

       

        self.dense_retriever = DenseRetriever(
            chroma
        )

        self.bm25_retriever = BM25Retriever(
            self.docs
        )


        self.reranker = CrossEncoderReranker()

   

    def retrieve(self, query):
        print("\n[1] Starting dense retrieval...")
        dense_docs = self.dense_retriever.retrieve(
            query, k=self.dense_k
        )
        print(f"[1] Dense retrieval done: {len(dense_docs)} docs")

        print("\n[2] Starting BM25 retrieval...")
        bm25_docs = self.bm25_retriever.retrieve(
            query, k=self.bm25_k
        )
        print(f"[2] BM25 retrieval done: {len(bm25_docs)} docs")

        print("\n[3] Starting RRF fusion...")
        hybrid_docs = rrf_fusion(
            dense_docs,
            bm25_docs
        )
        print(f"[3] RRF done: {len(hybrid_docs)} docs")

        candidates = hybrid_docs[:self.hybrid_k]

        print(f"\n[4] Starting reranking of {len(candidates)} docs...")

        reranked = self.reranker.rerank(
            query,
            candidates,
            top_k=self.final_k
        )

        print("[4] Reranking done!")

        return [doc for doc, score in reranked]