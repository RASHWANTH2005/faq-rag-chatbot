FAQ RAG Chatbot

A production-style retrieval-augmented generation (RAG) chatbot that answers customer-support questions using only a provided FAQ knowledge base. It goes beyond a basic RAG demo: hybrid retrieval, cross-encoder reranking, a self-verifying LangGraph pipeline, automated quality evaluation with RAGAS as a CI gate, a FastAPI backend, and a Gradio frontend.

Highlights
1.Hybrid retrieval: BM25 (keyword) and ChromaDB (dense vectors) results are merged with Reciprocal Rank Fusion (RRF), so both exact-term and semantic queries are covered.
2.Cross-encoder reranking: retrieved candidates are re-scored with cross-encoder/ms-marco-MiniLM-L-6-v2 before generation.
Self-verifying generation (LangGraph): a retrieve → generate → verify loop checks each answer for faithfulness to the retrieved context and retries generation when it isn't grounded. Retries are capped.
3.Flexible model backends: run fully local with Ollama (Llama / Qwen) or use hosted APIs (Groq, Google Gemini).
4.Automated evaluation: an evaluation dataset scored with RAGAS (LLM-as-judge) runs in GitHub Actions and fails the build if quality drops below thresholds.
5.Served as an API: FastAPI backend with a /chat endpoint, plus a lightweight Gradio chat UI.


Tech stack
Area	            Tools
Orchestration	LangGraph, LangChain
Retrieval	      ChromaDB, BM25, RRF fusion, cross-encoder reranking
Models	      Ollama (Llama, Qwen), Groq API, Google Gemini API
Evaluation	      RAGAS, pytest, GitHub Actions
Observability	Langfuse
Backend	      FastAPI, Uvicorn
Frontend	      Gradio

