FAQ RAG Chatbot

A retrieval-augmented generation (RAG) chatbot that answers customer FAQ-style questions using only the provided knowledge base — built to explore how LLMs can be grounded in real data instead of relying on general training knowledge, in the context of customer support/success use cases.

How it works
FAQ data (question/answer pairs) is embedded and stored in a Chroma vector database.
When a user asks a question, the relevant FAQ entries are retrieved via vector similarity search.
The retrieved context is passed, along with the user's question, to Google Gemini to generate a grounded, tailored answer.
The whole pipeline is orchestrated with LangChain (LCEL), and served through a Streamlit chat interface.

Architecture
FAQ JSON  →  Embeddings  →  Chroma (vector store)
                                    │
User question ──────────────► Retriever (top-k similarity search)
                                    │
                          Retrieved context + question
                                    │
                              Prompt template
                                    │
                            Gemini (generation)
                                    │
                              Streamlit UI

Tech stack
LangChain — orchestration (LCEL chains, prompt templates, document loaders)
Chroma — vector database for storing and retrieving embedded FAQ entries
Google Gemini (gemini-3.5-flash) — response generation
Streamlit — chat interface

Possible next steps
Add conversation memory so the bot handles multi-turn follow-up questions
Add hybrid (keyword + vector) search for better recall on exact-term queries
Add a feedback mechanism (thumbs up/down) to evaluate answer quality over time
Swap in a larger FAQ dataset or connect to a live support ticketing system

