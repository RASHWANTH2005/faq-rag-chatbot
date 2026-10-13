import os
from typing import TypedDict
from langgraph.graph import StateGraph , START , END
# from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.dense_retriever import DenseRetriever
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")




gen_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
)

verify_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=GROQ_API_KEY,
)

# llm = ChatOllama(
#     model="llama3.2",
#     base_url="http://localhost:11434",
#     temperature=0,
# )

embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
CHROMA_DIR = "./chroma_db"
chroma = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)



class State(TypedDict):
    query: str
    docs: list[Document]
    answer: str
    is_faithful: bool
    verify_reason: str
    retry_count: int


def retrieve_docs(state: State):
    # start = time.time()
    retriever = DenseRetriever(chroma=chroma)
    docs = retriever.retrieve(state["query"])
    # print(f"retrieve_docs took {time.time() - start:.2f}s")
    return {"docs": docs}

def generate_answer(state: State):

    query = state["query"]
    docs = state["docs"]

    # print("\n===== TOP 5 DOCS =====")

    # for i, doc in enumerate(docs[:5], 1):
    #     print(f"\n--- Document {i} ---")
    #     print("Question:", doc.metadata.get("question"))
    #     print("Answer:", doc.metadata.get("answer"))


    context = "\n\n".join(
        f"Question: {doc.metadata['question']}\n"
        f"Answer: {doc.metadata['answer']}"
        for doc in docs
    )

    prompt = f"""
    You are an FAQ assistant.Answer the user's question using ONLY the provided FAQ context.
    If the answer cannot be found in the context, say:"I don't know based on the available FAQs."
    User question:{query} FAQ context:{context} Give one clear and concise answer.
    """

    # print("\n===== gen PROMPT =====")
    # print(prompt)

    # start = time.time()
    response = gen_llm.invoke(prompt)
    # print(f"generate_answer took {time.time() - start:.2f}s")
    # print("\n===== gen RESPONSE =====")
    # print(response.content)

    return {
        "answer": response.content,
        "retry_count": state.get("retry_count", 0) + 1
    }


def verify_answer(state: State):
    docs = state["docs"]
    answer = state["answer"]

    # The fallback is always faithful by definition — skip the LLM check
    if answer.strip() == "I don't know based on the available FAQs.":
        return {"is_faithful": True, "verify_reason": "No relevant context found; correctly declined to answer."}

    context = "\n\n".join(
        f"Question: {doc.metadata['question']}\n"
        f"Answer: {doc.metadata['answer']}"
        for doc in docs
    )

    prompt = f"""
    You are a fact-checker.
    Context:{context}Answer to check:{answer}
    Does the answer ONLY use information found in the context above?
    Reply in this exact format:
    VERDICT: yes or no
    REASON: one sentence explanation
    """

    # print("\n===== verify PROMPT =====")
    # print(prompt)

    # start = time.time()
    response = verify_llm.invoke(prompt)
    # print(f"verify_answer took {time.time() - start:.2f}s")

    lines = response.content.strip().splitlines()

    verdict_line = next((l for l in lines if l.upper().startswith("VERDICT:")), "VERDICT: no")
    reason_line = next((l for l in lines if l.upper().startswith("REASON:")), "REASON: could not parse")

    is_faithful = "yes" in verdict_line.lower()
    reason = reason_line.split(":", 1)[-1].strip()

    return {
        "is_faithful": is_faithful,
        "verify_reason": reason
    }


def route_after_verify(state: State) -> str:
    if state["is_faithful"]:
        print("Verified faithful, ending.")
        return "end"

    if state["retry_count"] >= 2:
        print("Max retries hit, giving up and ending.")
        return "end"

    print(f"Not faithful (attempt {state['retry_count']}), retrying generation...")
    return "retry"


graph = StateGraph(State)

graph.add_node("retrieve", retrieve_docs)
graph.add_node("generate", generate_answer)
graph.add_node("verify", verify_answer)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "verify")
graph.add_conditional_edges(
    "verify",
    route_after_verify,
    {
        "retry": "generate",
        "end": END
    }
)

app = graph.compile()

# result = app.invoke({"query": "how to cancel my order"})

# print("Answer",result["answer"])
# print("Faithful:", result["is_faithful"])
# print("Reason:", result["verify_reason"])
# print("Retry count:", result["retry_count"])