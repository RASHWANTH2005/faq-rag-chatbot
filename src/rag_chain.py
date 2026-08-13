# src/rag_chain.py
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

def build_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    template = ChatPromptTemplate.from_messages([
        ("system", "Answer using only the provided FAQ context. "
                   "If it's not covered, say you don't know."),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

    def format_docs(docs):
        return "\n\n".join(f"Q: {d.page_content}\nA: {d.metadata['answer']}" for d in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | template
        | model
        | StrOutputParser()
    )