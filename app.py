
import streamlit as st
from src.rag_chain import build_chain

st.title("FAQ Chatbot")

# Cache the chain so it's built once per session, not on every rerun
@st.cache_resource
def get_chain():
    return build_chain()

rag_chain = get_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question := st.chat_input("Ask a question..."):
    st.chat_message("user").write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    answer = rag_chain.invoke(question)

    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})