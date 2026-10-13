import os

import gradio as gr
import httpx

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def chat_fn(message: str, history: list) -> str:
    try:
        r = httpx.post(
            f"{API_URL}/chat",
            json={"query": message},
            timeout=120,  # the graph (retrieve -> generate -> verify) can be slow
        )
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        return f"Backend error {e.response.status_code}: {e.response.text}"
    except httpx.TimeoutException:
        return "The request timed out. Please try again."
    except httpx.HTTPError as e:
        return f"Backend error: {e}"

    state = r.json()["state"]
    answer = state.get("answer")  # adjust key to your state schema
    return answer


demo = gr.ChatInterface(
    fn=chat_fn,
    title="FAQ Chatbot",
    description="Ask a question about the FAQ knowledge base.",
    examples=["How do I reset my password?", "What is your refund policy?"],
)

if __name__ == "__main__":
    demo.launch()