from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from src.graph import app as graph_app 

app = FastAPI(title="FAQ RAG Chatbot")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    state: dict[str, Any]


def to_jsonable(obj: Any) -> Any:
    """Recursively convert LangGraph state into JSON-safe types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "page_content"):      # langchain Document
        return {"page_content": obj.page_content, "metadata": to_jsonable(obj.metadata)}
    if hasattr(obj, "model_dump"):        # pydantic models / langchain messages
        return to_jsonable(obj.model_dump())
    return str(obj)                       # last resort for anything else


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        final_state = graph_app.invoke({"query": req.question})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph error: {e}")
    return ChatResponse(state=to_jsonable(final_state))