from __future__ import annotations

import os
import uuid
from typing import Dict, Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import START

from src.agent import FinancialReasoningAgent
from src.dataloader import load_convfinqa_dataset, State
from src.prompts import PromptBuilder
from src.utils import get_context

app = FastAPI(
    title="ConvFinQA Service",
    description="HTTP API for chatting about ConvFinQA records.",
    version="1.0.0",
)

DATA_PATH = "data/convfinqa_dataset.json"

# Load data once at startup
train_df, test_df = load_convfinqa_dataset(DATA_PATH)

# Build quick index for record lookup
_RECORDS: Dict[str, object] = {r.id: r for r in train_df}

llm = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],  # Replace with your actual API key or load from env
    model="gpt-4o-mini",
)

print(llm)
# For few-shot examples (reuse small prefix like your Typer app)
_prompt_builder = PromptBuilder(train_df[:3])

class SessionData(BaseModel):
    record_id: str
    # We keep the graph and evolving State in memory for multi-turn dialogs
    # Graph can't be BaseModel, store separately in memory registry.
    session_id: str
    history_len: int = 0


class _InMemorySession:
    """
    Internal container for graph + state that we don't expose via Pydantic.
    """
    def __init__(self, record_id: str):
        record = _RECORDS.get(record_id)
        if not record:
            raise KeyError(f"Record {record_id} not found")

        self.record_id = record_id
        self.agent = FinancialReasoningAgent(llm, _prompt_builder)
        print(self.agent.llm)
        self.graph = self.agent.build_graph()

        context = get_context(record)
        self.state = State(
            question="",
            context=context,
            retriever="",
            steps_generated="",
            executor="",
            answer="",
            retrieval_history=[],
            generation_history=[],
            history=[],
            skip_generation=False,
        )

_sessions: Dict[str, _InMemorySession] = {}


class CreateSessionRequest(BaseModel):
    record_id: str = Field(..., description="ID of the record to chat about")


class CreateSessionResponse(BaseModel):
    session_id: str
    record_id: str
    document_pre_text: Optional[str] = None
    table_preview: Optional[str] = None
    post_text: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Active session id from /sessions")
    message: str = Field(..., description="User message/question")


class ChatResponse(BaseModel):
    session_id: str
    record_id: str
    answer: str
    history_len: int


class RecordInfo(BaseModel):
    record_id: str
    document_pre_text: Optional[str] = None
    table: Optional[str] = None
    post_text: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/records/{record_id}", response_model=RecordInfo)
def get_record(record_id: str):
    record = _RECORDS.get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Record {record_id} not found")
    return RecordInfo(
        record_id=record.id,
        document_pre_text=getattr(record.doc, "pre_text", None),
        table=str(getattr(record.doc, "table", None)),
        post_text=getattr(record.doc, "post_text", None),
    )


@app.post("/sessions", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest):
    record = _RECORDS.get(req.record_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Record {req.record_id} not found")

    session_id = uuid.uuid4().hex
    try:
        sess = _InMemorySession(req.record_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Record {req.record_id} not found")

    _sessions[session_id] = sess

    return CreateSessionResponse(
        session_id=session_id,
        record_id=req.record_id,
        document_pre_text=getattr(record.doc, "pre_text", None),
        table_preview=str(getattr(record.doc, "table", None))[:1000],  # trim to keep payload small
        post_text=getattr(record.doc, "post_text", None),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sess = _sessions.get(req.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    # Update state with latest question
    sess.state.question = req.message

    # Run the graph
    state_dict = sess.graph.invoke(sess.state, start=START)
    sess.state = State(**state_dict)

    answer = (sess.state.answer or "").strip()
    if not answer:
        answer = "I couldn't produce an answer for that question."

    history_len = len(sess.state.history) if hasattr(sess.state, "history") else 0

    return ChatResponse(
        session_id=req.session_id,
        record_id=sess.record_id,
        answer=answer,
        history_len=history_len,
    )


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
        return {"deleted": True}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/")
def root():
    return {
        "message": "ConvFinQA FastAPI is up. Create a session with POST /sessions, then chat via POST /chat."
    }
