import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

# Ensure backend directory is in sys.path for clean package imports
CURRENT_DIR = Path(__file__).parent.resolve()
BACKEND_DIR = CURRENT_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from llm.rag_pipeline import answer_query

app = FastAPI(
    title="NovaTRix AeroLLM API",
    description="AeroLLM RAG & Aviation Maintenance Intelligence Backend API",
    version="1.0.0"
)

# Enable CORS for Member 4 frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., example="A320 hydraulic low pressure during takeoff")
    top_k: int = Field(5, ge=1, le=20)


class SourceItem(BaseModel):
    id: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceItem]


@app.get("/health")
def health_check():
    """Health check endpoint for API status verification."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    """
    RAG Query endpoint.
    Retrieves top-K FAA SDR reports, generates grounded maintenance context, and queries LLM.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        pipeline_result = answer_query(query=request.query, top_k=request.top_k)

        sources = [
            SourceItem(
                id=doc.get("id", "N/A"),
                score=round(float(doc.get("score", 0.0)), 4)
            )
            for doc in pipeline_result.get("retrieved_sources", [])
        ]

        return QueryResponse(
            query=pipeline_result["query"],
            answer=pipeline_result["answer"],
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Pipeline execution error: {str(e)}")
