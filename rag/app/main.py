from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.ai.rag.local_rag import initialize_rag, query_rag
from app.schemas.rag import QueryRequest, QueryResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize RAG index on startup
    initialize_rag()
    yield
    # Clean up state if needed on shutdown

app = FastAPI(lifespan=lifespan)

@app.post("/ask", response_model=QueryResponse)
def ask_question(payload: QueryRequest):
    result = query_rag(payload.question)
    return result