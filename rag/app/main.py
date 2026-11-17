from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.ai.rag.local_rag import initialize_rag, query_rag
from app.schemas.rag import QueryRequest, QueryResponse
from app import logs

from pydantic import BaseModel, Field, field_serializer


import asyncio

import os
from datetime import datetime, UTC
from http import HTTPStatus


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Initialize RAG index on startup
#     initialize_rag()
#     yield
#     # Clean up state if needed on shutdown

# app = FastAPI(lifespan=lifespan)

app = FastAPI()

# @app.post("/ask", response_model=QueryResponse)
# def ask_question(payload: QueryRequest):
#     result = query_rag(payload.question)
#     return result



@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/logs")
def get_logs(start: datetime, end: datetime, level: str=None):
    if start >= end:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail='Start must be before end date'
        )
    if not level or not logs.is_valid_level(level):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail='invalid log level'
        )
        
    records = logs.query(start, end, level)
    if not records:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="Not found."
        )
    return {
        'count': len(records),
        'records': records
    }


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}


@app.get('/sleep')
async def sleep():
    await asyncio.sleep(1)
    return {'slept': 1}

@app.get('/info')
def info():
    # await asyncio.sleep(2)
    return {
        'version': 3.24, 
        'user': os.environ.get("USER"),
        'date': datetime.now(tz=UTC)
    }