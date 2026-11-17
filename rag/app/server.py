from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel, Field, field_serializer
import logging

from datetime import datetime, timedelta

from http import HTTPStatus
from typing import Annotated
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import db


app = FastAPI()

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
)

class Sale(BaseModel):
    time: datetime
    customer_id: str = Field(min_length=2)
    sku: str = Field(min_length=2)
    amount: int = Field(gt=0)
    price: float = Field(gt=0)
    

class TimeResponse(BaseModel):
    delta: timedelta
    
    @field_serializer('delta')
    def serialize_delta(self, v: timedelta) -> float:
        return v.total_seconds()
    
@app.get('/time_delta')
def time_diff(start: datetime, end: datetime) -> TimeResponse:
    delta = end - start
    return TimeResponse(delta=delta)

@app.post('/sales/')
def new_sale(sale: Sale):
    records = db.Sale(
        time = sale.time,
        customer_id = sale.customer_id,
        sku = sale.sku,
        amount = int(sale.amount * 100)
    )
    key = db.insert(records)
    return {
        'key': key
    }
    
@app.get('/sales/{key}')
def get_sale(key:str) -> Sale:
    record = db.get(key)
    if not record:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="sales not found"
        )
    
    s = Sale(
        time = record.time,
        sku = record.sku,
        customer_id = record.customer_id,
        amount = int(record.amount * 100),
        price = record.price / 100
    )
    
    return s


@app.post('/survey')
def survey(
    name: Annotated[str, Form()],
    happy: Annotated[str, Form()],
    talk: Annotated[str, Form()],
):
    logging.info('[survey] name: %r, happy: %r, talk: %r', name, happy, talk)
    return RedirectResponse(
        url='/static/thanks.html',
        status_code=HTTPStatus.FOUND,
    )