import fastapi
import pandas as pd

from fastapi import HTTPException
from pathlib import Path
from pydantic import BaseModel
from typing import List

from challenge.model import DelayModel


app = fastapi.FastAPI()

# Module-level reference so route handlers can reach the model without
# dependency injection. Read-only after startup — no concurrency risk.
model: DelayModel = None


@app.on_event("startup")
async def startup():
    global model
    model = DelayModel.load(Path(__file__).parent / "model.pkl")


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


class PredictRequest(BaseModel):
    flights: List[Flight]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    for flight in request.flights:
        if not (1 <= flight.MES <= 12):
            raise HTTPException(status_code=400, detail="MES must be between 1 and 12")
        if flight.TIPOVUELO not in ("I", "N"):
            raise HTTPException(status_code=400, detail="TIPOVUELO must be I or N")

    df = pd.DataFrame([flight.dict() for flight in request.flights])
    features = model.preprocess(df)
    predictions = model.predict(features)
    return {"predict": predictions}
