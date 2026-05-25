import fastapi
import pandas as pd

from fastapi import HTTPException
from pathlib import Path
from pydantic import BaseModel
from typing import List

from challenge.model import DelayModel


model = DelayModel()

# Train model at import time using the packaged training data.
# Path is resolved relative to this file so it works regardless of CWD.
_data_path = Path(__file__).parent.parent / "data" / "data.csv"
if _data_path.exists():
    _data = pd.read_csv(_data_path)
    _features, _target = model.preprocess(_data, target_column="delay")
    model.fit(_features, _target)


app = fastapi.FastAPI()


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
