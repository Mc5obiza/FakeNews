from datetime import datetime
from typing import Literal
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "backend" / "artifacts" / "fake_news_pipeline.joblib"


class PredictRequest(BaseModel):
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    subject: str = Field(default="")
    date: str = Field(..., description="Publication date in YYYY-MM-DD format")


class PredictResponse(BaseModel):
    prediction: Literal["real", "fake"]
    probability_real: float
    probability_fake: float


app = FastAPI(title="Fake News Detection API", version="1.0.0")
_model = None


def _contains_number(value: str) -> int:
    return int(any(char.isdigit() for char in value))


def _build_features(payload: PredictRequest) -> pd.DataFrame:
    try:
        parsed_date = datetime.strptime(payload.date, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format") from exc

    merged_text = f"{payload.title} {payload.text} {payload.subject}".strip()

    features = {
        "text": [merged_text],
        "year": [parsed_date.year],
        "month": [parsed_date.month],
        "day": [parsed_date.day],
        "title_length": [len(payload.title)],
        "has_num": [_contains_number(payload.text)],
    }
    return pd.DataFrame(features)


@app.on_event("startup")
def load_model() -> None:
    global _model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run train_model.py first to create artifacts."
        )

    _model = joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Welcome to the Fake News Detection API. Use /predict to classify news articles."}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = _build_features(payload)
    probability_real = float(_model.predict_proba(features)[0][1])
    probability_fake = 1.0 - probability_real
    prediction = "real" if probability_real >= 0.5 else "fake"

    return PredictResponse(
        prediction=prediction,
        probability_real=probability_real,
        probability_fake=probability_fake,
    )


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
