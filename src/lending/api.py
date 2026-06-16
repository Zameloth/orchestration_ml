from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import polars as pl
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

from lending import config
from lending.features import build_features

log = logging.getLogger(__name__)

MODEL_PATH: Path = config.BEST_MODEL_PATH

_pipeline: Pipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = joblib.load(MODEL_PATH)
    yield


app = FastAPI(lifespan=lifespan)


class LoanApplication(BaseModel):
    loan_amnt: float
    int_rate: float
    installment: float
    annual_inc: float
    dti: float
    delinq_2yrs: float
    fico_range_low: float
    fico_range_high: float
    inq_last_6mths: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    term: str
    grade: str
    sub_grade: str
    emp_length: str
    home_ownership: str
    verification_status: str
    purpose: str


class PredictionResponse(BaseModel):
    default_probability: float
    prediction: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(loan: LoanApplication):
    assert _pipeline is not None
    df = pl.DataFrame([loan.model_dump()])
    X = build_features(df).to_pandas().to_numpy(dtype=float)
    prob = float(_pipeline.predict_proba(X)[0, 1])
    label = "charged_off" if _pipeline.predict(X)[0] == 1 else "fully_paid"
    log.info("prediction — probability: %.4f, label: %s", prob, label)
    return PredictionResponse(default_probability=prob, prediction=label)
