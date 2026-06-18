from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import polars as pl
from fastapi import FastAPI
from pydantic import BaseModel, Field
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


app = FastAPI(
    lifespan=lifespan,
    title="Lending Club ML API",
    description=(
        "API de prédiction de défaut de remboursement de prêt, "
        "basée sur un modèle ML entraîné sur les données Lending Club."
    ),
    version="0.1.0",
)


class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., description="Montant du prêt en USD", example=3600.0)
    int_rate: float = Field(..., description="Taux d'intérêt annuel (%)", example=13.99)
    installment: float = Field(..., description="Mensualité en USD", example=123.03)
    annual_inc: float = Field(..., description="Revenu annuel de l'emprunteur en USD", example=55000.0)
    dti: float = Field(..., description="Ratio dette/revenu (%)", example=5.91)
    delinq_2yrs: float = Field(..., description="Nombre d'incidents de paiement sur 2 ans", example=0.0)
    fico_range_low: float = Field(..., description="Borne basse du score FICO", example=675.0)
    fico_range_high: float = Field(..., description="Borne haute du score FICO", example=679.0)
    inq_last_6mths: float = Field(..., description="Nombre de demandes de crédit sur 6 mois", example=1.0)
    open_acc: float = Field(..., description="Nombre de lignes de crédit ouvertes", example=7.0)
    pub_rec: float = Field(..., description="Nombre d'incidents publics (faillites, etc.)", example=0.0)
    revol_bal: float = Field(..., description="Solde revolving total en USD", example=2765.0)
    revol_util: float = Field(..., description="Taux d'utilisation du crédit revolving (%)", example=29.7)
    total_acc: float = Field(..., description="Nombre total de lignes de crédit", example=13.0)
    term: str = Field(..., description="Durée du prêt", example=" 36 months")
    grade: str = Field(..., description="Note de risque Lending Club (A–G)", example="C")
    sub_grade: str = Field(..., description="Sous-note de risque (ex: C4)", example="C4")
    emp_length: str = Field(..., description="Ancienneté professionnelle", example="10+ years")
    home_ownership: str = Field(..., description="Statut résidentiel (RENT, OWN, MORTGAGE…)", example="MORTGAGE")
    verification_status: str = Field(..., description="Statut de vérification du revenu", example="Not Verified")
    purpose: str = Field(..., description="Objet du prêt", example="debt_consolidation")


class PredictionResponse(BaseModel):
    default_probability: float = Field(..., description="Probabilité de défaut entre 0 et 1", example=0.23)
    prediction: str = Field(..., description="Classe prédite : 'fully_paid' ou 'charged_off'", example="fully_paid")


@app.get("/health", tags=["Monitoring"], summary="Vérification de l'état du service")
def health():
    """Retourne `ok` si l'API est opérationnelle."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"], summary="Prédire le risque de défaut")
def predict(loan: LoanApplication):
    """
    Prédit si un emprunteur va rembourser son prêt (`fully_paid`) ou faire défaut (`charged_off`).

    Retourne la probabilité de défaut et la classe prédite.
    """
    assert _pipeline is not None
    df = pl.DataFrame([loan.model_dump()])
    X = build_features(df).to_pandas().to_numpy(dtype=float)
    prob = float(_pipeline.predict_proba(X)[0, 1])
    label = "charged_off" if _pipeline.predict(X)[0] == 1 else "fully_paid"
    log.info("prediction — probability: %.4f, label: %s", prob, label)
    return PredictionResponse(default_probability=prob, prediction=label)
