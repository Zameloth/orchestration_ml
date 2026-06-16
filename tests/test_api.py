from __future__ import annotations

from pathlib import Path

import joblib
import polars as pl
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from starlette.testclient import TestClient

from lending.data import clean
from lending.features import build_features


def _make_df(n_rows: int = 40) -> pl.DataFrame:
    statuses = (["Fully Paid", "Charged Off"] * ((n_rows // 2) + 1))[:n_rows]
    return clean(pl.DataFrame({
        "loan_status": statuses,
        "loan_amnt": [float(i * 1000) for i in range(1, n_rows + 1)],
        "int_rate": [10.0] * n_rows,
        "installment": [200.0] * n_rows,
        "annual_inc": [50000.0] * n_rows,
        "dti": [15.0] * n_rows,
        "delinq_2yrs": [0.0] * n_rows,
        "fico_range_low": [700.0] * n_rows,
        "fico_range_high": [704.0] * n_rows,
        "inq_last_6mths": [1.0] * n_rows,
        "open_acc": [10.0] * n_rows,
        "pub_rec": [0.0] * n_rows,
        "revol_bal": [5000.0] * n_rows,
        "revol_util": [30.0] * n_rows,
        "total_acc": [20.0] * n_rows,
        "term": [" 36 months"] * n_rows,
        "grade": ["A"] * n_rows,
        "sub_grade": ["A1"] * n_rows,
        "emp_length": ["2 years"] * n_rows,
        "home_ownership": ["RENT"] * n_rows,
        "verification_status": ["Verified"] * n_rows,
        "purpose": ["debt_consolidation"] * n_rows,
    }))


@pytest.fixture()
def model_path(tmp_path: Path) -> Path:
    df = _make_df()
    X = build_features(df).to_pandas().to_numpy(dtype=float)
    y = df["target"].to_numpy()
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X, y)
    path = tmp_path / "best_model.joblib"
    joblib.dump(pipeline, path)
    return path


@pytest.fixture()
def client(model_path: Path, monkeypatch) -> TestClient:
    import lending.api as api_module
    monkeypatch.setattr(api_module, "MODEL_PATH", model_path)
    from lending.api import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Cycle 1 — tracer bullet: /health returns 200
# ---------------------------------------------------------------------------


def test_health_returns_ok(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Cycle 2 — /predict returns default_probability and prediction label
# ---------------------------------------------------------------------------

VALID_LOAN = {
    "loan_amnt": 10000.0,
    "int_rate": 10.0,
    "installment": 200.0,
    "annual_inc": 50000.0,
    "dti": 15.0,
    "delinq_2yrs": 0.0,
    "fico_range_low": 700.0,
    "fico_range_high": 704.0,
    "inq_last_6mths": 1.0,
    "open_acc": 10.0,
    "pub_rec": 0.0,
    "revol_bal": 5000.0,
    "revol_util": 30.0,
    "total_acc": 20.0,
    "term": " 36 months",
    "grade": "A",
    "sub_grade": "A1",
    "emp_length": "2 years",
    "home_ownership": "RENT",
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
}


def test_predict_returns_probability_and_label(client: TestClient):
    response = client.post("/predict", json=VALID_LOAN)
    assert response.status_code == 200
    body = response.json()
    assert "default_probability" in body
    assert "prediction" in body
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["prediction"] in {"charged_off", "fully_paid"}


# ---------------------------------------------------------------------------
# Cycle 3 — /predict with missing field returns 422
# ---------------------------------------------------------------------------


def test_predict_missing_field_returns_422(client: TestClient):
    incomplete = {k: v for k, v in VALID_LOAN.items() if k != "annual_inc"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Cycle 4 — prediction label is consistent with probability (>= 0.5 ↔ charged_off)
# ---------------------------------------------------------------------------


def test_predict_label_consistent_with_probability(client: TestClient):
    response = client.post("/predict", json=VALID_LOAN)
    body = response.json()
    prob = body["default_probability"]
    label = body["prediction"]
    if prob >= 0.5:
        assert label == "charged_off"
    else:
        assert label == "fully_paid"


# ---------------------------------------------------------------------------
# Cycle 5 — /predict logs probability and label at INFO level
# ---------------------------------------------------------------------------


def test_predict_logs_probability_and_label(client: TestClient, caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="lending.api"):
        response = client.post("/predict", json=VALID_LOAN)
    body = response.json()
    assert any(
        str(round(body["default_probability"], 4)) in r.message and body["prediction"] in r.message
        for r in caplog.records
    )
