from pathlib import Path

import joblib
import mlflow
import polars as pl
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lending import config
from lending.data import clean
from lending.evaluate import evaluate, main
from lending.features import build_features


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path: Path, monkeypatch):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    artifact_root = str(tmp_path / "artifacts")
    mlflow.set_tracking_uri(uri)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    mlflow.MlflowClient().create_experiment(config.MLFLOW_EXPERIMENT_NAME, artifact_location=artifact_root)
    yield
    mlflow.set_tracking_uri(None)


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
def eval_df() -> pl.DataFrame:
    return _make_df()


@pytest.fixture()
def trained_pipeline(eval_df: pl.DataFrame) -> Pipeline:
    X = build_features(eval_df).to_pandas().to_numpy(dtype=float)
    y = eval_df["target"].to_numpy()
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X, y)
    return pipeline


# ---------------------------------------------------------------------------
# Cycle 1 — tracer bullet: evaluate() returns auc_roc and f1_charged_off
# ---------------------------------------------------------------------------


def test_evaluate_returns_auc_and_f1(trained_pipeline, eval_df):
    metrics = evaluate(trained_pipeline, eval_df, min_auc=0.0)
    assert "auc_roc" in metrics
    assert "f1_charged_off" in metrics
    assert 0.0 <= metrics["auc_roc"] <= 1.0
    assert 0.0 <= metrics["f1_charged_off"] <= 1.0


# ---------------------------------------------------------------------------
# Cycle 2 — evaluate() raises ValueError when auc_roc < threshold
# ---------------------------------------------------------------------------


def test_evaluate_raises_below_threshold(trained_pipeline, eval_df):
    with pytest.raises(ValueError, match="AUC-ROC"):
        evaluate(trained_pipeline, eval_df, min_auc=1.0)


# ---------------------------------------------------------------------------
# Cycle 3 — evaluate() logs auc_roc and f1_charged_off to MLflow
# ---------------------------------------------------------------------------


def test_evaluate_logs_metrics_to_mlflow(trained_pipeline, eval_df):
    evaluate(trained_pipeline, eval_df, min_auc=0.0)

    experiment = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    assert runs.iloc[0]["tags.mlflow.runName"] == "evaluate"
    assert 0.0 <= runs.iloc[0]["metrics.auc_roc"] <= 1.0
    assert 0.0 <= runs.iloc[0]["metrics.f1_charged_off"] <= 1.0


# ---------------------------------------------------------------------------
# Cycle 4 — main() loads model from disk and evaluates on EVAL_YEARS
# ---------------------------------------------------------------------------


def test_main_loads_model_and_evaluates(tmp_path: Path, monkeypatch):
    import lending.evaluate as ev

    df = _make_df()
    df.write_csv(tmp_path / "2014.csv")

    X = build_features(df).to_pandas().to_numpy(dtype=float)
    y = df["target"].to_numpy()
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X, y)
    model_path = tmp_path / "best_model.joblib"
    joblib.dump(pipeline, model_path)

    monkeypatch.setattr(ev, "MODEL_PATH", model_path)
    monkeypatch.setattr(ev.config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(ev.config, "EVAL_YEARS", range(2014, 2015))
    monkeypatch.setattr(ev.config, "MIN_AUC_ROC", 0.0)

    main()
