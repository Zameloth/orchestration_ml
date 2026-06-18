from pathlib import Path

import mlflow
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from lending import config
from lending.data import clean
from lending.train import load_processed_years
from lending.train_optuna import tune, main


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path: Path, monkeypatch):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    artifact_root = str(tmp_path / "artifacts")
    mlflow.set_tracking_uri(uri)
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    mlflow.MlflowClient().create_experiment(config.MLFLOW_EXPERIMENT_NAME, artifact_location=artifact_root)
    yield
    mlflow.set_tracking_uri(None)


def _write_year_csv(directory: Path, year: int, n_rows: int = 30) -> None:
    statuses = (["Fully Paid", "Charged Off"] * ((n_rows // 2) + 1))[:n_rows]
    pl.DataFrame({
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
    }).write_csv(directory / f"{year}.csv")


@pytest.fixture()
def training_df(tmp_path: Path) -> pl.DataFrame:
    for year in (2010, 2011):
        _write_year_csv(tmp_path, year, n_rows=30)
    return clean(load_processed_years(range(2010, 2012), tmp_path))


# ---------------------------------------------------------------------------
# Cycle 1 — tracer bullet: tune() returns a fitted Pipeline
# ---------------------------------------------------------------------------


def test_tune_returns_fitted_pipeline(training_df):
    import numpy as np
    from lending.features import build_features

    pipeline, auc_roc = tune(training_df, n_trials=2, cv=2)

    assert isinstance(pipeline, Pipeline)
    assert isinstance(auc_roc, float)
    X = build_features(training_df).to_pandas().to_numpy(dtype=float)
    probs = pipeline.predict_proba(X)[:, 1]
    assert probs.shape == (len(training_df),)
    assert np.all(probs >= 0) and np.all(probs <= 1)


# ---------------------------------------------------------------------------
# Cycle 2 — tune() logs the best trial's auc_roc to MLflow
# ---------------------------------------------------------------------------


def test_tune_logs_best_trial_to_mlflow(training_df):
    tune(training_df, n_trials=2, cv=2)

    experiment = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    assert runs.iloc[0]["tags.mlflow.runName"] == "optuna_best"
    assert 0.0 <= runs.iloc[0]["metrics.auc_roc"] <= 1.0


# ---------------------------------------------------------------------------
# Cycle 3 — main() saves the tuned model to BEST_MODEL_PATH
# ---------------------------------------------------------------------------


def test_main_registers_best_model(tmp_path: Path, monkeypatch):
    import lending.train_optuna as tm

    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    for year in range(2007, 2013):
        _write_year_csv(data_dir, year, n_rows=20)

    monkeypatch.setattr(tm.config, "PROCESSED_DIR", data_dir)

    main(n_trials=2, cv=2)

    champion = mlflow.MlflowClient().get_model_version_by_alias(
        tm.config.MODEL_NAME, "champion"
    )
    assert champion is not None
