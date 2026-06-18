from pathlib import Path

import mlflow
import polars as pl
import pytest

from lending.data import clean
from lending.train import load_processed_years, make_pipeline, train, main
import lending.train as train_module


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path: Path, monkeypatch):
    import lending.config as cfg
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    artifact_root = str(tmp_path / "artifacts")
    mlflow.set_tracking_uri(uri)
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", uri)
    mlflow.MlflowClient().create_experiment(cfg.MLFLOW_EXPERIMENT_NAME, artifact_location=artifact_root)
    yield
    mlflow.set_tracking_uri(None)


def _write_year_csv(directory: Path, year: int, n_rows: int = 3) -> None:
    statuses = (["Fully Paid", "Charged Off"] * ((n_rows // 2) + 1))[:n_rows]
    pl.DataFrame(
        {
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
        }
    ).write_csv(directory / f"{year}.csv")


# --- main ---


def test_main_registers_model_to_mlflow(tmp_path, monkeypatch):
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    for year in range(2007, 2013):
        _write_year_csv(data_dir, year, n_rows=20)

    monkeypatch.setattr(train_module.config, "PROCESSED_DIR", data_dir)
    monkeypatch.setattr(train_module.config, "TRAIN_YEARS", range(2007, 2013))
    monkeypatch.setattr(train_module, "clean", clean)

    main()

    champion = mlflow.MlflowClient().get_model_version_by_alias(
        train_module.config.MODEL_NAME, "champion"
    )
    assert champion is not None


# --- load_processed_years ---


def test_load_processed_years_concatenates_requested_years(tmp_path):
    for year in (2010, 2011, 2012):
        _write_year_csv(tmp_path, year, n_rows=3)

    df = load_processed_years(range(2010, 2012), tmp_path)

    assert len(df) == 6  # 2010 + 2011 only


def test_load_processed_years_raises_if_file_missing(tmp_path):
    _write_year_csv(tmp_path, 2010)

    with pytest.raises(FileNotFoundError):
        load_processed_years(range(2010, 2012), tmp_path)


# --- train ---


def _training_df(tmp_path: Path) -> pl.DataFrame:
    for year in (2010, 2011):
        _write_year_csv(tmp_path, year, n_rows=30)
    return clean(load_processed_years(range(2010, 2012), tmp_path))


def test_train_returns_metrics_with_auc_and_report(tmp_path):
    df = _training_df(tmp_path)
    _, metrics = train(df)

    assert "auc_roc" in metrics
    assert "report" in metrics
    assert 0.0 <= metrics["auc_roc"] <= 1.0
    assert "Fully Paid" in metrics["report"]


def test_train_pipeline_predicts_probabilities_in_range(tmp_path):
    df = _training_df(tmp_path)
    pipeline, _ = train(df)

    import numpy as np
    from lending.features import build_features

    X = build_features(df).to_pandas().to_numpy(dtype=float)
    probs = pipeline.predict_proba(X)[:, 1]

    assert probs.shape == (len(df),)
    assert np.all(probs >= 0) and np.all(probs <= 1)


# ---------------------------------------------------------------------------
# Cycle 7 — train() logs a run to MLflow experiment "lending-model-comparison"
# ---------------------------------------------------------------------------


def test_train_logs_run_to_mlflow(tmp_path):
    df = _training_df(tmp_path)
    train(df)

    experiment = mlflow.get_experiment_by_name("lending-model-comparison")
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    assert runs.iloc[0]["tags.mlflow.runName"] == "logistic_regression"
    assert 0.0 <= runs.iloc[0]["metrics.auc_roc"] <= 1.0
