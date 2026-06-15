from pathlib import Path

import mlflow
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from lending.data import clean
from lending.train import load_processed_years
from lending.train_models import compare_models


def _write_year_csv(directory: Path, year: int, n_rows: int = 20) -> None:
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


@pytest.fixture()
def training_df(tmp_path: Path) -> pl.DataFrame:
    for year in (2010, 2011):
        _write_year_csv(tmp_path, year, n_rows=30)
    return clean(load_processed_years(range(2010, 2012), tmp_path))


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path: Path, monkeypatch):
    import lending.config as cfg
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    artifact_root = str(tmp_path / "artifacts")
    mlflow.set_tracking_uri(uri)
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", uri)
    client = mlflow.MlflowClient()
    for name in (cfg.MLFLOW_EXPERIMENT_NAME, "test-experiment"):
        client.create_experiment(name, artifact_location=artifact_root)
    yield
    mlflow.set_tracking_uri(None)


# ---------------------------------------------------------------------------
# Cycle 1 — tracer bullet: compare_models returns the right types
# ---------------------------------------------------------------------------


def test_compare_models_returns_pipeline_and_name(training_df):
    pipeline, name, results = compare_models(training_df, cv=2)

    assert isinstance(pipeline, Pipeline)
    assert name in ("random_forest", "xgboost", "lightgbm")
    assert isinstance(results, dict)


# ---------------------------------------------------------------------------
# Cycle 2 — MLflow experiment is created with the right name
# ---------------------------------------------------------------------------


def test_compare_models_creates_mlflow_experiment(training_df):
    compare_models(training_df, cv=2, experiment_name="test-experiment")

    experiment = mlflow.get_experiment_by_name("test-experiment")
    assert experiment is not None


# ---------------------------------------------------------------------------
# Cycle 3 — parent run logs mean_auc_roc and std_auc_roc
# ---------------------------------------------------------------------------


def test_compare_models_parent_runs_log_mean_auc(training_df):
    compare_models(training_df, cv=2, experiment_name="test-experiment")

    experiment = mlflow.get_experiment_by_name("test-experiment")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.parentRunId IS NULL",
    )
    assert len(runs) == 3  # one per model
    for _, row in runs.iterrows():
        assert "metrics.mean_auc_roc" in row.index
        assert 0.0 <= row["metrics.mean_auc_roc"] <= 1.0


# ---------------------------------------------------------------------------
# Cycle 4 — child runs: one per CV fold, each with auc_roc
# ---------------------------------------------------------------------------


def test_compare_models_child_runs_per_fold(training_df):
    cv = 2
    compare_models(training_df, cv=cv, experiment_name="test-experiment")

    experiment = mlflow.get_experiment_by_name("test-experiment")
    child_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.parentRunId IS NOT NULL",
    )
    # 3 models × cv folds each
    assert len(child_runs) == 3 * cv
    for _, row in child_runs.iterrows():
        assert 0.0 <= row["metrics.auc_roc"] <= 1.0


# ---------------------------------------------------------------------------
# Cycle 5 — main() saves best model to models/best_model.joblib
# ---------------------------------------------------------------------------


def test_main_saves_best_model(tmp_path: Path, monkeypatch):
    import lending.train_models as tm

    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    for year in range(2007, 2013):
        _write_year_csv(data_dir, year, n_rows=20)

    model_path = tmp_path / "models" / "best_model.joblib"
    monkeypatch.setattr(tm.config, "PROCESSED_DIR", data_dir)
    monkeypatch.setattr(tm, "BEST_MODEL_PATH", model_path)

    tm.main(cv=2)

    assert model_path.exists()


# ---------------------------------------------------------------------------
# Cycle 6 — parent run for best model has a SHAP artifact
# ---------------------------------------------------------------------------


def test_best_model_parent_run_has_shap_artifact(training_df):
    _, best_name, _ = compare_models(training_df, cv=2, experiment_name="test-experiment")

    experiment = mlflow.get_experiment_by_name("test-experiment")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{best_name}' AND tags.mlflow.parentRunId IS NULL",
    )
    assert len(runs) == 1
    run_id = runs.iloc[0]["run_id"]
    artifacts = mlflow.MlflowClient().list_artifacts(run_id)
    artifact_names = [a.path for a in artifacts]
    assert any("shap" in name for name in artifact_names)
