from pathlib import Path

import mlflow
import numpy as np
import pytest

from lending.tracking import log_cv_run, log_run


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path: Path, monkeypatch):
    import lending.config as cfg

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    artifact_root = str(tmp_path / "artifacts")
    mlflow.set_tracking_uri(uri)
    monkeypatch.setattr(cfg, "MLFLOW_TRACKING_URI", uri)
    client = mlflow.MlflowClient()
    for name in ("test-baseline", "test-cv"):
        client.create_experiment(name, artifact_location=artifact_root)
    yield
    mlflow.set_tracking_uri(None)


# ---------------------------------------------------------------------------
# Cycle 1 — tracer bullet: log_run creates a run with the given metric
# ---------------------------------------------------------------------------


def test_log_run_creates_run_with_metric():
    log_run("test-baseline", "logistic_regression", {"auc_roc": 0.75})

    experiment = mlflow.get_experiment_by_name("test-baseline")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    assert pytest.approx(runs.iloc[0]["metrics.auc_roc"], abs=1e-6) == 0.75


# ---------------------------------------------------------------------------
# Cycle 2 — log_run uses the given run name
# ---------------------------------------------------------------------------


def test_log_run_uses_run_name():
    log_run("test-baseline", "my_model", {"auc_roc": 0.75})

    experiment = mlflow.get_experiment_by_name("test-baseline")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs.iloc[0]["tags.mlflow.runName"] == "my_model"


# ---------------------------------------------------------------------------
# Cycle 3 — log_cv_run creates a parent run with mean_auc_roc and std_auc_roc
# ---------------------------------------------------------------------------


def test_log_cv_run_parent_run_has_mean_and_std():
    fold_aucs = [0.70, 0.72, 0.68]
    log_cv_run("test-cv", "random_forest", fold_aucs)

    experiment = mlflow.get_experiment_by_name("test-cv")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.parentRunId IS NULL",
    )
    assert len(runs) == 1
    assert pytest.approx(runs.iloc[0]["metrics.mean_auc_roc"], abs=1e-6) == float(np.mean(fold_aucs))
    assert pytest.approx(runs.iloc[0]["metrics.std_auc_roc"], abs=1e-6) == float(np.std(fold_aucs))


# ---------------------------------------------------------------------------
# Cycle 4 — log_cv_run creates one nested child run per fold with auc_roc
# ---------------------------------------------------------------------------


def test_log_cv_run_creates_child_run_per_fold():
    fold_aucs = [0.70, 0.72, 0.68]
    log_cv_run("test-cv", "random_forest", fold_aucs)

    experiment = mlflow.get_experiment_by_name("test-cv")
    child_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.parentRunId IS NOT NULL",
    )
    assert len(child_runs) == len(fold_aucs)
    for _, row in child_runs.iterrows():
        assert 0.0 <= row["metrics.auc_roc"] <= 1.0
