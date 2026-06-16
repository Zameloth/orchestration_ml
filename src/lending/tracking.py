from __future__ import annotations

import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import shap
from sklearn.pipeline import Pipeline


def log_run(experiment_name: str, run_name: str, metrics: dict[str, float]) -> None:
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)


def log_cv_run(
    experiment_name: str,
    name: str,
    fold_aucs: list[float],
    pipeline: Pipeline,
    X: np.ndarray,
) -> None:
    mlflow.set_experiment(experiment_name)
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    with mlflow.start_run(run_name=name):
        for fold_idx, auc in enumerate(fold_aucs):
            with mlflow.start_run(run_name=f"fold_{fold_idx + 1}", nested=True):
                mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("mean_auc_roc", mean_auc)
        mlflow.log_metric("std_auc_roc", std_auc)
        _log_shap(pipeline, X)


def _log_shap(pipeline: Pipeline, X: np.ndarray) -> None:
    X_transformed = pipeline[:-1].transform(X)
    model = pipeline[-1]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_transformed)
    if isinstance(sv, list):
        sv = sv[1]
    shap.summary_plot(sv, X_transformed, show=False)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        plt.savefig(f.name, bbox_inches="tight")
        mlflow.log_artifact(f.name, artifact_path="shap")
    plt.close()
