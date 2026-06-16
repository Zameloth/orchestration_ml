from __future__ import annotations

import mlflow
import numpy as np


def log_run(experiment_name: str, run_name: str, metrics: dict[str, float]) -> None:
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)


def log_cv_run(experiment_name: str, name: str, fold_aucs: list[float]) -> None:
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=name):
        for fold_idx, auc in enumerate(fold_aucs):
            with mlflow.start_run(run_name=f"fold_{fold_idx + 1}", nested=True):
                mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("mean_auc_roc", float(np.mean(fold_aucs)))
        mlflow.log_metric("std_auc_roc", float(np.std(fold_aucs)))
