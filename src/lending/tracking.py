from __future__ import annotations

import logging

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.pipeline import Pipeline

from lending import config

log = logging.getLogger(__name__)


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


def register_model(
    experiment_name: str,
    run_name: str,
    pipeline: Pipeline,
    auc_roc: float,
) -> None:
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_metric("auc_roc", auc_roc)
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            skops_trusted_types=[
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
                "lightgbm.sklearn.LGBMClassifier",
                "sklearn.ensemble._forest.RandomForestClassifier",
            ],
        )

    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", config.MODEL_NAME)

    client = mlflow.MlflowClient()
    try:
        champion = client.get_model_version_by_alias(config.MODEL_NAME, "champion")
        if champion.run_id is not None:
            champion_run = client.get_run(champion.run_id)
            champion_auc = float(champion_run.data.metrics.get("auc_roc", -1.0))
        else:
            champion_auc = -1.0
    except mlflow.exceptions.MlflowException:
        champion_auc = -1.0

    if auc_roc >= champion_auc:
        client.set_registered_model_alias(config.MODEL_NAME, "champion", mv.version)
        log.info(
            "New champion: version %s (AUC-ROC: %.4f > %.4f)", mv.version, auc_roc, champion_auc
        )
    else:
        log.info("Model not promoted (AUC-ROC: %.4f <= champion %.4f)", auc_roc, champion_auc)
