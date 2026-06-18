from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def _retrain() -> None:
    import logging

    import mlflow

    from lending import config
    from lending.data import clean
    from lending.tracking import register_model
    from lending.train import load_processed_years, train

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s"
    )
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    df = clean(load_processed_years(config.TRAIN_YEARS, config.PROCESSED_DIR))
    pipeline, metrics = train(df)
    print(metrics["report"])
    register_model(
        config.MLFLOW_EXPERIMENT_NAME, "logistic_regression", pipeline, metrics["auc_roc"]
    )


with DAG(
    dag_id="lending_retrain",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ml", "lending"],
) as dag:
    PythonOperator(task_id="retrain", python_callable=_retrain)
