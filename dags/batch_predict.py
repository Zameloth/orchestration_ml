from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator


def _batch_predict() -> None:
    import logging

    import mlflow
    import mlflow.sklearn
    import polars as pl

    from lending import config
    from lending.data import clean
    from lending.features import build_features
    from lending.train import load_processed_years

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s"
    )
    log = logging.getLogger(__name__)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    pipeline = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}@champion")
    log.info("Champion model loaded")

    df = clean(load_processed_years(config.EVAL_YEARS, config.PROCESSED_DIR))
    log.info("Eval data loaded: %d rows", len(df))

    X = build_features(df).to_pandas().to_numpy(dtype=float)
    probs = pipeline.predict_proba(X)[:, 1]
    preds = pipeline.predict(X)
    labels = ["charged_off" if p == 1 else "fully_paid" for p in preds]

    db_path = Path(config.DATA_DIR) / "predictions.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                loan_amnt REAL,
                grade TEXT,
                int_rate REAL,
                purpose TEXT,
                default_probability REAL,
                prediction TEXT
            )
            """
        )
        rows = [
            (
                datetime.now(timezone.utc).isoformat(),
                float(row["loan_amnt"]),
                str(row["grade"]),
                float(row["int_rate"]),
                str(row["purpose"]),
                float(prob),
                label,
            )
            for row, prob, label in zip(df.iter_rows(named=True), probs, labels)
        ]
        conn.executemany(
            """
            INSERT INTO predictions
                (created_at, loan_amnt, grade, int_rate, purpose, default_probability, prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    log.info("Saved %d batch predictions to %s", len(rows), db_path)


with DAG(
    dag_id="lending_batch_predict",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ml", "lending"],
) as dag:
    PythonOperator(task_id="batch_predict", python_callable=_batch_predict)
