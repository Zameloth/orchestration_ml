from __future__ import annotations

import logging
from pathlib import Path

import joblib
import mlflow
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lending import config
from lending.data import clean
from lending.features import build_features
from lending.tracking import log_run

log = logging.getLogger(__name__)

EXPERIMENT_NAME = config.MLFLOW_EXPERIMENT_NAME
MODEL_PATH = config.ROOT / "data" / "models" / "baseline.joblib"


def load_processed_years(years: range, data_dir: Path) -> pl.DataFrame:
    frames = []
    for year in years:
        p = data_dir / f"{year}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing processed file: {p}")
        frames.append(pl.read_csv(p, infer_schema_length=10000, ignore_errors=True))
    return pl.concat(frames, how="diagonal_relaxed")


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train(df: pl.DataFrame, test_size: float = 0.2) -> tuple[Pipeline, dict]:
    log.info("Training logistic regression on %d samples", len(df))
    y = df["target"].to_numpy()
    X = build_features(df).to_pandas().to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=config.RANDOM_STATE, stratify=y
    )

    pipeline = make_pipeline()
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    metrics = {
        "auc_roc": roc_auc_score(y_test, y_prob),
        "report": classification_report(y_test, y_pred, target_names=["Fully Paid", "Charged Off"], zero_division=0),
    }
    log.info("AUC-ROC: %.4f", metrics["auc_roc"])

    log_run(EXPERIMENT_NAME, "logistic_regression", {"auc_roc": metrics["auc_roc"]})

    return pipeline, metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    log.info("Loading data (%s)", config.PROCESSED_DIR)
    df = clean(load_processed_years(config.TRAIN_YEARS, config.PROCESSED_DIR))
    pipeline, metrics = train(df)

    print(metrics["report"])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    log.info("Model saved to %s", MODEL_PATH)


if __name__ == "__main__":
    main()
