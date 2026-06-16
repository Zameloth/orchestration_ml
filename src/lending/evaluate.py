from __future__ import annotations

import logging

import joblib
import mlflow
import polars as pl
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from lending import config
from lending.data import clean
from lending.features import build_features
from lending.train import load_processed_years
from lending.tracking import log_run

log = logging.getLogger(__name__)

MODEL_PATH = config.MODELS_DIR / "best_model.joblib"


def evaluate(
    pipeline: Pipeline,
    df: pl.DataFrame,
    min_auc: float = config.MIN_AUC_ROC,
    experiment_name: str = config.MLFLOW_EXPERIMENT_NAME,
) -> dict[str, float]:
    X = build_features(df).to_pandas().to_numpy(dtype=float)
    y = df["target"].to_numpy()

    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = pipeline.predict(X)

    metrics = {
        "auc_roc": float(roc_auc_score(y, y_prob)),
        "f1_charged_off": float(f1_score(y, y_pred)),
    }

    log.info(
        "AUC-ROC: %.4f | F1 (Charged Off): %.4f", metrics["auc_roc"], metrics["f1_charged_off"]
    )
    log_run(experiment_name, "evaluate", metrics)

    if metrics["auc_roc"] < min_auc:
        raise ValueError(
            f"AUC-ROC {metrics['auc_roc']:.4f} is below the minimum threshold {min_auc}"
        )

    return metrics


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s"
    )
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    pipeline = joblib.load(MODEL_PATH)
    log.info("Loaded model from %s", MODEL_PATH)

    df = clean(load_processed_years(config.EVAL_YEARS, config.PROCESSED_DIR))
    metrics = evaluate(pipeline, df, min_auc=config.MIN_AUC_ROC)

    print(f"AUC-ROC:         {metrics['auc_roc']:.4f}")
    print(f"F1 (Charged Off): {metrics['f1_charged_off']:.4f}")


if __name__ == "__main__":
    main()
