from __future__ import annotations

import logging

import joblib
import mlflow
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from lending import config
from lending.data import clean
from lending.features import build_features
from lending.train import load_processed_years
from lending.tracking import log_cv_run

log = logging.getLogger(__name__)

BEST_MODEL_PATH = config.MODELS_DIR / "best_model.joblib"

MODELS: dict[str, object] = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1),
    "xgboost": XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=config.RANDOM_STATE),
    "lightgbm": LGBMClassifier(n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1),
}

PARAM_GRIDS: dict[str, dict] = {
    "random_forest": {"max_depth": [None, 10]},
    "xgboost": {"max_depth": [3, 6]},
    "lightgbm": {"max_depth": [-1, 6]},
}


def _make_pipeline(model) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def compare_models(
    df: pl.DataFrame,
    cv: int = 5,
    scoring: str = "roc_auc",
    experiment_name: str = config.MLFLOW_EXPERIMENT_NAME,
) -> tuple[Pipeline, str, dict]:
    X = build_features(df).to_pandas().to_numpy(dtype=float)
    y = df["target"].to_numpy()

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_STATE)

    best_mean_auc = -1.0
    best_name = ""
    best_pipeline: Pipeline | None = None
    results: dict = {}

    log.info("Comparing %d models with cv=%d, scoring=%s", len(MODELS), cv, scoring)
    for name, model in MODELS.items():
        log.info("Training %s", name)
        fold_aucs: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            pipeline = _make_pipeline(model)
            pipeline.fit(X_train, y_train)
            auc = roc_auc_score(y_val, pipeline.predict_proba(X_val)[:, 1])
            fold_aucs.append(auc)
            log.debug("  fold %d/%d — AUC-ROC: %.4f", fold_idx + 1, cv, auc)

        mean_auc = float(np.mean(fold_aucs))
        log.info("%s — mean AUC-ROC: %.4f ± %.4f", name, mean_auc, float(np.std(fold_aucs)))

        full_pipeline = _make_pipeline(model)
        full_pipeline.fit(X, y)

        log_cv_run(experiment_name, name, fold_aucs, full_pipeline, X)

        results[name] = {"mean_auc_roc": mean_auc, "fold_aucs": fold_aucs}

        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_name = name
            best_pipeline = full_pipeline

    log.info("Best model: %s (mean AUC-ROC: %.4f)", best_name, best_mean_auc)
    assert best_pipeline is not None
    return best_pipeline, best_name, results


def main(cv: int = 5, scoring: str = "roc_auc") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    log.info("Loading data (%s)", config.PROCESSED_DIR)
    df = clean(load_processed_years(config.TRAIN_YEARS, config.PROCESSED_DIR))
    pipeline, best_name, results = compare_models(df, cv=cv, scoring=scoring)

    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, BEST_MODEL_PATH)
    log.info("Model saved to %s", BEST_MODEL_PATH)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--scoring", type=str, default="roc_auc")
    args = parser.parse_args()
    main(cv=args.cv, scoring=args.scoring)
