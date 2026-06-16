from __future__ import annotations

import logging

import joblib
import mlflow
import numpy as np
import optuna
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from lending import config
from lending.data import clean
from lending.features import build_features
from lending.train import load_processed_years
from lending.tracking import log_run

optuna.logging.set_verbosity(optuna.logging.WARNING)

log = logging.getLogger(__name__)

BEST_MODEL_PATH = config.BEST_MODEL_PATH


def _make_pipeline(model) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def _objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray, cv: int) -> float:
    model_name = trial.suggest_categorical("model", ["random_forest", "xgboost", "lightgbm"])
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 15)

    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            n_jobs=1,
            random_state=config.RANDOM_STATE,
        )
    elif model_name == "xgboost":
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
        )
    else:
        model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            num_leaves=trial.suggest_int("num_leaves", 20, 100),
            n_jobs=1,
            verbose=-1,
            random_state=config.RANDOM_STATE,
        )

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_STATE)
    fold_aucs = [
        roc_auc_score(
            y[val_idx],
            _make_pipeline(model).fit(X[train_idx], y[train_idx]).predict_proba(X[val_idx])[:, 1],
        )
        for train_idx, val_idx in skf.split(X, y)
    ]
    return float(np.mean(fold_aucs))


def _build_model(params: dict) -> object:
    name = params["model"]
    n_estimators = params["n_estimators"]
    max_depth = params["max_depth"]
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=params["min_samples_leaf"],
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        )
    elif name == "xgboost":
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
        )
    else:
        return LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            num_leaves=params["num_leaves"],
            n_jobs=-1,
            verbose=-1,
            random_state=config.RANDOM_STATE,
        )


def tune(
    df: pl.DataFrame,
    n_trials: int = 50,
    cv: int = 3,
    experiment_name: str = config.MLFLOW_EXPERIMENT_NAME,
) -> Pipeline:
    X = build_features(df).to_pandas().to_numpy(dtype=float)
    y = df["target"].to_numpy()

    def _log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        log.info(
            "Trial %d/%d — model: %-14s AUC-ROC: %.4f (best: %.4f)",
            trial.number + 1,
            n_trials,
            trial.params.get("model", "?"),
            trial.value,
            study.best_value,
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: _objective(trial, X, y, cv), n_trials=n_trials, callbacks=[_log_trial])

    best = study.best_trial
    log.info("Best trial — model: %s, AUC-ROC: %.4f", best.params.get("model"), best.value)
    log_run(experiment_name, "optuna_best", {"auc_roc": best.value})

    best_pipeline = _make_pipeline(_build_model(best.params))
    best_pipeline.fit(X, y)
    return best_pipeline


def main(n_trials: int = 50, cv: int = 3) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    df = clean(load_processed_years(config.TRAIN_YEARS, config.PROCESSED_DIR))
    pipeline = tune(df, n_trials=n_trials, cv=cv)

    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, BEST_MODEL_PATH)
    log.info("Best model saved to %s", BEST_MODEL_PATH)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--cv", type=int, default=3)
    args = parser.parse_args()
    main(n_trials=args.n_trials, cv=args.cv)
