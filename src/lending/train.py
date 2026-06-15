from pathlib import Path

import joblib
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

TRAIN_YEARS = range(2007, 2013)
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
    y = df["target"].to_numpy()
    X = build_features(df).to_pandas().to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    pipeline = make_pipeline()
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    metrics = {
        "auc_roc": roc_auc_score(y_test, y_prob),
        "report": classification_report(y_test, y_pred, target_names=["Fully Paid", "Charged Off"]),
    }
    return pipeline, metrics


def main() -> None:
    df = clean(load_processed_years(TRAIN_YEARS, config.PROCESSED_DIR))
    pipeline, metrics = train(df)

    print(f"AUC-ROC : {metrics['auc_roc']:.4f}")
    print(metrics["report"])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Modèle sauvegardé : {MODEL_PATH}")


if __name__ == "__main__":
    main()
