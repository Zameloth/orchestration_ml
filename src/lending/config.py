from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

RAW_PATH = ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

NUMERIC_COLS = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
]

CAT_COLS = [
    "term",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
]

RANDOM_STATE = 42
TRAIN_YEARS = range(2007, 2013)
EVAL_YEARS = range(2014, 2015)

MIN_AUC_ROC = 0.65

MLFLOW_TRACKING_URI = f"sqlite:///{ROOT}/mlruns/mlflow.db"
MLFLOW_EXPERIMENT_NAME = "lending-model-comparison"
MODEL_NAME = "lending-classifier"
