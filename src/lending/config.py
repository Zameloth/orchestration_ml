from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

RAW_PATH = ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

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
