import polars as pl
import pytest

from lending.config import CAT_COLS, NUMERIC_COLS
from lending.features import build_features


def _base_df() -> pl.DataFrame:
    return pl.DataFrame({
        "loan_amnt": [10000.0, 20000.0],
        "int_rate": [10.5, 14.0],
        "installment": [200.0, 400.0],
        "annual_inc": [50000.0, 80000.0],
        "dti": [15.0, 20.0],
        "delinq_2yrs": [0.0, 1.0],
        "fico_range_low": [700.0, 680.0],
        "fico_range_high": [704.0, 684.0],
        "inq_last_6mths": [1.0, 2.0],
        "open_acc": [10.0, 8.0],
        "pub_rec": [0.0, 0.0],
        "revol_bal": [5000.0, 8000.0],
        "revol_util": [30.0, 55.0],
        "total_acc": [20.0, 15.0],
        "term": [" 36 months", " 60 months"],
        "grade": ["A", "C"],
        "sub_grade": ["A1", "C3"],
        "emp_length": ["10+ years", "2 years"],
        "home_ownership": ["MORTGAGE", "RENT"],
        "verification_status": ["Not Verified", "Verified"],
        "purpose": ["debt_consolidation", "small_business"],
        "target": [0, 1],
        "extra_col": ["drop_me", "drop_me"],
    })


def test_build_features_selects_correct_columns():
    df = build_features(_base_df())
    expected = set(NUMERIC_COLS) | set(CAT_COLS) | {"loan_to_income"}
    assert set(df.columns) == expected


def test_build_features_loan_to_income():
    df = build_features(_base_df())
    expected = [10000.0 / 50000.0, 20000.0 / 80000.0]
    assert df["loan_to_income"].to_list() == pytest.approx(expected)


def test_build_features_encodes_term_as_int():
    df = build_features(_base_df())
    assert df["term"].dtype == pl.Int16
    assert df["term"].to_list() == [36, 60]


def test_build_features_encodes_emp_length_as_int():
    df = build_features(_base_df())
    assert df["emp_length"].dtype == pl.Int8
    assert df["emp_length"].to_list() == [10, 2]


def test_build_features_encodes_remaining_cats_as_int():
    df = build_features(_base_df())
    for col in ["grade", "sub_grade", "home_ownership", "verification_status", "purpose"]:
        assert df[col].dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64), \
            f"{col} should be integer, got {df[col].dtype}"
