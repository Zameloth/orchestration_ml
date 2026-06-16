from pathlib import Path

import polars as pl
import pytest

from lending import config
from lending.data import clean, load_raw, split_by_year


@pytest.mark.skipif(not config.RAW_PATH.exists(), reason="raw data file not available")
def test_load_raw_returns_dataframe_with_loan_status():
    df = load_raw()
    assert isinstance(df, pl.DataFrame)
    assert "loan_status" in df.columns
    assert len(df) > 0


def _mini_df() -> pl.DataFrame:
    return pl.DataFrame({
        "loan_status": ["Fully Paid", "Charged Off", "Current", "Late (31-120 days)"],
        "loan_amnt": [1000.0, 2000.0, 3000.0, 4000.0],
    })


def test_clean_binarizes_target():
    df = clean(_mini_df())
    assert "target" in df.columns
    targets = dict(zip(df["loan_status"].to_list(), df["target"].to_list()))
    assert targets["Fully Paid"] == 0
    assert targets["Charged Off"] == 1


def test_clean_filters_ambiguous_statuses():
    df = clean(_mini_df())
    assert set(df["loan_status"].to_list()) == {"Fully Paid", "Charged Off"}
    assert len(df) == 2


def _annual_df() -> pl.DataFrame:
    return pl.DataFrame({
        "issue_d": ["Jan-2015", "Jun-2015", "Mar-2016", "Dec-2016", "Aug-2017"],
        "loan_amnt": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
    })


def test_split_by_year_creates_one_file_per_year(tmp_path: Path):
    paths = split_by_year(_annual_df(), tmp_path)
    created = {p.stem for p in paths}
    assert created == {"2015", "2016", "2017"}


def test_split_by_year_file_contains_correct_rows(tmp_path: Path):
    split_by_year(_annual_df(), tmp_path)
    df_2016 = pl.read_csv(tmp_path / "2016.csv")
    assert len(df_2016) == 2
    assert set(df_2016["issue_d"].to_list()) == {"Mar-2016", "Dec-2016"}
