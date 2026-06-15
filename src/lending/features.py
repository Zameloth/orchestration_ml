import re

import polars as pl

from lending import config

_EMP_LENGTH_MAP = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10,
}


def _encode_term(series: pl.Series) -> pl.Series:
    return series.str.extract(r"(\d+)").cast(pl.Int16)


def _encode_emp_length(series: pl.Series) -> pl.Series:
    mapping = pl.DataFrame(
        {"emp_length": list(_EMP_LENGTH_MAP.keys()), "_val": list(_EMP_LENGTH_MAP.values())}
    )
    result = (
        series.to_frame("emp_length")
        .join(mapping.lazy().collect(), on="emp_length", how="left")["_val"]
        .cast(pl.Int8)
    )
    return result


def _label_encode(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    for col in cols:
        df = df.with_columns(pl.col(col).cast(pl.Categorical).to_physical().alias(col))
    return df


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    label_cols = [c for c in config.CAT_COLS if c not in ("term", "emp_length")]

    return (
        df.with_columns(
            (pl.col("loan_amnt") / pl.col("annual_inc")).alias("loan_to_income"),
            _encode_term(df["term"]).alias("term"),
            _encode_emp_length(df["emp_length"]).alias("emp_length"),
        )
        .pipe(_label_encode, label_cols)
        .select(config.NUMERIC_COLS + config.CAT_COLS + ["loan_to_income"])
    )
