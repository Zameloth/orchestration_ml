from pathlib import Path

import polars as pl

from lending import config


def load_raw(path: Path | str | None = None) -> pl.DataFrame:
    p = Path(path) if path else config.RAW_PATH
    return pl.read_csv(p, infer_schema_length=10000, ignore_errors=True)


def clean(df: pl.DataFrame) -> pl.DataFrame:
    keep = {"Fully Paid", "Charged Off"}
    return df.filter(pl.col("loan_status").is_in(keep)).with_columns(
        pl.when(pl.col("loan_status") == "Charged Off")
        .then(1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias("target")
    )


def split_by_year(df: pl.DataFrame, output_dir: Path | str) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = df.with_columns(pl.col("issue_d").str.slice(-4).cast(pl.Int16).alias("_year"))
    paths = []
    for year, group in df.filter(pl.col("_year").is_not_null()).group_by("_year"):
        p = out / f"{year[0]}.csv"
        group.drop("_year").write_csv(p)
        paths.append(p)
    return paths
