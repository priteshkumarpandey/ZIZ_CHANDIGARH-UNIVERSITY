import polars as pl

from src.glucotwin.data.loaders import load_cohort_dataset


def run_data_pipeline(data_path: str | None = None) -> pl.DataFrame:
    df = load_cohort_dataset(data_path)
    if isinstance(df, pl.DataFrame):
        return df
    return pl.DataFrame(df)
