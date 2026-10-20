from pathlib import Path

import pandas as pd
import polars as pl


def as_pandas_df(df) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    if hasattr(df, "to_pandas"):
        try:
            return df.to_pandas()
        except Exception:
            if hasattr(df, "to_dict"):
                return pd.DataFrame(df.to_dict(as_series=False))
    return pd.DataFrame(df)


def read_parquet(file_path: str | Path):
    path = Path(file_path)
    csv_path = path.with_suffix(".csv")
    if not path.exists() and csv_path.exists():
        return pd.read_csv(str(csv_path))
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if pl is not None:
        return pl.read_parquet(str(path))
    try:
        return pd.read_parquet(str(path))
    except Exception:
        if csv_path.exists():
            return pd.read_csv(str(csv_path))
        raise


def write_parquet(df, file_path: str | Path) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if pl is not None and isinstance(df, pl.DataFrame):
        df.write_parquet(str(path))
    elif isinstance(df, pd.DataFrame):
        try:
            df.to_parquet(str(path), index=False)
        except Exception:
            # Fallback for environments lacking pyarrow/fastparquet
            df.to_csv(str(path.with_suffix(".csv")), index=False)
    else:
        try:
            df.to_parquet(str(path))
        except Exception:
            pass
