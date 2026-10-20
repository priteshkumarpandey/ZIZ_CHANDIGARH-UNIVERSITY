from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from src.glucotwin.synth.generator import generate_cohort
from src.glucotwin.utils.io import read_parquet


def load_cohort_dataset(file_path: str | None = None) -> Any:
    if file_path and Path(file_path).exists():
        return read_parquet(file_path)
    ts, st = generate_cohort(days=7)
    if pl is not None and isinstance(ts, pl.DataFrame):
        return ts.join(st, on="patient_id", how="left")
    return pd.DataFrame(ts).merge(pd.DataFrame(st), on="patient_id", how="left")
