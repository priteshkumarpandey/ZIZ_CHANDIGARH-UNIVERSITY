import tempfile
from pathlib import Path

import pandas as pd

from src.glucotwin.utils.io import read_parquet, write_parquet
from src.glucotwin.utils.logging import logger
from src.glucotwin.utils.seeds import set_global_seed


def test_seeds_and_logging():
    set_global_seed(42)
    logger.info("Test log message")
    assert True


def test_io_parquet():
    df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / "test.parquet"
        write_parquet(df, fpath)
        loaded = read_parquet(fpath)
        assert len(loaded) == 2
