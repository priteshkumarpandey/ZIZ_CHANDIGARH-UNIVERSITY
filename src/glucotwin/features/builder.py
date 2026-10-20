import pandas as pd

from src.glucotwin.features.windowing import FeatureWindowBuilder, WindowedDataset


def build_features(
    df_patients: pd.DataFrame,
    df_timeseries: pd.DataFrame,
    df_meals: pd.DataFrame,
    lookback_minutes: int = 360,
    horizons_minutes: list[int] | None = None,
) -> WindowedDataset:
    if horizons_minutes is None:
        horizons_minutes = [30, 60, 120]
    builder = FeatureWindowBuilder(
        lookback_minutes=lookback_minutes,
        horizons_minutes=horizons_minutes,
    )
    return builder.build_dataset(df_patients, df_timeseries, df_meals)
