from pathlib import Path

import numpy as np

from src.glucotwin.serving.service import InferenceService
from src.glucotwin.utils.io import as_pandas_df, read_parquet


def test_explain_flat_series_returns_variation_note():
    service = InferenceService()
    service.load()
    flat_series = [120.0] * 72
    result = service.explain(
        patient_id="IND_PAT_001",
        horizon=60,
        recent_cgm_series=flat_series,
    )
    assert result.variation_note == "little variation in this window"
    assert len(result.top_features) == 0


def test_explain_varying_series_computes_features():
    service = InferenceService()
    service.load()
    # Sinusoidal varying series with substantial variation
    time_points = np.linspace(0, 4 * np.pi, 72)
    varying_series = [float(140.0 + 35.0 * np.sin(t)) for t in time_points]
    result = service.explain(
        patient_id="IND_PAT_001",
        horizon=60,
        recent_cgm_series=varying_series,
    )
    assert result.variation_note is None
    assert len(result.top_features) > 0
    # Current glucose should match the last reading of the series
    current_glucose_feat = next(
        (f for f in result.top_features if f.feature_name == "glucose_mgdl"), None
    )
    if current_glucose_feat:
        assert abs(current_glucose_feat.user_value - varying_series[-1]) < 0.5


def test_explain_as_of_matches_cgm_reading_at_cutoff():
    service = InferenceService()
    service.load()

    demo_path = Path("data/demo/timeseries_demo.parquet")
    if not demo_path.exists():
        return

    df_ts = as_pandas_df(read_parquet(demo_path))
    rows = df_ts[df_ts["patient_id"] == "DEMO_PAT_001"].sort_values("timestamp")
    assert len(rows) >= 80

    target_row = rows.iloc[75]
    as_of_ts = target_row["timestamp"]
    if hasattr(as_of_ts, "isoformat"):
        as_of_str = as_of_ts.isoformat()
    else:
        as_of_str = str(as_of_ts)
    expected_glucose = float(target_row["glucose"])

    result = service.explain(
        patient_id="DEMO_PAT_001",
        horizon=60,
        as_of=as_of_str,
    )

    glucose_feat = next((f for f in result.top_features if f.feature_name == "glucose_mgdl"), None)
    assert glucose_feat is not None
    assert abs(glucose_feat.user_value - expected_glucose) < 0.1
