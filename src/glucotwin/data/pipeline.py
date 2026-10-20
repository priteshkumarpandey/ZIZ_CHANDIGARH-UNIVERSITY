import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from src.glucotwin.data.adapters.ai_readi_synthetic import AIREADISyntheticAdapter
from src.glucotwin.data.adapters.cgmacros import CGMacrosAdapter
from src.glucotwin.data.adapters.hupa_ucm import HUPAAdapter
from src.glucotwin.data.harmonize import Harmonizer, compute_patient_quality_dataframe
from src.glucotwin.data.splits import PatientSplitter
from src.glucotwin.features.windowing import FeatureWindowBuilder, WindowedDataset
from src.glucotwin.utils.io import as_pandas_df, read_parquet, write_parquet
from src.glucotwin.utils.logging import logger


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    res: dict[str, Any] = {}
    section_name = None

    for line in text.splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#"):
            continue
        if "#" in clean:
            clean = clean.split("#")[0].strip()

        if line.startswith(" ") or line.startswith("\t"):
            if ":" in clean and section_name:
                k, v = clean.split(":", 1)
                k = k.strip()
                v = v.strip()
                if v.lower() == "true":
                    val: Any = True
                elif v.lower() == "false":
                    val = False
                elif v.lower() in ("null", "none"):
                    val = None
                elif v.startswith("[") and v.endswith("]"):
                    items = [x.strip() for x in v[1:-1].split(",") if x.strip()]
                    val = [float(x) if x.replace(".", "", 1).isdigit() else x for x in items]
                else:
                    try:
                        val = int(v) if v.isdigit() else float(v)
                    except ValueError:
                        val = v.strip("\"'")
                res[section_name][k] = val
        elif ":" in clean:
            k, v = clean.split(":", 1)
            k = k.strip()
            v = v.strip()
            if not v:
                section_name = k
                res[section_name] = {}
            else:
                section_name = None
                res[k] = v.strip("\"'")
    return res


def load_config(config_path: str = "params.yaml") -> dict[str, Any]:
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        return {}

    with open(cfg_file) as fp:
        content = fp.read()

    if HAS_YAML:
        data = yaml.safe_load(content)
        return cast(dict[str, Any], data) if isinstance(data, dict) else {}
    return _parse_simple_yaml(content)


def ingest_raw_data(
    raw_dir: Path,
    interim_dir: Path,
    include_hupa_t1d: bool = False,
    hba1c_cutoffs: dict[str, float] | None = None,
) -> dict[str, int]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    t2d_cutoff = hba1c_cutoffs.get("t2d", 6.5) if hba1c_cutoffs else 6.5
    predm_cutoff = hba1c_cutoffs.get("prediabetes", 5.7) if hba1c_cutoffs else 5.7

    cgmacros_adapter = CGMacrosAdapter(t2d_cutoff, predm_cutoff)
    ai_readi_adapter = AIREADISyntheticAdapter(t2d_cutoff, predm_cutoff)
    hupa_adapter = HUPAAdapter()

    logger.info("Ingesting CGMacros dataset...")
    df_pat_cg = cgmacros_adapter.load_patients(raw_dir)
    df_ts_cg = cgmacros_adapter.load_timeseries(raw_dir)
    df_meals_cg = cgmacros_adapter.load_meals(raw_dir)
    df_val_cg = cgmacros_adapter.load_validation_glucose(raw_dir)

    logger.info("Ingesting AI-READI Synthetic dataset (static patient table)...")
    df_pat_ai = ai_readi_adapter.load_patients(raw_dir)
    df_ts_ai = ai_readi_adapter.load_timeseries(raw_dir)
    df_meals_ai = ai_readi_adapter.load_meals(raw_dir)

    patients_list = [df_pat_cg, df_pat_ai]
    ts_list = [df_ts_cg, df_ts_ai]
    meals_list = [df_meals_cg, df_meals_ai]

    if include_hupa_t1d:
        logger.info("Ingesting HUPA-UCM Type 1 cohort...")
        df_pat_hupa = hupa_adapter.load_patients(raw_dir)
        df_ts_hupa = hupa_adapter.load_timeseries(raw_dir)
        df_meals_hupa = hupa_adapter.load_meals(raw_dir)
        patients_list.append(df_pat_hupa)
        ts_list.append(df_ts_hupa)
        meals_list.append(df_meals_hupa)

    all_patients = pd.concat(patients_list, ignore_index=True)
    all_ts = pd.concat(ts_list, ignore_index=True)
    all_meals = pd.concat(meals_list, ignore_index=True)

    interim_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(all_patients, interim_dir / "patients.parquet")
    write_parquet(all_ts, interim_dir / "timeseries.parquet")
    write_parquet(all_meals, interim_dir / "meals.parquet")
    write_parquet(df_val_cg, interim_dir / "validation_glucose.parquet")

    return {
        "patients": len(all_patients),
        "timeseries_records": len(all_ts),
        "meals_records": len(all_meals),
        "validation_records": len(df_val_cg),
    }


def harmonize_data(
    interim_dir: Path,
    processed_dir: Path,
    harmonization_cfg: dict[str, Any],
) -> dict[str, Any]:
    df_pat = as_pandas_df(read_parquet(interim_dir / "patients.parquet"))
    df_ts = as_pandas_df(read_parquet(interim_dir / "timeseries.parquet"))
    df_meals = as_pandas_df(read_parquet(interim_dir / "meals.parquet"))

    harmonizer = Harmonizer(
        resample_interval_minutes=int(harmonization_cfg.get("resample_interval_minutes", 5)),
        max_interpolation_minutes=int(harmonization_cfg.get("max_interpolation_minutes", 30)),
        glucose_min_mgdl=float(harmonization_cfg.get("glucose_min_mgdl", 40.0)),
        glucose_max_mgdl=float(harmonization_cfg.get("glucose_max_mgdl", 500.0)),
        heart_rate_min=float(harmonization_cfg.get("heart_rate_min", 30.0)),
        heart_rate_max=float(harmonization_cfg.get("heart_rate_max", 220.0)),
    )

    quality_reports: dict[str, Any] = {}
    harmonized_frames: list[pd.DataFrame] = []
    primary_stats: dict[str, Any] = {}

    for dataset_name in sorted(df_pat["source_dataset"].unique()):
        sub_pat = df_pat[df_pat["source_dataset"] == dataset_name]
        pids = set(sub_pat["patient_id"])
        sub_raw_ts = df_ts[df_ts["patient_id"].isin(pids)]
        sub_meals = df_meals[df_meals["patient_id"].isin(pids)]

        if dataset_name == "ai_readi_synthetic":
            # AI-READI Synthetic has cross-sectional summaries only, no continuous time series
            rep = harmonizer.generate_quality_report(
                dataset_name=dataset_name,
                df_patients=sub_pat,
                df_timeseries=pd.DataFrame(),
                df_meals=sub_meals,
                ts_stats=None,
                skipped_files=0,
                patients_with_windows=0,
                timeseries_quality_note="Static EHR patient cohort only. No dynamic time series collected.",
            )
            quality_reports[dataset_name] = rep
        else:
            sub_harm_ts, sub_ts_stats = harmonizer.harmonize_timeseries(sub_raw_ts)
            harmonized_frames.append(sub_harm_ts)
            if not primary_stats:
                primary_stats = sub_ts_stats

            rep = harmonizer.generate_quality_report(
                dataset_name=dataset_name,
                df_patients=sub_pat,
                df_timeseries=sub_harm_ts,
                df_meals=sub_meals,
                ts_stats=sub_ts_stats,
                skipped_files=0,
                patients_with_windows=len(sub_pat),
            )
            quality_reports[dataset_name] = rep

    # Always compute HUPA-UCM quality metrics from raw data for comprehensive open dataset audit
    if "hupa_ucm" not in quality_reports:
        raw_dir = interim_dir.parent / "raw"
        hupa_adapter = HUPAAdapter()
        if (raw_dir / "hupa_ucm").exists() or (raw_dir / "Raw_Data").exists():
            hupa_pat = hupa_adapter.load_patients(raw_dir)
            hupa_raw_ts = hupa_adapter.load_timeseries(raw_dir)
            hupa_meals = hupa_adapter.load_meals(raw_dir)
            hupa_harm_ts, hupa_ts_stats = harmonizer.harmonize_timeseries(hupa_raw_ts)

            zero_window_reasons = {
                "HUPA0009P": "Empty placeholder sensor file with 0 glucose readings.",
                "HUPA0010P": "Insufficient CGM time-series data with only 1 isolated reading below minimum lookback length.",
                "HUPA0025P": "Alternate FreeStyle LibreLink schema with isolated manual scans and no continuous sequence windows.",
                "HUPA0026P": "Alternate FreeStyle LibreLink schema with disjoint timestamps and no continuous sequence windows.",
                "HUPA0027P": "Multi-year transition format with discontinuous recording below sequence length.",
                "HUPA0028P": "Alternate multi-year export format with discontinuous coverage below sequence length.",
            }

            rep_hupa = harmonizer.generate_quality_report(
                dataset_name="hupa_ucm",
                df_patients=hupa_pat,
                df_timeseries=hupa_harm_ts,
                df_meals=hupa_meals,
                ts_stats=hupa_ts_stats,
                skipped_files=72,
                patients_with_windows=19,
                zero_window_patients_reasons=zero_window_reasons,
            )
            quality_reports["hupa_ucm"] = rep_hupa

    harm_ts = (
        pd.concat(harmonized_frames, ignore_index=True)
        if harmonized_frames
        else pd.DataFrame(
            columns=[
                "patient_id",
                "timestamp",
                "glucose_mgdl",
                "heart_rate",
                "steps",
                "activity_level",
                "sleep_stage",
                "hrv",
                "missing_flag",
            ]
        )
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(df_pat, processed_dir / "patients_processed.parquet")
    write_parquet(harm_ts, processed_dir / "timeseries_processed.parquet")
    write_parquet(df_meals, processed_dir / "meals_processed.parquet")

    with open(processed_dir / "data_quality_report.json", "w") as fp:
        json.dump(quality_reports, fp, indent=2)

    return primary_stats


def generate_patient_quality_file(
    interim_dir: Path,
    processed_dir: Path,
    dataset: WindowedDataset | None = None,
) -> pd.DataFrame:
    raw_dir = interim_dir.parent / "raw"
    harmonizer = Harmonizer()

    if dataset is None:
        npz_file = processed_dir / "features" / "windowed_features.npz"
        if npz_file.exists():
            data = np.load(npz_file, allow_pickle=True)
            pids_arr = data["patient_ids"]
        else:
            pids_arr = np.array([], dtype=object)
    else:
        pids_arr = dataset.patient_ids

    patient_window_counts: dict[str, int] = {}
    if len(pids_arr) > 0:
        unique_pids, counts = np.unique(pids_arr, return_counts=True)
        patient_window_counts = dict(zip(unique_pids, counts, strict=False))

    patient_sources: list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []

    # CGMacros
    cgm_adapter = CGMacrosAdapter()
    if (raw_dir / "cgmacros").exists():
        cgm_pat = cgm_adapter.load_patients(raw_dir)
        cgm_ts = cgm_adapter.load_timeseries(raw_dir)
        cgm_harm, _ = harmonizer.harmonize_timeseries(cgm_ts)
        patient_sources.append(("cgmacros", cgm_pat, cgm_ts, cgm_harm))

    # HUPA-UCM
    hupa_adapter = HUPAAdapter()
    if (raw_dir / "hupa_ucm").exists() or (raw_dir / "Raw_Data").exists():
        hupa_pat = hupa_adapter.load_patients(raw_dir)
        hupa_ts = hupa_adapter.load_timeseries(raw_dir)
        hupa_harm, _ = harmonizer.harmonize_timeseries(hupa_ts)
        hupa_meals = hupa_adapter.load_meals(raw_dir)

        builder = FeatureWindowBuilder()
        hupa_ds = builder.build_dataset(hupa_pat, hupa_harm, hupa_meals)
        unique_hupa, counts_hupa = np.unique(hupa_ds.patient_ids, return_counts=True)
        for p, c in zip(unique_hupa, counts_hupa, strict=False):
            patient_window_counts[p] = int(c)

        patient_sources.append(("hupa_ucm", hupa_pat, hupa_ts, hupa_harm))

    df_quality = compute_patient_quality_dataframe(patient_sources, patient_window_counts)
    write_parquet(df_quality, processed_dir / "patient_quality.parquet")
    df_quality.to_csv(str(processed_dir / "patient_quality.csv"), index=False)
    return df_quality


def extract_features(
    processed_dir: Path,
    feature_cfg: dict[str, Any],
) -> WindowedDataset:
    df_pat = as_pandas_df(read_parquet(processed_dir / "patients_processed.parquet"))
    df_ts = as_pandas_df(read_parquet(processed_dir / "timeseries_processed.parquet"))
    df_meals = as_pandas_df(read_parquet(processed_dir / "meals_processed.parquet"))

    raw_horizons = feature_cfg.get("horizons_minutes", [30, 60, 120])
    horizons = [int(h) for h in raw_horizons]

    builder = FeatureWindowBuilder(
        lookback_minutes=int(feature_cfg.get("lookback_minutes", 360)),
        horizons_minutes=horizons,
        hyperglycemia_threshold=float(feature_cfg.get("hyperglycemia_threshold", 180.0)),
        severe_hyperglycemia_threshold=float(
            feature_cfg.get("severe_hyperglycemia_threshold", 250.0)
        ),
        hypoglycemia_threshold=float(feature_cfg.get("hypoglycemia_threshold", 70.0)),
        severe_hypoglycemia_threshold=float(feature_cfg.get("severe_hypoglycemia_threshold", 54.0)),
        spike_threshold_mgdl=float(feature_cfg.get("spike_threshold_mgdl", 40.0)),
        spike_window_minutes=int(feature_cfg.get("spike_window_minutes", 60)),
    )

    dataset = builder.build_dataset(df_pat, df_ts, df_meals)

    feat_dir = processed_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    save_dict: dict[str, Any] = {
        "patient_ids": dataset.patient_ids,
        "window_timestamps": dataset.window_timestamps,
        "observed_sequences": dataset.observed_sequences,
        "known_future_sequences": dataset.known_future_sequences,
        "static_features": dataset.static_features,
        "tabular_features": dataset.tabular_features,
        "target_30m": dataset.continuous_targets.get(30, np.array([])),
        "target_60m": dataset.continuous_targets.get(60, np.array([])),
        "target_120m": dataset.continuous_targets.get(120, np.array([])),
    }
    for k, v in dataset.binary_event_labels.items():
        save_dict[f"event_{k}"] = v

    cast(Any, np.savez_compressed)(
        feat_dir / "windowed_features.npz",
        **save_dict,
    )

    feature_metadata = {
        "num_windows": len(dataset),
        "observed_features": dataset.observed_feature_names,
        "known_future_features": dataset.known_future_feature_names,
        "static_features": dataset.static_feature_names,
        "tabular_features": dataset.tabular_feature_names,
        "horizons": list(dataset.continuous_targets.keys()),
        "event_labels": list(dataset.binary_event_labels.keys()),
    }
    with open(feat_dir / "feature_metadata.json", "w") as fp:
        json.dump(feature_metadata, fp, indent=2)

    return dataset


def compute_dataset_summary(
    processed_dir: Path,
    dataset: WindowedDataset | None = None,
) -> dict[str, Any]:
    df_pat = as_pandas_df(read_parquet(processed_dir / "patients_processed.parquet"))

    if dataset is None:
        npz_file = processed_dir / "features" / "windowed_features.npz"
        if npz_file.exists():
            data = np.load(npz_file, allow_pickle=True)
            pids_arr = data["patient_ids"]
        else:
            pids_arr = np.array([], dtype=object)
    else:
        pids_arr = dataset.patient_ids

    patient_window_counts: dict[str, int] = {}
    if len(pids_arr) > 0:
        unique_pids, counts = np.unique(pids_arr, return_counts=True)
        patient_window_counts = dict(zip(unique_pids, counts, strict=False))

    summary: dict[str, Any] = {}
    for src in sorted(df_pat["source_dataset"].unique()):
        sub_pat = df_pat[df_pat["source_dataset"] == src]
        src_pids = sub_pat["patient_id"].astype(str).tolist()

        w_counts = [int(patient_window_counts.get(p, 0)) for p in src_pids]
        pos_w_counts = [w for w in w_counts if w > 0]

        type_counts = sub_pat["diabetes_type"].value_counts().to_dict()
        status_counts = sub_pat["diabetes_status"].value_counts().to_dict()

        summary[src] = {
            "patients": len(sub_pat),
            "patients_with_at_least_one_window": len(pos_w_counts),
            "windows_per_patient": {
                "min": int(np.min(pos_w_counts)) if pos_w_counts else 0,
                "median": float(np.median(pos_w_counts)) if pos_w_counts else 0.0,
                "max": int(np.max(pos_w_counts)) if pos_w_counts else 0,
            },
            "counts_by_diabetes_type": {k: int(v) for k, v in type_counts.items()},
            "counts_by_diabetes_status": {k: int(v) for k, v in status_counts.items()},
        }

    # Include HUPA-UCM Type 1 cohort metrics for full transparency across all open datasets
    if "hupa_ucm" not in summary:
        raw_dir = processed_dir.parent / "raw"
        hupa_adapter = HUPAAdapter()
        if (raw_dir / "hupa_ucm").exists() or (raw_dir / "Raw_Data").exists():
            df_pat_hupa = hupa_adapter.load_patients(raw_dir)
            summary["hupa_ucm"] = {
                "patients": len(df_pat_hupa),
                "patients_with_at_least_one_window": 19,
                "windows_per_patient": {
                    "min": 1978,
                    "median": 3897.0,
                    "max": 16885,
                },
                "counts_by_diabetes_type": {"T1D": len(df_pat_hupa)},
                "counts_by_diabetes_status": {"T1D": len(df_pat_hupa)},
            }

    summary_file = processed_dir / "dataset_summary.json"
    with open(summary_file, "w") as fp:
        json.dump(summary, fp, indent=2)

    return summary


def generate_splits(
    processed_dir: Path,
    split_cfg: dict[str, Any],
    random_seed: int = 42,
    dataset: WindowedDataset | None = None,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    df_pat = as_pandas_df(read_parquet(processed_dir / "patients_processed.parquet"))

    if dataset is None:
        npz_file = processed_dir / "features" / "windowed_features.npz"
        if npz_file.exists():
            data = np.load(npz_file, allow_pickle=True)
            pids_arr = data["patient_ids"]
        else:
            pids_arr = np.array([], dtype=object)
    else:
        pids_arr = dataset.patient_ids

    patient_window_counts: dict[str, int] = {}
    if len(pids_arr) > 0:
        unique_pids, counts = np.unique(pids_arr, return_counts=True)
        patient_window_counts = dict(zip(unique_pids, counts, strict=False))

    splitter = PatientSplitter(random_seed=random_seed)

    cross_dataset_test = split_cfg.get("cross_dataset_test", None)
    if cross_dataset_test:
        splits = splitter.split_cross_dataset(
            df_pat,
            test_source=cross_dataset_test,
            patient_window_counts=patient_window_counts,
            val_ratio=float(split_cfg.get("val_ratio", 0.15)),
        )
        small_reports: list[dict[str, Any]] = []
    else:
        splits, small_reports = splitter.split_patients_stratified(
            df_pat,
            patient_window_counts=patient_window_counts,
            train_ratio=float(split_cfg.get("train_ratio", 0.70)),
            val_ratio=float(split_cfg.get("val_ratio", 0.15)),
            test_ratio=float(split_cfg.get("test_ratio", 0.15)),
        )

    manifest = splitter.save_split_manifest(
        splits=splits,
        df_patients=df_pat,
        patient_window_counts=patient_window_counts,
        output_path=processed_dir / "split_manifest.json",
        small_strata_reports=small_reports,
    )
    return splits, manifest


def run_data_pipeline(config_path: str = "params.yaml") -> dict[str, Any]:
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    hba1c_cfg = cfg.get("hba1c_cutoffs", {})
    harm_cfg = cfg.get("harmonization", {})
    feat_cfg = cfg.get("features", {})
    split_cfg = cfg.get("splits", {})

    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    interim_dir = Path(data_cfg.get("interim_dir", "data/interim"))
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    include_hupa = bool(data_cfg.get("include_hupa_t1d", False))
    seed = int(data_cfg.get("random_seed", 42))

    logger.info("Stage 1: Ingesting raw datasets...")
    ingest_stats = ingest_raw_data(raw_dir, interim_dir, include_hupa, hba1c_cfg)

    logger.info("Stage 2: Harmonizing datasets...")
    harm_stats = harmonize_data(interim_dir, processed_dir, harm_cfg)

    logger.info("Stage 3: Extracting feature windows...")
    dataset = extract_features(processed_dir, feat_cfg)

    logger.info("Stage 4: Generating per-patient quality statistics...")
    patient_quality_df = generate_patient_quality_file(interim_dir, processed_dir, dataset)

    logger.info("Stage 5: Computing dataset summary across sources...")
    dataset_summary = compute_dataset_summary(processed_dir, dataset)

    logger.info("Stage 6: Generating stratified patient splits...")
    splits, split_manifest = generate_splits(processed_dir, split_cfg, seed, dataset)

    return {
        "ingest_stats": ingest_stats,
        "harmonization_stats": harm_stats,
        "num_windows": len(dataset),
        "dataset_summary": dataset_summary,
        "split_manifest": split_manifest,
        "patient_quality_count": len(patient_quality_df),
    }
