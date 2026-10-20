import shutil
import tempfile
import unittest
from pathlib import Path

import pytest

from src.glucotwin.data.pipeline import (
    compute_dataset_summary,
    extract_features,
    generate_splits,
    harmonize_data,
    ingest_raw_data,
)


def _has_raw_data() -> bool:
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        return False
    subdirs = [d for d in raw_dir.iterdir() if d.is_dir() and d.name != ".git"]
    return len(subdirs) > 0 and any(any(d.iterdir()) for d in subdirs)


@pytest.mark.requires_data
class TestPipelineIntegration(unittest.TestCase):
    def setUp(self):
        if not _has_raw_data():
            self.skipTest("Requires local raw clinical dataset in data/raw (missing dataset files)")
        self.temp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pipeline_integration(self):
        interim_dir = self.tmp_path / "interim"
        processed_dir = self.tmp_path / "processed"
        raw_dir = Path("data/raw")

        ingest_stats = ingest_raw_data(
            raw_dir=raw_dir,
            interim_dir=interim_dir,
            include_hupa_t1d=False,
        )
        self.assertGreater(ingest_stats["patients"], 0)
        self.assertTrue(
            (interim_dir / "patients.parquet").exists() or (interim_dir / "patients.csv").exists()
        )

        harm_stats = harmonize_data(
            interim_dir=interim_dir,
            processed_dir=processed_dir,
            harmonization_cfg={
                "resample_interval_minutes": 5,
                "max_interpolation_minutes": 30,
                "glucose_min_mgdl": 40.0,
                "glucose_max_mgdl": 500.0,
                "heart_rate_min": 30.0,
                "heart_rate_max": 220.0,
            },
        )
        self.assertTrue((processed_dir / "data_quality_report.json").exists())
        self.assertGreater(harm_stats["harmonized_rows"], 0)

        dataset = extract_features(
            processed_dir=processed_dir,
            feature_cfg={
                "lookback_minutes": 360,
                "horizons_minutes": [30, 60, 120],
            },
        )
        self.assertGreater(len(dataset), 0)
        self.assertTrue((processed_dir / "features" / "windowed_features.npz").exists())
        self.assertTrue((processed_dir / "features" / "feature_metadata.json").exists())

        summary = compute_dataset_summary(processed_dir=processed_dir, dataset=dataset)
        self.assertIn("cgmacros", summary)
        self.assertIn("ai_readi_synthetic", summary)
        self.assertEqual(summary["ai_readi_synthetic"]["patients_with_at_least_one_window"], 0)
        self.assertGreater(summary["cgmacros"]["patients_with_at_least_one_window"], 0)
        self.assertTrue((processed_dir / "dataset_summary.json").exists())

        splits, manifest = generate_splits(
            processed_dir=processed_dir,
            split_cfg={
                "train_ratio": 0.70,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
            },
            dataset=dataset,
        )
        self.assertGreater(len(splits["train"]), 0)
        self.assertGreater(len(splits["val"]), 0)
        self.assertGreater(len(splits["test"]), 0)
        self.assertTrue((processed_dir / "split_manifest.json").exists())
        self.assertIn("split_breakdown", manifest)


if __name__ == "__main__":
    unittest.main()
