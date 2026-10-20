import unittest

import pandas as pd

from src.glucotwin.data.splits import PatientSplitter


class TestSplits(unittest.TestCase):
    def test_stratified_patient_splits_leakage_safety_and_window_filter(self):
        splitter = PatientSplitter(random_seed=42)

        patients = []
        window_counts = {}
        for i in range(100):
            status = "T2D" if i < 40 else ("prediabetes" if i < 70 else "healthy")
            pid = f"PAT_{i:03d}"
            patients.append(
                {
                    "patient_id": pid,
                    "source_dataset": "cgmacros",
                    "diabetes_status": status,
                }
            )
            # 20 patients have zero windows
            window_counts[pid] = 0 if i >= 80 else 100

        df_pat = pd.DataFrame(patients)

        splits, reports = splitter.split_patients_stratified(
            df_pat,
            patient_window_counts=window_counts,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        s_train = set(splits["train"])
        s_val = set(splits["val"])
        s_test = set(splits["test"])

        # Total in splits must be exactly 80 (patients with >=1 window)
        self.assertEqual(len(s_train) + len(s_val) + len(s_test), 80)
        self.assertTrue(s_train.isdisjoint(s_val), "Patient overlap detected between train and val")
        self.assertTrue(
            s_train.isdisjoint(s_test), "Patient overlap detected between train and test"
        )
        self.assertTrue(s_val.isdisjoint(s_test), "Patient overlap detected between val and test")

        # Prove that zero-window patients never appear in split manifests
        for i in range(80, 100):
            pid = f"PAT_{i:03d}"
            self.assertNotIn(pid, s_train)
            self.assertNotIn(pid, s_val)
            self.assertNotIn(pid, s_test)

    def test_strata_representation_in_val_and_test(self):
        splitter = PatientSplitter(random_seed=42)
        patients = []
        window_counts = {}
        # CGMacros-like distribution: 14 T2D, 16 prediabetes, 15 healthy
        for i in range(14):
            pid = f"CG_T2D_{i}"
            patients.append(
                {"patient_id": pid, "source_dataset": "cgmacros", "diabetes_status": "T2D"}
            )
            window_counts[pid] = 2000
        for i in range(16):
            pid = f"CG_PRE_{i}"
            patients.append(
                {"patient_id": pid, "source_dataset": "cgmacros", "diabetes_status": "prediabetes"}
            )
            window_counts[pid] = 2000
        for i in range(15):
            pid = f"CG_HEA_{i}"
            patients.append(
                {"patient_id": pid, "source_dataset": "cgmacros", "diabetes_status": "healthy"}
            )
            window_counts[pid] = 2000

        df_pat = pd.DataFrame(patients)
        splits, reports = splitter.split_patients_stratified(
            df_pat,
            patient_window_counts=window_counts,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        breakdown = splitter.compute_split_breakdown(splits, df_pat, window_counts)
        for split_name in ["train", "val", "test"]:
            self.assertIn("T2D", breakdown[split_name]["by_diabetes_status"])
            self.assertIn("prediabetes", breakdown[split_name]["by_diabetes_status"])
            self.assertIn("healthy", breakdown[split_name]["by_diabetes_status"])
            self.assertGreater(breakdown[split_name]["by_diabetes_status"]["T2D"]["patients"], 0)
            self.assertGreater(
                breakdown[split_name]["by_diabetes_status"]["prediabetes"]["patients"], 0
            )
            self.assertGreater(
                breakdown[split_name]["by_diabetes_status"]["healthy"]["patients"], 0
            )

    def test_cross_dataset_splits(self):
        splitter = PatientSplitter(random_seed=42)
        patients = [
            {"patient_id": f"CG_{i}", "source_dataset": "cgmacros", "diabetes_status": "T2D"}
            for i in range(20)
        ] + [
            {
                "patient_id": f"AI_{i}",
                "source_dataset": "ai_readi_synthetic",
                "diabetes_status": "T2D",
            }
            for i in range(20)
        ]
        df_pat = pd.DataFrame(patients)

        splits = splitter.split_cross_dataset(df_pat, test_source="cgmacros", val_ratio=0.20)

        self.assertEqual(set(splits["test"]), {f"CG_{i}" for i in range(20)})
        self.assertTrue(all(p.startswith("AI_") for p in splits["train"]))
        self.assertTrue(all(p.startswith("AI_") for p in splits["val"]))
        self.assertTrue(set(splits["train"]).isdisjoint(set(splits["val"])))

    def test_leave_k_patients_out_cv(self):
        splitter = PatientSplitter(random_seed=42)
        df_pat = pd.DataFrame([{"patient_id": f"P_{i}"} for i in range(25)])

        folds = list(splitter.leave_k_patients_out_cv(df_pat, n_splits=5))
        self.assertEqual(len(folds), 5)

        seen_val_patients = set()
        for train_pids, val_pids in folds:
            s_tr = set(train_pids)
            s_va = set(val_pids)
            self.assertTrue(s_tr.isdisjoint(s_va))
            self.assertEqual(len(s_tr) + len(s_va), 25)
            seen_val_patients.update(s_va)

        self.assertEqual(len(seen_val_patients), 25)


if __name__ == "__main__":
    unittest.main()
