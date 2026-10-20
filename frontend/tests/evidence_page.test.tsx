import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import EvidencePage from "../app/evidence/page";
import * as api from "../lib/api";
import pairedStatsJson from "../../reports/paired_stats.json";
import operatingThresholdsJson from "../../reports/operating_thresholds.json";

const mockEvidence = {
  paired_stats: pairedStatsJson,
  metrics_cv: {
    n_splits: 5,
    models: {
      persistence: {
        h30: { rmse_mean: 20.06, rmse_std: 0.89, mae_mean: 12.9, mae_std: 0.44, mard_mean: 9.13, mard_std: 0.38, zone_a_mean: 88.0, zone_a_std: 1.26, zone_ab_mean: 99.58, zone_ab_std: 0.12 },
        h60: { rmse_mean: 31.05, rmse_std: 1.15, mae_mean: 21.0, mae_std: 0.85, mard_mean: 14.8, mard_std: 0.65, zone_a_mean: 72.0, zone_a_std: 2.1, zone_ab_mean: 97.2, zone_ab_std: 0.45 },
        h120: { rmse_mean: 45.1, rmse_std: 1.8, mae_mean: 32.0, mae_std: 1.2, mard_mean: 22.5, mard_std: 0.95, zone_a_mean: 58.0, zone_a_std: 2.8, zone_ab_mean: 91.5, zone_ab_std: 0.8 },
      },
      lightgbm: {
        h30: { rmse_mean: 17.62, rmse_std: 0.65, mae_mean: 10.5, mae_std: 0.35, mard_mean: 7.2, mard_std: 0.28, zone_a_mean: 93.0, zone_a_std: 1.0, zone_ab_mean: 99.9, zone_ab_std: 0.05 },
        h60: { rmse_mean: 26.85, rmse_std: 1.1, mae_mean: 17.8, mae_std: 0.75, mard_mean: 12.1, mard_std: 0.52, zone_a_mean: 80.0, zone_a_std: 1.9, zone_ab_mean: 98.2, zone_ab_std: 0.35 },
        h120: { rmse_mean: 38.2, rmse_std: 1.7, mae_mean: 27.1, mae_std: 1.15, mard_mean: 19.0, mard_std: 0.82, zone_a_mean: 67.0, zone_a_std: 2.5, zone_ab_mean: 93.5, zone_ab_std: 0.65 },
      },
      gru: {
        h30: { rmse_mean: 18.86, rmse_std: 0.75, mae_mean: 11.2, mae_std: 0.4, mard_mean: 7.9, mard_std: 0.3, zone_a_mean: 92.5, zone_a_std: 1.1, zone_ab_mean: 99.8, zone_ab_std: 0.08, raw_coverage_mean: 60.0, calibrated_coverage_mean: 80.4 },
        h60: { rmse_mean: 24.1, rmse_std: 1.05, mae_mean: 16.5, mae_std: 0.7, mard_mean: 11.6, mard_std: 0.5, zone_a_mean: 81.0, zone_a_std: 1.8, zone_ab_mean: 98.5, zone_ab_std: 0.3 },
        h120: { rmse_mean: 32.86, rmse_std: 1.6, mae_mean: 25.8, mae_std: 1.1, mard_mean: 18.2, mard_std: 0.8, zone_a_mean: 68.0, zone_a_std: 2.4, zone_ab_mean: 94.0, zone_ab_std: 0.6 },
      },
    },
    personalization: {
      total_patients_evaluated: 45,
      bias_correction: {
        improved_patients_count: 35,
        improvement_percentage: 77.8,
        mean_rmse_before: 24.14,
        mean_rmse_after: 23.23,
        mean_rmse_gain: 0.91,
        mean_mard_before: 12.82,
        mean_mard_after: 12.46,
        mean_mard_gain: 0.36,
      },
      lora_adapters: {
        improved_patients_count: 39,
        improvement_percentage: 86.7,
        mean_rmse_before: 24.14,
        mean_rmse_after: 23.2,
        mean_rmse_gain: 0.94,
        mean_mard_before: 12.82,
        mean_mard_after: 11.9,
        mean_mard_gain: 0.91,
      },
    },
  },
  operating_thresholds: operatingThresholdsJson,
  whatif_validation: {
    data_agreement_test: {
      total_meals_evaluated: 497,
      cross_fold_summary: {
        pre_specified_pearson_target: 0.3,
      },
      by_diabetes_status: {
        T2D: {
          meals_evaluated: 121,
          mean_predicted_rise_mgdl: 32.67,
          mean_observed_rise_mgdl: 66.82,
          pearson_r: 0.3931,
          positive_effect_share: 0.9835,
        },
      },
    },
    monotonicity_test: {
      raw_model_evaluation: {
        share_monotonic_non_decreasing: 0.725,
        pre_specified_target: 0.9,
      },
      constrained_physiology_projection: {
        share_monotonic_non_decreasing: 1.0,
        share_windows_with_constraint: 0.275,
      },
    },
    exercise_effect_specification: {
      maximum_deduction_ceiling_mgdl: 50.0,
    },
  },
  limitations: [
    "Small research cohort: only 14 type 2 diabetes participants in CGMacros evaluation.",
  ],
  licenses: [
    {
      dataset: "CGMacros",
      license: "CC BY-NC-SA 4.0",
      redistributed: false,
      citation: "Gutierrez-Osuna et al. PhysioNet 2025",
    },
    {
      dataset: "Synthetic AI-READI",
      license: "ADVANCE Center synthetic dataset, terms not published",
      redistributed: false,
      citation: "ADVANCE Center",
    },
  ],
};

describe("EvidencePage Benchmarks and Licensing parity", () => {
  it("renders benchmark numbers exactly matching the evidence reports with standard deviations", async () => {
    vi.spyOn(api, "getEvidenceData").mockResolvedValue(mockEvidence);

    render(<EvidencePage />);

    // Page title is Evidence, and forbidden terms are not present
    const heading = await screen.findByRole("heading", { level: 1 });
    expect(heading.textContent).toContain("Evidence");
    expect(screen.queryByText(/Clinical Evidence/i)).toBeNull();
    expect(screen.queryByText(/Clinical Gating/i)).toBeNull();

    // Accuracy table mean +/- SD across folds
    expect(screen.getByText("20.06 ± 0.89")).toBeDefined();
    expect(screen.getByText("12.90 ± 0.44")).toBeDefined();
    expect(screen.getByText("9.13 ± 0.38%")).toBeDefined();
    expect(screen.getByText("88.00 ± 1.26%")).toBeDefined();
    expect(screen.getByText("99.58 ± 0.12%")).toBeDefined();

    // GRU Primary Serving note derived from numbers
    expect(
      screen.getByText(/LightGBM has the lowest RMSE at \+30m/i)
    ).toBeDefined();

    // Personalization results: Bias correction vs LoRA comparison
    expect(screen.getByText(/Simple Bias Correction \(Offset\)/i)).toBeDefined();
    expect(screen.getByText(/LoRA Adapters \(Rank-4\)/i)).toBeDefined();
    expect(screen.getAllByText(/35 of 45/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/39 of 45/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/achieves 97% of the total RMSE improvement/i)).toBeDefined();

    // What-If T2D empirical meal rise
    expect(screen.getByText("32.67 mg/dL")).toBeDefined();
    expect(screen.getByText("66.82 mg/dL")).toBeDefined();
    expect(screen.getByText(/100\.0% \(passed, by construction\)/i)).toBeDefined();

    // Conformal coverage: nominal 80.0%, conformal 80.4%
    expect(screen.getByText("80.4%")).toBeDefined();

    // Data license: CGMacros CC BY-NC-SA 4.0 and ADVANCE Center synthetic dataset
    expect(screen.getByText("CGMacros")).toBeDefined();
    expect(screen.getByText("ADVANCE Center synthetic dataset, terms not published")).toBeDefined();

    // Check provenance heading doesn't contain "Verified"
    expect(screen.queryByText(/Verified licensing compliance/i)).toBeNull();
  });

  it("renders paired statistics table with 12 rows identical to reports/paired_stats.json", async () => {
    vi.spyOn(api, "getEvidenceData").mockResolvedValue(mockEvidence);

    render(<EvidencePage />);

    expect(await screen.findByText(/Per-Patient Paired Statistics vs Persistence Baseline/i)).toBeDefined();
    expect(pairedStatsJson.results.length).toBe(12);

    for (const row of pairedStatsJson.results) {
      // Mean reduction
      expect(screen.getAllByText(`+${row.mean_difference.toFixed(2)} mg/dL`).length).toBeGreaterThanOrEqual(1);

      // 10,000-draw bootstrap 95% CI
      expect(
        screen.getAllByText(`[${row.ci_lower_95.toFixed(2)}, ${row.ci_upper_95.toFixed(2)}]`).length
      ).toBeGreaterThanOrEqual(1);

      // Improved count out of 45 and percentage
      expect(
        screen.getAllByText(`${row.improved_count} of ${row.total_count} (${row.improved_percentage.toFixed(1)}%)`).length
      ).toBeGreaterThanOrEqual(1);
    }
  });

  it("renders all 12 operating threshold rows with usability badge strictly matching alert_usable in operating_thresholds.json", async () => {
    vi.spyOn(api, "getEvidenceData").mockResolvedValue(mockEvidence);

    render(<EvidencePage />);

    expect(await screen.findByText(/Adverse Glycemic Event Operating Thresholds & Alert Gating/i)).toBeDefined();

    const events = operatingThresholdsJson.events as Record<string, any>;
    const eventKeys = Object.keys(events);
    expect(eventKeys.length).toBe(12);

    // Verify all 12 rows are present
    const usableBadges = screen.getAllByText("Usable in Production");
    const unusableBadges = screen.getAllByText("Unusable (Risk Only)");
    expect(usableBadges.length + unusableBadges.length).toBe(12);

    // Count how many are alert_usable in operating_thresholds.json
    const expectedUsableCount = eventKeys.filter((k) => events[k].alert_usable === true).length;
    const expectedUnusableCount = 12 - expectedUsableCount;

    expect(usableBadges.length).toBe(expectedUsableCount);
    expect(unusableBadges.length).toBe(expectedUnusableCount);
  });

  it("recomputes alert_usable from printed rule for every row and displays computation breakdown", async () => {
    vi.spyOn(api, "getEvidenceData").mockResolvedValue(mockEvidence);

    const { container } = render(<EvidencePage />);

    expect(await screen.findByText(/Adverse Glycemic Event Operating Thresholds/i)).toBeDefined();

    const gate = operatingThresholdsJson.gate_rule;
    const events = operatingThresholdsJson.events as Record<string, any>;

    // For each of the 12 rows, verify recomputation from printed rule inputs
    for (const [key, ev] of Object.entries(events)) {
      const recPass = ev.recall_mean >= gate.min_recall;
      const precPass = ev.precision_mean >= gate.min_precision;
      const relPass = ev.is_reliable === true;
      const faPass = ev.false_alerts_per_patient_day_mean <= gate.false_alert_budget;
      const recomputedUsable = recPass && precPass && relPass && faPass;

      expect(ev.alert_usable).toBe(recomputedUsable);
    }

    // Explicitly verify Severe Hyperglycemia (+60m) and Rapid Spike (+30m) are usable
    expect(events["severe_hyperglycemia_60m"].alert_usable).toBe(true);
    expect(events["spike_30m"].alert_usable).toBe(true);

    // Verify the page renders computation text
    expect(container.textContent).toContain("alert_usable = (Recall");
    expect(container.textContent).toContain("Computation: (");
  });
});

