import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { AlertsPanel } from "../components/AlertsPanel";
import { ForecastResponse } from "../lib/types";

describe("Alert Gating on Real API Data", () => {
  // Exact payload structure and metrics from reports/operating_thresholds.json and backend service
  const realApiForecast: ForecastResponse = {
    patient_id: "DEMO_PAT_001",
    forecast_generated_at: new Date().toISOString(),
    current_glucose: 176.5,
    model_version: "v1.0-torchscript-gru",
    adverse_event_warning: true,
    summary_message: "Onset of postprandial hyperglycemia predicted (+30m).",
    is_synthetic: true,
    serving_fold: 0,
    forecast_horizons: [
      {
        horizon_minutes: 30,
        target_timestamp: new Date(Date.now() + 30 * 60000).toISOString(),
        q10: 172.0,
        q50: 194.5,
        q90: 222.0,
        calibrated_lower_95: 162.0,
        calibrated_upper_95: 232.0,
        spike_risk_probability: 0.88,
        hypo_risk_probability: 0.01,
        predicted_event: "hyperglycemia_level_1",
        alert_usable: {
          hyperglycemia: true,
          severe_hyperglycemia: false,
          hypoglycemia: false,
          spike: true,
        },
        operating_metrics: {
          hyperglycemia: {
            recall: 0.65,
            recall_mean: 0.65,
            recall_std: 0.1,
            precision: 0.61,
            precision_mean: 0.61,
            precision_std: 0.04,
            false_alerts_per_patient_day: 0.81,
            false_alerts_per_patient_day_mean: 0.81,
            false_alerts_per_patient_day_std: 0.2,
          },
        },
      },
      {
        horizon_minutes: 60,
        target_timestamp: new Date(Date.now() + 60 * 60000).toISOString(),
        q10: 182.0,
        q50: 210.0,
        q90: 245.0,
        calibrated_lower_95: 170.0,
        calibrated_upper_95: 256.0,
        spike_risk_probability: 0.94,
        hypo_risk_probability: 0.01,
        predicted_event: "hyperglycemia_level_1",
        alert_usable: {
          hyperglycemia: false, // NOT USABLE per operating_thresholds.json
          severe_hyperglycemia: false,
          hypoglycemia: false,
          spike: false,
        },
        operating_metrics: {
          hyperglycemia: {
            recall: 0.06,
            precision: 0.25,
            false_alerts_per_patient_day: 0.68,
          },
        },
      },
      {
        horizon_minutes: 120,
        target_timestamp: new Date(Date.now() + 120 * 60000).toISOString(),
        q10: 140.0,
        q50: 168.0,
        q90: 200.0,
        calibrated_lower_95: 128.0,
        calibrated_upper_95: 212.0,
        spike_risk_probability: 0.35,
        hypo_risk_probability: 0.05,
        predicted_event: "none",
        alert_usable: {
          hyperglycemia: false,
          severe_hyperglycemia: false,
          hypoglycemia: false,
          spike: false,
        },
        operating_metrics: {
          hyperglycemia: {
            recall: 0.18,
            precision: 0.33,
            false_alerts_per_patient_day: 0.77,
          },
        },
      },
    ],
  };

  it("renders exactly 1 actionable alert card for +30m and suppresses +60m because alert_usable is false", () => {
    render(<AlertsPanel forecast={realApiForecast} />);

    // Exactly 1 Actionable Alert badge
    expect(screen.getByText("1 Actionable Alert")).toBeDefined();

    // +30m alert card is rendered with compact title
    expect(screen.getByText("Hyperglycemia risk in the next 30 min")).toBeDefined();

    // +60m MUST NOT produce an alert card (alert_usable: false)
    expect(screen.queryByText("Hyperglycemia risk in the next 60 min")).toBeNull();

    // Expand Evidence to verify real cross-validated metrics from operating_thresholds.json (recall 65 +- 10%, precision 61 +- 4%, false alerts 0.8 +- 0.2/day)
    const evidenceBtn = screen.getByRole("button", { name: /^evidence/i });
    fireEvent.click(evidenceBtn);

    expect(screen.getByText(/65 ± 10%/)).toBeDefined();
    expect(screen.getByText(/61 ± 4%/)).toBeDefined();
    expect(screen.getByText(/0.8 ± 0.2\/day/)).toBeDefined();
    expect(screen.getByText(/Cross-validated across 5 validation folds \(mean ± SD\)/)).toBeDefined();

    // Non-usable +60m and +120m are collapsed in Risk estimates only
    const riskBtn = screen.getByText(/Risk estimates only/i);
    expect(riskBtn).toBeDefined();
    fireEvent.click(riskBtn);

    expect(screen.getByText("+60m horizon")).toBeDefined();
    expect(screen.getByText("+120m horizon")).toBeDefined();
  });

  it("suppresses all alerts when current glucose already exceeds 180 mg/dL", () => {
    const elevatedForecast = {
      ...realApiForecast,
      current_glucose: 185.0,
    };

    render(<AlertsPanel forecast={elevatedForecast} />);

    expect(screen.getByText("Currently above 180 mg/dL")).toBeDefined();
    expect(screen.queryByText(/1 Actionable Alert/i)).toBeNull();
    expect(screen.queryByText("Hyperglycemia expected in about 30 min")).toBeNull();
  });
});
