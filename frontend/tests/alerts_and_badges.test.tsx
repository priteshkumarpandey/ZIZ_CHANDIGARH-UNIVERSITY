import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { AlertsPanel } from "../components/AlertsPanel";
import { getCohortBadgeText, getPatientCohort, ForecastResponse } from "../lib/types";

describe("Alert Gating and Event Onset Policy", () => {
  it("renders actionable alert cards with tuned validation badge ONLY on event onset when current reading is below 180 mg/dL", () => {
    const mockForecast: ForecastResponse = {
      patient_id: "IND_PAT_001",
      forecast_generated_at: new Date().toISOString(),
      current_glucose: 145, // BELOW 180 threshold -> event onset
      model_version: "v1.0-torchscript-gru",
      adverse_event_warning: true,
      summary_message: "Elevated glycemic risk predicted.",
      is_synthetic: true,
      forecast_horizons: [
        {
          horizon_minutes: 30,
          target_timestamp: new Date().toISOString(),
          q10: 130,
          q50: 185,
          q90: 210,
          calibrated_lower_95: 120,
          calibrated_upper_95: 220,
          spike_risk_probability: 0.85,
          hypo_risk_probability: 0.02,
          predicted_event: "hyperglycemia_level_1",
          alert_usable: { hyperglycemia: true },
          operating_metrics: {
            hyperglycemia: {
              recall: 0.84,
              precision: 0.72,
              false_alerts_per_patient_day: 0.6,
            },
          },
        },
        {
          horizon_minutes: 60,
          target_timestamp: new Date().toISOString(),
          q10: 120,
          q50: 140,
          q90: 175,
          calibrated_lower_95: 110,
          calibrated_upper_95: 185,
          spike_risk_probability: 0.65,
          hypo_risk_probability: 0.05,
          predicted_event: "hyperglycemia_level_1",
          alert_usable: { hyperglycemia: false }, // NON-USABLE
        },
      ],
    };

    render(<AlertsPanel forecast={mockForecast} />);

    // Actionable onset card must be present for +30m with expected time format
    expect(screen.getByText(/1 Actionable Alert/i)).toBeTruthy();
    expect(screen.getByText(/Hyperglycemia risk in the next 30 min/i)).toBeTruthy();

    // Click to expand Evidence row
    const evidenceBtn = screen.getByRole("button", { name: /^evidence/i });
    expect(evidenceBtn).toBeTruthy();
    fireEvent.click(evidenceBtn);

    // Held-out recall, precision, and false alerts from API metrics
    expect(screen.getByText(/Held-out Recall:/i)).toBeTruthy();
    expect(screen.getByText(/84%/i)).toBeTruthy();
    expect(screen.getByText(/72%/i)).toBeTruthy();
    expect(screen.getByText(/0.6\/day/i)).toBeTruthy();

    // +60m is non-usable and must NOT produce an alert card; it is in collapsible risk estimates
    const riskBtn = screen.getByText(/Risk estimates only/i);
    expect(riskBtn).toBeTruthy();
    fireEvent.click(riskBtn);
    expect(screen.getByText(/\+60m horizon/i)).toBeTruthy();
    expect(screen.getAllByText(/Below threshold criterion/i).length).toBeGreaterThanOrEqual(1);
  });

  it("never renders alert card when current reading is already >= 180 mg/dL, showing status chip instead", () => {
    const ongoingForecast: ForecastResponse = {
      patient_id: "DEMO_PAT_001",
      forecast_generated_at: new Date().toISOString(),
      current_glucose: 195, // ALREADY ABOVE 180 -> excursion in progress
      model_version: "v1.0-torchscript-gru",
      adverse_event_warning: true,
      summary_message: "Patient in hyperglycemia.",
      is_synthetic: true,
      forecast_horizons: [
        {
          horizon_minutes: 30,
          target_timestamp: new Date().toISOString(),
          q10: 185,
          q50: 210,
          q90: 240,
          calibrated_lower_95: 175,
          calibrated_upper_95: 250,
          spike_risk_probability: 0.9,
          hypo_risk_probability: 0.01,
          predicted_event: "hyperglycemia_level_1",
          alert_usable: { hyperglycemia: true },
        },
      ],
    };

    render(<AlertsPanel forecast={ongoingForecast} />);

    // Status chip must be visible
    expect(screen.getByText(/Currently above 180 mg\/dL/i)).toBeTruthy();
    expect(screen.getByText(/Excursion in Progress \(195 mg\/dL\)/i)).toBeTruthy();

    // No actionable onset alert card should be shown
    expect(screen.queryByText(/1 Actionable Alert/i)).toBeNull();
    expect(screen.queryByText(/Threshold tuned on validation patients/i)).toBeNull();
  });

  it("renders distinct 'Rule-based alert (observed, not predicted)' cards when observed low or rapid fall occurs", () => {
    const lowForecast: ForecastResponse = {
      patient_id: "DEMO_PAT_002",
      forecast_generated_at: new Date().toISOString(),
      current_glucose: 62,
      model_version: "v1.0-torchscript-gru",
      adverse_event_warning: true,
      summary_message: "Observed low glucose detected.",
      is_synthetic: true,
      rule_based_alerts: [
        {
          alert_type: "low",
          severity: "warning",
          title: "Low Glucose (<70 mg/dL)",
          message: "Observed glucose is 62.0 mg/dL, crossing below the 70 mg/dL safety floor.",
          observed_glucose: 62.0,
          threshold_mgdl: 70.0,
          is_rule_based: true,
          provenance: "Rule-based alert (observed, not predicted)",
        },
        {
          alert_type: "rapid_fall",
          severity: "warning",
          title: "Rapid Glucose Fall (>=2 mg/dL/min)",
          message: "Observed glucose dropped by 34.0 mg/dL over 15 minutes (2.3 mg/dL/min) while below 120 mg/dL.",
          observed_glucose: 62.0,
          threshold_mgdl: 120.0,
          rate_mgdl_per_min: 2.27,
          is_rule_based: true,
          provenance: "Rule-based alert (observed, not predicted)",
        },
      ],
      forecast_horizons: [],
    };

    render(<AlertsPanel forecast={lowForecast} currentGlucose={62} />);

    expect(screen.getByText(/Low Glucose \(<70 mg\/dL\)/i)).toBeTruthy();
    expect(screen.getByText(/Rapid Glucose Fall \(>=2 mg\/dL\/min\)/i)).toBeTruthy();
    const badges = screen.getAllByText("Rule-based alert (observed, not predicted)");
    expect(badges.length).toBe(2);
    expect(screen.getByText(/2 Safety Alerts/i)).toBeTruthy();
    expect(screen.getAllByText(/62.0 mg\/dL/i).length).toBeGreaterThanOrEqual(2);
  });
});

describe("Data Badge Attributions", () => {
  it("strictly maps patient cohorts to honest provenance labels", () => {
    expect(getPatientCohort("DEMO_PAT_001")).toBe("synthetic_demo");
    expect(getCohortBadgeText("DEMO_PAT_001")).toBe("Synthetic demo data");

    expect(getPatientCohort("IND_PAT_001")).toBe("synthetic_persona");
    expect(getCohortBadgeText("IND_PAT_001")).toBe("Synthetic persona (synthetic linkage)");

    expect(getPatientCohort("CGMacros-004")).toBe("cgmacros_research");
    expect(getCohortBadgeText("CGMacros-004")).toBe("De-identified research data");
  });
});
