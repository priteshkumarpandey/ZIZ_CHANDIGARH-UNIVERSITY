import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, act } from "@testing-library/react";
import "@testing-library/jest-dom";
import TwinDetailView from "../app/twin/[id]/page";
import * as api from "../lib/api";
import { useGlucoTwinStore } from "../lib/store";

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "IND_PAT_001" }),
}));

// Mock recharts to avoid layout engine errors in test environment
vi.mock("recharts", () => ({
  ResponsiveContainer: ({ children }: any) => <div>{children}</div>,
  ComposedChart: ({ children }: any) => <div>{children}</div>,
  Line: () => <div />,
  Area: () => <div />,
  XAxis: () => <div />,
  YAxis: () => <div />,
  Tooltip: () => <div />,
  ReferenceLine: () => <div />,
  ReferenceArea: () => <div />,
  CartesianGrid: () => <div />,
}));

const mockProfile = {
  id: "IND_PAT_001",
  name: "Rajesh Sharma",
  age: 52,
  gender: "male",
  region: "North India",
  hba1c: 8.4,
  bmi: 27.8,
  fasting_glucose: 142.0,
  dietary_habit: "Vegetarian",
  occupation: "Civil Engineer",
  active_medications: ["Metformin 1000mg BID"],
  phenotype_notes: "Pronounced dawn phenomenon.",
  is_synthetic: true,
};

const mockSummary = {
  patient_id: "IND_PAT_001",
  monitoring_window_text: "last 24 hours",
  metrics: {
    mean_glucose: 140.0,
    gmi: 6.8,
    time_in_range_percentage: 75.0,
    time_above_range_percentage: 20.0,
    time_below_range_percentage: 5.0,
    time_above_range_level2_percentage: 5.0,
    time_below_range_level2_percentage: 1.0,
    coefficient_of_variation: 28.0,
  },
  clinical_takeaways: ["Glycemic control within target range."],
  doctor_notes_draft: "Stable trajectory.",
  risk_assessment: "Moderate",
  disclaimer: "For clinical review only.",
};

const makeMockCgm = (count: number) => {
  const baseTime = new Date("2026-03-30T08:00:00Z").getTime();
  return Array.from({ length: count }, (_, i) => ({
    timestamp: new Date(baseTime + i * 5 * 60 * 1000).toISOString(),
    glucose_mgdl: 120 + (i % 20) * 2,
    trend_arrow: "Flat",
  }));
};

describe("Why Panel Race Condition Elimination", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useGlucoTwinStore.setState({
      selectedPatient: null,
      twinState: null,
      cgmHistory: [],
      latestForecast: null,
      explainability: null,
      scrubberIndex: 0,
      maxScrubberIndex: 0,
    });
  });

  it("handles out-of-order explainability responses without stale overwrites", async () => {
    const cgmPoints = makeMockCgm(100);
    const mountTimestamp = cgmPoints[99].timestamp;
    const step1Timestamp = cgmPoints[72].timestamp;
    const step2Timestamp = cgmPoints[73].timestamp;

    vi.spyOn(api, "getPatientProfile").mockResolvedValue(mockProfile as any);
    vi.spyOn(api, "getTwinState").mockResolvedValue({
      patient_id: "IND_PAT_001",
      current_glucose: 150.0,
      trend_arrow: "Flat",
      last_updated: mountTimestamp,
      is_realtime: true,
      model_fold: 0,
    } as any);
    vi.spyOn(api, "getPatientCGM").mockResolvedValue(cgmPoints as any);
    vi.spyOn(api, "getPersonalizationStatus").mockResolvedValue(null as any);
    vi.spyOn(api, "getClinicalSummary").mockResolvedValue(mockSummary as any);

    vi.spyOn(api, "predictGlucose").mockResolvedValue({
      forecast_horizons: [
        {
          horizon_minutes: 30,
          q10: 140,
          q50: 155,
          q90: 170,
          calibrated_lower_95: 135,
          calibrated_upper_95: 175,
          spike_risk_probability: 0.1,
          hypo_risk_probability: 0.05,
          hyper_risk_probability: 0.15,
          severe_hyper_risk_probability: 0.02,
          predicted_event: "none",
          operating_thresholds: {},
          event_reliabilities: {},
          reliability_notes: {},
          alert_usable: {},
          risk_labels: {},
          operating_metrics: {},
        },
      ],
      serving_fold: 0,
    } as any);

    // Track resolvers to simulate out-of-order delivery
    let resolveStep1Explain: ((val: any) => void) | null = null;
    let resolveStep2Explain: ((val: any) => void) | null = null;

    vi.spyOn(api, "getExplainability").mockImplementation((_pid, _h, asOf, _signal) => {
      if (asOf === mountTimestamp) {
        // Initial mount request: resolves immediately
        return Promise.resolve({
          patient_id: "IND_PAT_001",
          horizon_minutes: 60,
          model_name: "Multimodal GRU",
          top_features: [
            {
              feature_name: "glucose_mgdl",
              shap_value: 0.35,
              base_value: 120,
              user_value: cgmPoints[99].glucose_mgdl,
              description: "Initial Mount Glucose Feature",
              model: "Multimodal GRU",
            },
          ],
          as_of: mountTimestamp,
          current_glucose: cgmPoints[99].glucose_mgdl,
        } as any);
      } else if (asOf === step1Timestamp) {
        // Step 1 request (triggered first, will resolve SECOND)
        return new Promise((resolve) => {
          resolveStep1Explain = () =>
            resolve({
              patient_id: "IND_PAT_001",
              horizon_minutes: 60,
              model_name: "Multimodal GRU",
              top_features: [
                {
                  feature_name: "glucose_mgdl",
                  shap_value: 0.2,
                  base_value: 120,
                  user_value: 111.1,
                  description: "Old Step 1 Feature",
                  model: "Multimodal GRU",
                },
              ],
              as_of: step1Timestamp,
              current_glucose: 111.1,
            } as any);
        });
      } else if (asOf === step2Timestamp) {
        // Step 2 request (triggered second, will resolve FIRST)
        return new Promise((resolve) => {
          resolveStep2Explain = () =>
            resolve({
              patient_id: "IND_PAT_001",
              horizon_minutes: 60,
              model_name: "Multimodal GRU",
              top_features: [
                {
                  feature_name: "glucose_mgdl",
                  shap_value: 0.5,
                  base_value: 120,
                  user_value: 222.2,
                  description: "Latest Step 2 Feature",
                  model: "Multimodal GRU",
                },
              ],
              as_of: step2Timestamp,
              current_glucose: 222.2,
            } as any);
        });
      }
      return Promise.resolve(null as any);
    });

    render(<TwinDetailView />);

    await waitFor(() => {
      expect(screen.getByText("Rajesh Sharma")).toBeTruthy();
    });

    const stepReplay = (window as any).__stepReplay;
    expect(stepReplay).toBeDefined();

    // Trigger step 1 (idx = 1) -> windowSlice ends at cgmPoints[72]
    await act(async () => {
      stepReplay(1);
    });

    // Immediately trigger step 2 (idx = 2) -> windowSlice ends at cgmPoints[73]
    await act(async () => {
      stepReplay(2);
    });

    // Step 2 resolves FIRST
    await act(async () => {
      if (resolveStep2Explain) resolveStep2Explain(null);
    });

    // Verify Why panel displays latest step feature and synchronized hero glucose
    await waitFor(
      () => {
        expect(screen.getByText("Latest Step 2 Feature")).toBeTruthy();
        const heroGlucose = screen.getByTestId("hero-current-glucose");
        const whyGlucose = screen.getByTestId("why-panel-current-glucose");
        expect(heroGlucose.textContent).toContain("146.0");
        expect(whyGlucose.textContent).toContain("146");
      },
      { timeout: 4000 }
    );

    // Out-of-order arrival: Step 1 resolves AFTER step 2
    await act(async () => {
      if (resolveStep1Explain) resolveStep1Explain(null);
    });

    // The stale step 1 response MUST NOT overwrite the latest state
    expect(screen.getByText("Latest Step 2 Feature")).toBeTruthy();
    expect(screen.queryByText("Old Step 1 Feature")).toBeNull();
    const heroGlucose = screen.getByTestId("hero-current-glucose");
    const whyGlucose = screen.getByTestId("why-panel-current-glucose");
    expect(heroGlucose.textContent).toContain("146.0");
    expect(whyGlucose.textContent).toContain("146");
  });
});
