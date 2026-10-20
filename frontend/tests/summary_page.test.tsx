import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import ClinicalSummaryPage from "../app/summary/[id]/page";
import * as api from "../lib/api";

const mockProfile = {
  id: "IND_PAT_001",
  name: "Rajesh Sharma",
  age: 52,
  gender: "male",
  region: "North India",
  occupation: "Civil Engineer",
  dietary_habit: "Vegetarian",
  postprandial_glucose: 188.0,
  ethnicity: "South Asian",
  hba1c: 8.4,
  bmi: 27.8,
  fasting_glucose: 142.0,
  active_medications: ["Metformin 1000mg BID"],
  phenotype_notes: "Pronounced dawn phenomenon.",
  is_synthetic: true,
};

const mockSummary = {
  patient_id: "IND_PAT_001",
  patient_name: "Rajesh Sharma",
  generated_at: "2026-10-15T12:00:00Z",
  monitoring_window_text: "last 24 hours",
  metrics: {
    time_in_range_percentage: 68.5,
    time_above_range_percentage: 27.5,
    time_above_range_level2_percentage: 8.2,
    time_below_range_percentage: 4.0,
    time_below_range_level2_percentage: 1.0,
    gmi: 7.4,
    mean_glucose: 154.2,
    standard_deviation: 38.6,
    coefficient_of_variation: 25.0,
  },
  adverse_event_count_24h: {},
  key_findings: ["High nocturnal variability."],
  actionable_recommendations: ["Maintain consistent meal timing."],
  plain_language_narrative: "Patient shows consistent post-meal excursions.",
  is_llm_enhanced: false,
  disclaimer: "Decision support only. Not intended as medical diagnosis or direct prescription.",
  is_synthetic: false,
};

describe("Clinical Summary Page Specification", () => {
  it("renders under-24h message when CGM history is insufficient and shows GMI estimated from window", async () => {
    vi.spyOn(api, "getPatientProfile").mockResolvedValue(mockProfile as any);
    vi.spyOn(api, "getClinicalSummary").mockResolvedValue(mockSummary as any);
    vi.spyOn(api, "getPersonalizationStatus").mockResolvedValue({ is_personalized: false } as any);
    vi.spyOn(api, "getPatientCGM").mockResolvedValue([]);

    render(<ClinicalSummaryPage />);

    // Stated monitoring window
    expect(await screen.findByText("last 24 hours")).toBeDefined();

    // 5-band TIR percentages
    const inRangeSegment = screen.getByText(/68.5% in range/);
    expect(inRangeSegment).toBeDefined();
    expect(inRangeSegment.className).toContain("bg-[#10B981]");

    // Consensus Metrics & GMI window label
    expect(screen.getByText("7.4%")).toBeDefined(); // GMI
    expect(screen.getByText("estimated from this window")).toBeDefined();
    expect(screen.getByText("25.0%")).toBeDefined(); // CV
    expect(screen.getByText("154")).toBeDefined(); // Mean Glucose
    expect(screen.getByText("38.6")).toBeDefined(); // SD

    // AGP section title and clean under-24h message
    expect(screen.getByText(/Ambulatory Glucose Profile/i)).toBeDefined();
    expect(
      screen.getByText("Not enough data for an AGP yet (needs at least 24 h)")
    ).toBeDefined();

    // Decision support disclaimer
    expect(screen.getByText(/Decision support only/i)).toBeDefined();
  });

  it("renders AGP chart when CGM history has at least 24 hours of data", async () => {
    vi.spyOn(api, "getPatientProfile").mockResolvedValue(mockProfile as any);
    vi.spyOn(api, "getClinicalSummary").mockResolvedValue(mockSummary as any);
    vi.spyOn(api, "getPersonalizationStatus").mockResolvedValue({ is_personalized: false } as any);

    // Mock 25 hours of CGM readings
    const cgm25h = [
      { timestamp: "2026-10-14T00:00:00Z", glucose_mgdl: 120.0, patient_id: "IND_PAT_001" },
      { timestamp: "2026-10-14T12:00:00Z", glucose_mgdl: 140.0, patient_id: "IND_PAT_001" },
      { timestamp: "2026-10-15T01:00:00Z", glucose_mgdl: 130.0, patient_id: "IND_PAT_001" },
    ];
    vi.spyOn(api, "getPatientCGM").mockResolvedValue(cgm25h as any);

    render(<ClinicalSummaryPage />);

    expect(await screen.findByText("last 24 hours")).toBeDefined();
    expect(
      screen.queryByText("Not enough data for an AGP yet (needs at least 24 h)")
    ).toBeNull();
    expect(screen.getByText("Median (50th)")).toBeDefined();

    // Verify renamed sections
    expect(screen.getByText("Auto-generated summary")).toBeDefined();
    expect(screen.getByText("Findings")).toBeDefined();
    expect(screen.getByText("Suggestions for clinician review")).toBeDefined();
    expect(screen.getByText("Maintain consistent meal timing.")).toBeDefined();
  });

  it("renders 'No data-driven suggestions for this window' when actionable recommendations is empty", async () => {
    vi.spyOn(api, "getPatientProfile").mockResolvedValue(mockProfile as any);
    const summaryEmptyRecs = {
      ...mockSummary,
      actionable_recommendations: [],
    };
    vi.spyOn(api, "getClinicalSummary").mockResolvedValue(summaryEmptyRecs as any);
    vi.spyOn(api, "getPersonalizationStatus").mockResolvedValue({ is_personalized: false } as any);
    vi.spyOn(api, "getPatientCGM").mockResolvedValue([]);

    render(<ClinicalSummaryPage />);

    expect(await screen.findByText("Suggestions for clinician review")).toBeDefined();
    expect(
      screen.getByText("No data-driven suggestions for this window")
    ).toBeDefined();
  });
});
