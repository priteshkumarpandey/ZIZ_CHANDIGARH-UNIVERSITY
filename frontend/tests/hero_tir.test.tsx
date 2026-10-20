import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import { PatientHeroCard } from "../components/PatientHeroCard";
import { PatientProfile, CGMReading } from "../lib/types";

const mockPatient: PatientProfile = {
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

describe("PatientHeroCard Dynamic Time in Range (TIR)", () => {
  it("computes TIR dynamically from cgmHistory window", () => {
    // 4 readings: 100 and 150 are in range (70 to 180), 200 is above, 50 is below
    // Expected TIR = 2 / 4 = 50%
    const mockCgm: CGMReading[] = [
      { patient_id: "IND_PAT_001", timestamp: "2026-10-10T10:00:00Z", glucose_mgdl: 100, trend_arrow: null, steps: null, mets: null },
      { patient_id: "IND_PAT_001", timestamp: "2026-10-10T10:05:00Z", glucose_mgdl: 150, trend_arrow: null, steps: null, mets: null },
      { patient_id: "IND_PAT_001", timestamp: "2026-10-10T10:10:00Z", glucose_mgdl: 200, trend_arrow: null, steps: null, mets: null },
      { patient_id: "IND_PAT_001", timestamp: "2026-10-10T10:15:00Z", glucose_mgdl: 50, trend_arrow: null, steps: null, mets: null },
    ];

    render(
      <PatientHeroCard
        profile={mockPatient}
        twinState={null}
        cgmHistory={mockCgm}
      />
    );

    expect(screen.getByText("50%")).toBeDefined();
    expect(screen.getByText("Time in Range (last 6 h)")).toBeDefined();
  });

  it("computes 100% TIR when all readings are in target range", () => {
    const mockCgm: CGMReading[] = [
      { patient_id: "IND_PAT_001", timestamp: "2026-10-10T10:00:00Z", glucose_mgdl: 95, trend_arrow: null, steps: null, mets: null },
      { patient_id: "IND_PAT_001", timestamp: "2026-10-10T10:05:00Z", glucose_mgdl: 120, trend_arrow: null, steps: null, mets: null },
      { patient_id: "IND_PAT_001", timestamp: "2026-10-10T10:10:00Z", glucose_mgdl: 140, trend_arrow: null, steps: null, mets: null },
    ];

    render(
      <PatientHeroCard
        profile={mockPatient}
        twinState={null}
        cgmHistory={mockCgm}
      />
    );

    expect(screen.getByText("100%")).toBeDefined();
  });
});
