import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { PatientHeroCard } from "../components/PatientHeroCard";
import { TopBar } from "../components/TopBar";
import { useGlucoTwinStore } from "../lib/store";
import { PatientProfile } from "../lib/types";

const mockCgmacrosPatient: PatientProfile = {
  id: "CGMacros-004",
  name: "CGMacros-004",
  age: 45,
  gender: "male",
  region: null,
  occupation: null,
  dietary_habit: null,
  postprandial_glucose: null,
  ethnicity: "South Asian",
  hba1c: 6.9,
  bmi: 24.2,
  fasting_glucose: 110.0,
  active_medications: [],
  phenotype_notes: "",
  is_synthetic: false,
};

const mockSyntheticPersona: PatientProfile = {
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

describe("Honest Patient Data Display", () => {
  it("never renders region, occupation, or diet for real CGMacros participants", () => {
    const { container } = render(
      <PatientHeroCard
        profile={mockCgmacrosPatient}
        twinState={null}
        servingFold={2}
      />
    );

    // Shows honest clinical fields
    expect(screen.getAllByText("CGMacros-004").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/45 yrs/)).toBeDefined();
    expect(screen.getByText(/male/i)).toBeDefined();
    expect(screen.getByText(/BMI 24.2/)).toBeDefined();
    expect(screen.getByText(/HbA1c 6.9%/)).toBeDefined();
    expect(screen.getByText(/South Asian/)).toBeDefined();

    // Verify fold note for real participant
    expect(
      screen.getByText("This participant was held out when the serving model was trained (fold 2).")
    ).toBeDefined();

    // MUST NOT render invented fields
    expect(container.textContent).not.toContain("North India");
    expect(container.textContent).not.toContain("Civil Engineer");
    expect(container.textContent).not.toContain("Volunteer");
    expect(container.textContent).not.toContain("Vegetarian");
    expect(container.textContent).not.toContain("Research Subject");
  });

  it("renders synthetic badge, region, and occupation for synthetic personas", () => {
    render(
      <PatientHeroCard
        profile={mockSyntheticPersona}
        twinState={null}
        servingFold={1}
      />
    );

    expect(screen.getByText("Rajesh Sharma")).toBeDefined();
    expect(screen.getByText("North India")).toBeDefined();
    expect(screen.getByText("Civil Engineer")).toBeDefined();
    expect(
      screen.getByText("Synthetic demo patient, never part of any training data. Served by the fold 1 model.")
    ).toBeDefined();
  });
});

describe("Patient Selector and Navigation in TopBar", () => {
  it("allows selecting a patient and displays the selected patient in dropdown", () => {
    const store = useGlucoTwinStore.getState();
    store.setPatients([mockSyntheticPersona, mockCgmacrosPatient]);
    store.setSelectedPatient(mockSyntheticPersona);

    render(<TopBar />);

    const select = screen.getByLabelText(/Patient:/i) as HTMLSelectElement;
    expect(select.value).toBe("IND_PAT_001");

    fireEvent.change(select, { target: { value: "CGMacros-004" } });
    expect(useGlucoTwinStore.getState().selectedPatient?.id).toBe("CGMacros-004");
  });
});
