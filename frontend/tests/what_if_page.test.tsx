import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import WhatIfPage from "../app/what-if/page";
import * as api from "../lib/api";

const mockPatients = [
  {
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
  },
];

const mockFoods = [
  {
    id: "ifct_001",
    name: "Roti with Dal Tadka",
    regional_category: "North Indian",
    carbohydrates_g: 45.0,
    fiber_g: 6.2,
    protein_g: 11.5,
    fat_g: 5.8,
    glycemic_index: 52,
  },
  {
    id: "ifct_002",
    name: "Masala Dosa with Sambar",
    regional_category: "South Indian",
    carbohydrates_g: 54.0,
    fiber_g: 4.8,
    protein_g: 8.2,
    fat_g: 9.4,
    glycemic_index: 68,
  },
];

describe("What-If Page Specification", () => {
  it("renders required provenance badges and fixed validation note", async () => {
    vi.spyOn(api, "getPatients").mockResolvedValue(mockPatients as any);
    vi.spyOn(api, "getFoodCatalog").mockResolvedValue(mockFoods as any);

    render(<WhatIfPage />);

    // Required Provenance Badges
    expect(screen.getByText("model-based")).toBeDefined();
    expect(screen.getByText("assumption-based, unvalidated")).toBeDefined();
    expect(screen.getByText("approximate nutrient values")).toBeDefined();

    // Indian meal library controls
    expect(await screen.findByText(/Indian Meal Library/i)).toBeDefined();
    expect(screen.getByText(/Post-Meal Brisk Walk|Walk/i)).toBeDefined();

    // Fixed validation note with real numbers from reports/whatif_validation.json
    expect(screen.getAllByText(/32.67/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/66.82/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/0.399/).length).toBeGreaterThanOrEqual(1);
  });

  it("populates the Regional Dish dropdown with all 20 meals of the Indian meal library", async () => {
    const meal20Names = [
      "Whole Wheat Roti (Chapati)",
      "Steamed White Rice (Sona Masoori)",
      "Steamed Brown Rice",
      "Yellow Dal Tadka (Toor Dal)",
      "Paneer Tikka / Bhurji",
      "Masala Dosa with Potato Filling",
      "Steamed Idli (2 pieces)",
      "Vegetable Sambar",
      "Hyderabadi Chicken Biryani",
      "Kanda Poha (Flattened Rice)",
      "Moong Dal Khichdi",
      "Bengali Fish Curry (Macher Jhol)",
      "Gulab Jamun (1 piece)",
      "Masala Chai with 2 tsp Sugar",
      "Ragi Roti (Finger Millet)",
      "Palak Paneer",
      "Chana Masala (Chickpea Curry)",
      "Besan Chilla (Gram Flour Pancake)",
      "Upma (Semolina / Sooji)",
      "Curd / Dahi (Plain Whole Milk)",
    ];

    const full20Foods = meal20Names.map((name, i) => ({
      id: `ifct_${(i + 1).toString().padStart(3, "0")}`,
      name,
      regional_category: "Indian Cuisine",
      serving_size_g: 100.0,
      carbohydrates_g: 30.0,
      fiber_g: 4.0,
      protein_g: 5.0,
      fat_g: 2.0,
      glycemic_index: 55.0,
      is_approximate: true,
    }));

    vi.spyOn(api, "getPatients").mockResolvedValue(mockPatients as any);
    vi.spyOn(api, "getFoodCatalog").mockResolvedValue(full20Foods as any);
    vi.spyOn(api, "getEvidenceData").mockResolvedValue({} as any);

    const { container } = render(<WhatIfPage />);

    // Wait for dropdown options to render
    const option = await screen.findByText((content) =>
      content.includes(meal20Names[0])
    );
    expect(option).toBeDefined();

    // Query all options in the first select (Scenario A Regional Dish)
    const options = container.querySelectorAll("select")[1]?.querySelectorAll("option");
    expect(options?.length).toBe(20);

    // Verify all 20 dish names are present
    meal20Names.forEach((dish) => {
      expect(
        screen.getByText((content) => content.includes(dish))
      ).toBeDefined();
    });
  });

  it("renders curves, fans, delta chips for each scenario, B minus A row, and assumption-based walk label in compare mode", async () => {
    vi.spyOn(api, "getPatients").mockResolvedValue(mockPatients as any);
    vi.spyOn(api, "getFoodCatalog").mockResolvedValue(mockFoods as any);
    vi.spyOn(api, "getEvidenceData").mockResolvedValue({} as any);

    const mockSimA = {
      patient_id: "IND_PAT_001",
      simulation_timestamp: new Date().toISOString(),
      baseline_trajectory: [140, 145, 150, 155, 160, 165],
      scenario_trajectory: [140, 155, 175, 190, 185, 170],
      scenario_q10: [135, 145, 160, 175, 170, 160],
      scenario_q90: [145, 165, 190, 205, 200, 180],
      baseline_peak_glucose: 165,
      scenario_peak_glucose: 190,
      clinical_takeaway: "Postprandial peak shift of +25 mg/dL.",
    };

    const mockSimB = {
      patient_id: "IND_PAT_001",
      simulation_timestamp: new Date().toISOString(),
      baseline_trajectory: [140, 145, 150, 155, 160, 165],
      scenario_trajectory: [140, 148, 160, 168, 162, 155],
      scenario_q10: [135, 140, 150, 158, 152, 145],
      scenario_q90: [145, 156, 170, 178, 172, 165],
      baseline_peak_glucose: 165,
      scenario_peak_glucose: 168,
      clinical_takeaway: "Postprandial peak shift attenuated with brisk walk.",
    };

    vi.spyOn(api, "runSimulation")
      .mockResolvedValueOnce(mockSimA as any)
      .mockResolvedValueOnce(mockSimB as any);

    const { container } = render(<WhatIfPage />);

    // Check that walk slider labels are assumption-based
    expect(screen.getAllByText(/assumption-based/i).length).toBeGreaterThanOrEqual(1);

    // Toggle compare mode
    const checkboxes = container.querySelectorAll('input[type="checkbox"]');
    const compareCheckbox = checkboxes[0];
    if (compareCheckbox) {
      compareCheckbox.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    }

    // Click Run What-If Simulation
    const runBtn = screen.getByRole("button", { name: /Run What-If Simulation/i });
    runBtn.dispatchEvent(new MouseEvent("click", { bubbles: true }));

    // Wait for simulation results
    expect(await screen.findByText(/Scenario A \(vs Baseline\):/i)).toBeDefined();
    expect(screen.getByText(/Scenario B \(vs Baseline\):/i)).toBeDefined();
    expect(screen.getByText(/B minus A \(Intervention Difference\):/i)).toBeDefined();

    // Check legend items
    expect(screen.getByText("Baseline")).toBeDefined();
    expect(screen.getByText("Scenario A")).toBeDefined();
    expect(screen.getByText("Scenario A Fan")).toBeDefined();
    expect(screen.getByText("Scenario B")).toBeDefined();
    expect(screen.getByText("Scenario B Fan")).toBeDefined();
  });
});
