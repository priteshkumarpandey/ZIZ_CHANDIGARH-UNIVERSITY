import React from "react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { TopBar } from "../components/TopBar";
import { useGlucoTwinStore } from "../lib/store";
import { PatientProfile } from "../lib/types";

const mockPatients = [
  {
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
    active_medications: ["Metformin 1000mg BID", "Glimepiride 2mg OD"],
    phenotype_notes: "Pronounced dawn phenomenon.",
    is_synthetic: true,
  },
  {
    id: "DEMO_PAT_001",
    name: "Demo Patient 1 (Synthetic T2D)",
    age: 54,
    gender: "male",
    region: "West India",
    hba1c: 7.8,
    bmi: 28.4,
    fasting_glucose: 142.0,
    dietary_habit: "Vegetarian",
    occupation: "Synthetic Engineer",
    active_medications: ["Metformin 500mg"],
    phenotype_notes: "Synthetic T2D patient with postprandial glycemic excursions.",
    is_synthetic: true,
  },
  {
    id: "CGMacros-004",
    name: "CGMacros-004",
    age: 45,
    gender: "male",
    region: null,
    occupation: null,
    dietary_habit: null,
    hba1c: 6.9,
    bmi: 24.2,
    fasting_glucose: 110.0,
    ethnicity: "South Asian",
    active_medications: [],
    phenotype_notes: "De-identified observational protocol.",
    is_synthetic: false,
  },
];

const typedPatients = mockPatients as unknown as PatientProfile[];

describe("Replay Controls Component Tests", () => {
  beforeEach(() => {
    const store = useGlucoTwinStore.getState();
    store.setPatients(typedPatients);
    store.setSelectedPatient(typedPatients[0]);
    store.setMode("replay");
    store.setIsPlaying(false);
    store.setReplaySpeed(1);
    store.setScrubberIndex(0);
    store.setMaxScrubberIndex(100);
  });

  it("toggles playback state between Play and Pause upon clicking", () => {
    const { rerender } = render(<TopBar />);

    const playBtn = screen.getByRole("button", { name: /^Play$/i });
    expect(playBtn).toBeDefined();

    fireEvent.click(playBtn);
    expect(useGlucoTwinStore.getState().isPlaying).toBe(true);

    rerender(<TopBar />);
    const pauseBtn = screen.getByRole("button", { name: /^Pause$/i });
    expect(pauseBtn).toBeDefined();

    fireEvent.click(pauseBtn);
    expect(useGlucoTwinStore.getState().isPlaying).toBe(false);
  });

  it("updates speed multiplier across 1x, 5x, 15x, 30x, 60x", () => {
    render(<TopBar />);

    const computeInterval = (speed: number) => Math.max(50, Math.floor(1000 / speed));

    // Initially 1x
    expect(useGlucoTwinStore.getState().replaySpeed).toBe(1);
    expect(computeInterval(1)).toBe(1000);

    // Click 5x
    const btn5x = screen.getByRole("button", { name: "5x" });
    fireEvent.click(btn5x);
    expect(useGlucoTwinStore.getState().replaySpeed).toBe(5);
    expect(computeInterval(5)).toBe(200);

    // Click 15x
    const btn15x = screen.getByRole("button", { name: "15x" });
    fireEvent.click(btn15x);
    expect(useGlucoTwinStore.getState().replaySpeed).toBe(15);
    expect(computeInterval(15)).toBe(66);

    // Click 30x
    const btn30x = screen.getByRole("button", { name: "30x" });
    fireEvent.click(btn30x);
    expect(useGlucoTwinStore.getState().replaySpeed).toBe(30);
    expect(computeInterval(30)).toBe(50); // Bound by 50ms floor

    // Click 60x
    const btn60x = screen.getByRole("button", { name: "60x" });
    fireEvent.click(btn60x);
    expect(useGlucoTwinStore.getState().replaySpeed).toBe(60);
    expect(computeInterval(60)).toBe(50);
  });

  it("renders step counter only when series is loaded", () => {
    const { rerender } = render(<TopBar />);
    // maxScrubberIndex is 100 > 0, so step counter is rendered
    expect(screen.getByText(/0\/100/)).toBeDefined();

    // Set maxScrubberIndex to 0
    useGlucoTwinStore.getState().setMaxScrubberIndex(0);
    rerender(<TopBar />);
    expect(screen.queryByText(/0\/100/)).toBeNull();
  });
});

describe("Data Badge Provenance on TopBar", () => {
  it("renders honest data badge corresponding to selected patient cohort", () => {
    const { rerender } = render(<TopBar />);

    // Pathname /twin/DEMO_PAT_001 synchronizes to DEMO_PAT_001 -> Synthetic demo data
    expect(screen.getByText("Synthetic demo data")).toBeDefined();

    // Switch to IND_PAT_001 -> Synthetic persona (synthetic linkage)
    useGlucoTwinStore.getState().setSelectedPatient(typedPatients[0]);
    rerender(<TopBar />);
    expect(screen.getByText("Synthetic persona (synthetic linkage)")).toBeDefined();

    // Switch to CGMacros-004 -> De-identified research data
    useGlucoTwinStore.getState().setSelectedPatient(typedPatients[2]);
    rerender(<TopBar />);
    expect(screen.getByText("De-identified research data")).toBeDefined();
  });

  it("gates replay controls to twin page and hides them on non-twin pages", async () => {
    // Current route in setup is /twin/DEMO_PAT_001
    const { rerender } = render(<TopBar />);
    expect(screen.getByRole("button", { name: /^Play$/i })).toBeDefined();

    // Mock navigation to /summary/DEMO_PAT_001
    const nextNav = await import("next/navigation");
    vi.spyOn(nextNav, "usePathname").mockReturnValue("/summary/DEMO_PAT_001");

    useGlucoTwinStore.getState().setClinicalSummary({
      patient_id: "DEMO_PAT_001",
      patient_name: "Demo Patient 1",
      generated_at: "2026-10-15T10:30:00Z",
    } as any);

    rerender(<TopBar />);
    // Replay controls hidden on summary
    expect(screen.queryByRole("button", { name: /^Play$/i })).toBeNull();
    // As of indicator shown on summary
    expect(screen.getByText(/As of:/i)).toBeDefined();

    // Mock navigation to /evidence
    vi.spyOn(nextNav, "usePathname").mockReturnValue("/evidence");
    rerender(<TopBar />);
    expect(screen.queryByRole("button", { name: /^Play$/i })).toBeNull();
    expect(screen.queryByText(/As of:/i)).toBeNull();

    // Mock navigation to /what-if
    vi.spyOn(nextNav, "usePathname").mockReturnValue("/what-if");
    rerender(<TopBar />);
    expect(screen.queryByRole("button", { name: /^Play$/i })).toBeNull();
    expect(screen.queryByText(/As of:/i)).toBeNull();
  });
});
