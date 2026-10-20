import { create } from "zustand";
import {
  PatientProfile,
  TwinStateResponse,
  CGMReading,
  ForecastResponse,
  ExplainabilityResponse,
  SimulationResponse,
  PersonalizationStatus,
  ClinicalSummaryResponse,
} from "./types";

export type ReplayMode = "replay" | "stream";

interface GlucoTwinState {
  patients: PatientProfile[];
  selectedPatient: PatientProfile | null;
  twinState: TwinStateResponse | null;
  cgmHistory: CGMReading[];
  latestForecast: ForecastResponse | null;
  explainability: ExplainabilityResponse | null;
  simulation: SimulationResponse | null;
  personalization: PersonalizationStatus | null;
  clinicalSummary: ClinicalSummaryResponse | null;
  isSummaryModalOpen: boolean;
  isLoading: boolean;
  error: string | null;

  // Replay & Stream controls
  mode: ReplayMode;
  isPlaying: boolean;
  replaySpeed: number; // 1, 5, 10, 30, 60
  scrubberIndex: number;
  maxScrubberIndex: number;

  setPatients: (patients: PatientProfile[]) => void;
  setSelectedPatient: (patient: PatientProfile | null) => void;
  setTwinState: (state: TwinStateResponse | null) => void;
  setCgmHistory: (readings: CGMReading[]) => void;
  appendCgmReading: (reading: CGMReading) => void;
  setLatestForecast: (forecast: ForecastResponse | null) => void;
  setExplainability: (exp: ExplainabilityResponse | null) => void;
  setSimulation: (sim: SimulationResponse | null) => void;
  setPersonalization: (pers: PersonalizationStatus | null) => void;
  setClinicalSummary: (summary: ClinicalSummaryResponse | null) => void;
  setIsSummaryModalOpen: (isOpen: boolean) => void;
  setIsLoading: (loading: boolean) => void;
  setError: (err: string | null) => void;

  setMode: (mode: ReplayMode) => void;
  setIsPlaying: (isPlaying: boolean) => void;
  setReplaySpeed: (speed: number) => void;
  setScrubberIndex: (idx: number) => void;
  setMaxScrubberIndex: (max: number) => void;
}

export const useGlucoTwinStore = create<GlucoTwinState>((set) => ({
  patients: [],
  selectedPatient: null,
  twinState: null,
  cgmHistory: [],
  latestForecast: null,
  explainability: null,
  simulation: null,
  personalization: null,
  clinicalSummary: null,
  isSummaryModalOpen: false,
  isLoading: false,
  error: null,

  mode: "replay",
  isPlaying: false,
  replaySpeed: 1,
  scrubberIndex: 0,
  maxScrubberIndex: 0,

  setPatients: (patients) => set({ patients }),
  setSelectedPatient: (selectedPatient) => set({ selectedPatient }),
  setTwinState: (twinState) => set({ twinState }),
  setCgmHistory: (cgmHistory) => set({ cgmHistory }),
  appendCgmReading: (reading) =>
    set((state) => ({
      cgmHistory: [...state.cgmHistory.slice(-288), reading],
    })),
  setLatestForecast: (latestForecast) => set({ latestForecast }),
  setExplainability: (explainability) => set({ explainability }),
  setSimulation: (simulation) => set({ simulation }),
  setPersonalization: (personalization) => set({ personalization }),
  setClinicalSummary: (clinicalSummary) => set({ clinicalSummary }),
  setIsSummaryModalOpen: (isSummaryModalOpen) => set({ isSummaryModalOpen }),
  setIsLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),

  setMode: (mode) => set({ mode }),
  setIsPlaying: (isPlaying) => set({ isPlaying }),
  setReplaySpeed: (replaySpeed) => set({ replaySpeed }),
  setScrubberIndex: (scrubberIndex) => set({ scrubberIndex }),
  setMaxScrubberIndex: (maxScrubberIndex) => set({ maxScrubberIndex }),
}));

