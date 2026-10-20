"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import { useParams } from "next/navigation";
import {
  getPatientProfile,
  getPatientCGM,
  getTwinState,
  predictGlucose,
  getClinicalSummary,
  getExplainability,
  getPersonalizationStatus,
} from "../../../lib/api";
import {
  PatientProfile,
  CGMReading,
  ForecastResponse,
  ClinicalSummaryResponse,
  TwinStateResponse,
  ExplainabilityResponse,
  PersonalizationStatus,
} from "../../../lib/types";
import { useGlucoTwinStore } from "../../../lib/store";
import { PatientHeroCard } from "../../../components/PatientHeroCard";
import { GlucoseForecastChart } from "../../../components/GlucoseForecastChart";
import { AlertsPanel } from "../../../components/AlertsPanel";
import { ExplainabilityPanel } from "../../../components/ExplainabilityPanel";
import { PersonalizationCard } from "../../../components/PersonalizationCard";
import { TwinFusionDiagram } from "../../../components/TwinFusionDiagram";
import { DoctorSummaryTeaser } from "../../../components/DoctorSummaryTeaser";
import { ClinicalSummaryModal } from "../../../components/ClinicalSummaryModal";

export default function TwinDetailView() {
  const params = useParams();
  const patientId = (params.id as string) || "IND_PAT_001";

  const {
    setSelectedPatient,
    twinState,
    setTwinState,
    cgmHistory,
    setCgmHistory,
    latestForecast,
    setLatestForecast,
    explainability,
    setExplainability,
    personalization,
    setPersonalization,
    clinicalSummary,
    setClinicalSummary,
    isSummaryModalOpen,
    setIsSummaryModalOpen,
    mode,
    isPlaying,
    replaySpeed,
    scrubberIndex,
    setScrubberIndex,
    setMaxScrubberIndex,
  } = useGlucoTwinStore();

  const [profile, setProfile] = useState<PatientProfile | null>(null);
  const [allCgmPoints, setAllCgmPoints] = useState<CGMReading[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  const explainAbortControllerRef = useRef<AbortController | null>(null);
  const activeTimestampRef = useRef<string | null>(null);

  // Load patient telemetry and baseline endpoints
  useEffect(() => {
    setLoading(true);
    Promise.all([
      getPatientProfile(patientId),
      getTwinState(patientId).catch(() => null),
      getPatientCGM(patientId, 3), // Fetch up to 3 days
      getPersonalizationStatus(patientId).catch(() => null),
    ])
      .then(([prof, state, cgm, pers]) => {
        setProfile(prof);
        setSelectedPatient(prof);
        if (state) setTwinState(state);
        setAllCgmPoints(cgm);
        setCgmHistory(cgm);
        if (pers) setPersonalization(pers);

        // Setup scrubber: check if a specific step is requested in URL search params (e.g. ?step=10)
        const total = cgm.length;
        const maxIdx = Math.max(0, total - 72);
        setMaxScrubberIndex(maxIdx);

        let initialIdx = maxIdx;
        if (typeof window !== "undefined") {
          const urlParams = new URLSearchParams(window.location.search);
          const stepParam = urlParams.get("step");
          if (stepParam !== null) {
            const parsed = parseInt(stepParam, 10);
            if (!isNaN(parsed) && parsed >= 0 && parsed <= maxIdx) {
              initialIdx = parsed;
            }
          }
        }

        setScrubberIndex(initialIdx);
        const windowSlice = cgm.slice(initialIdx, initialIdx + 72);
        setCgmHistory(windowSlice);

        if (state && windowSlice.length > 0) {
          const currentReading = windowSlice[windowSlice.length - 1];
          setTwinState({
            ...state,
            current_glucose: currentReading.glucose_mgdl,
            trend_arrow: currentReading.trend_arrow || state.trend_arrow,
            last_updated: currentReading.timestamp,
          });
        }

        // Initial prediction and explainability strictly aligned with as_of row
        const series = windowSlice.map((r) => r.glucose_mgdl);
        const lastTimestamp = windowSlice.length > 0 ? windowSlice[windowSlice.length - 1].timestamp : undefined;

        explainAbortControllerRef.current?.abort();
        const controller = new AbortController();
        explainAbortControllerRef.current = controller;
        activeTimestampRef.current = lastTimestamp ?? null;

        return Promise.all([
          predictGlucose(patientId, series, 0, 0, lastTimestamp, controller.signal)
            .then((fc) => {
              if (activeTimestampRef.current === (lastTimestamp ?? null) && !controller.signal.aborted) {
                setLatestForecast(fc);
              }
            })
            .catch((e) => {
              if (e?.name !== "AbortError") console.error("Predict error", e);
            }),
          getExplainability(patientId, 60, lastTimestamp, controller.signal)
            .then((exp) => {
              if (activeTimestampRef.current === (lastTimestamp ?? null) && !controller.signal.aborted) {
                setExplainability(exp);
              }
            })
            .catch((e) => {
              if (e?.name !== "AbortError") console.error("Explain error", e);
            }),
          getClinicalSummary(patientId, lastTimestamp, controller.signal)
            .then((summ) => {
              if (activeTimestampRef.current === (lastTimestamp ?? null) && !controller.signal.aborted) {
                setClinicalSummary(summ);
              }
            })
            .catch((e) => {
              if (e?.name !== "AbortError") console.error("Summary error", e);
            }),
        ]);
      })
      .catch((err) => console.error("Error loading twin:", err))
      .finally(() => setLoading(false));

    return () => {
      explainAbortControllerRef.current?.abort();
    };
  }, [patientId]);

  // Replay step function
  const stepReplay = useCallback(
    async (targetIdx: number) => {
      if (allCgmPoints.length < 72) return;
      const start = targetIdx;
      const end = targetIdx + 72;
      const windowReadings = allCgmPoints.slice(start, end);
      if (windowReadings.length === 0) return;

      setCgmHistory(windowReadings);
      setScrubberIndex(targetIdx);

      const series = windowReadings.map((r) => r.glucose_mgdl);
      const currentReading = windowReadings[windowReadings.length - 1];
      const snapshotTimestamp = currentReading.timestamp;

      // Update current glucose in twinState synchronously
      const currentTwin = useGlucoTwinStore.getState().twinState;
      if (currentTwin) {
        setTwinState({
          ...currentTwin,
          current_glucose: currentReading.glucose_mgdl,
          trend_arrow: currentReading.trend_arrow || currentTwin.trend_arrow,
          last_updated: snapshotTimestamp,
        });
      }

      // Abort previous in-flight requests and register new active timestamp
      explainAbortControllerRef.current?.abort();
      const controller = new AbortController();
      explainAbortControllerRef.current = controller;
      activeTimestampRef.current = snapshotTimestamp;

      try {
        const fc = await predictGlucose(
          patientId,
          series,
          0,
          0,
          snapshotTimestamp,
          controller.signal
        );
        if (activeTimestampRef.current === snapshotTimestamp && !controller.signal.aborted) {
          setLatestForecast(fc);
        }

        getExplainability(patientId, 60, snapshotTimestamp, controller.signal)
          .then((exp) => {
            if (activeTimestampRef.current === snapshotTimestamp && !controller.signal.aborted) {
              setExplainability(exp);
            }
          })
          .catch((e) => {
            if (e?.name !== "AbortError") console.error("Explain error", e);
          });

        getClinicalSummary(patientId, snapshotTimestamp, controller.signal)
          .then((summ) => {
            if (activeTimestampRef.current === snapshotTimestamp && !controller.signal.aborted) {
              setClinicalSummary(summ);
            }
          })
          .catch((e) => {
            if (e?.name !== "AbortError") console.error("Summary error", e);
          });
      } catch (e: any) {
        if (e?.name !== "AbortError") {
          console.error("Replay predict error", e);
        }
      }
    },
    [
      allCgmPoints,
      patientId,
      setCgmHistory,
      setScrubberIndex,
      setTwinState,
      setLatestForecast,
      setExplainability,
      setClinicalSummary,
    ]
  );

  useEffect(() => {
    if (typeof window !== "undefined") {
      (window as unknown as { __stepReplay?: (idx: number) => Promise<void> }).__stepReplay = stepReplay;
    }
    return () => {
      if (typeof window !== "undefined") {
        delete (window as unknown as { __stepReplay?: unknown }).__stepReplay;
      }
    };
  }, [stepReplay]);

  // Playback timer loop
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  useEffect(() => {
    if (mode === "replay" && isPlaying) {
      const delay = Math.max(50, Math.floor(1000 / replaySpeed));
      intervalRef.current = setInterval(() => {
        const maxIdx = Math.max(0, allCgmPoints.length - 72);
        const currentIdx = useGlucoTwinStore.getState().scrubberIndex;
        const nextIdx = currentIdx >= maxIdx ? 0 : currentIdx + 1;
        stepReplay(nextIdx);
      }, delay);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [mode, isPlaying, replaySpeed, allCgmPoints.length, stepReplay]);


  if (loading || !profile) {
    return (
      <div className="p-12 text-center text-xs text-text-muted">
        Calibrating digital twin for {patientId}...
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Main 12-column Grid (8 cols left, 4 cols right) */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-5">
        {/* Left 8 columns: Patient Hero Card + Hero Glucose Forecast Chart */}
        <div className="xl:col-span-8 space-y-4">
          <PatientHeroCard
            profile={profile}
            twinState={twinState}
            servingFold={latestForecast?.serving_fold ?? undefined}
            cgmHistory={cgmHistory}
          />
          <GlucoseForecastChart
            cgmHistory={cgmHistory}
            forecast={latestForecast}
            patientName={profile.name}
            isPersonalized={Boolean(personalization?.is_personalized || (personalization as any)?.is_active)}
          />
        </div>

        {/* Right 4 columns: Alerts, Why (Explainability), and Personalization stacked */}
        <div className="xl:col-span-4 space-y-3 pr-0.5">
          <AlertsPanel
            forecast={latestForecast}
            currentGlucose={twinState?.current_glucose}
          />
          <ExplainabilityPanel
            explainability={explainability}
            asOf={twinState?.last_updated || undefined}
            currentGlucose={twinState?.current_glucose}
          />
          <PersonalizationCard personalization={personalization} />
        </div>
      </div>

      {/* Below main row: Flat SVG Twin Fusion Pipeline Diagram & Doctor Clinical Summary Teaser */}
      <div className="space-y-5 pt-2">
        <TwinFusionDiagram />
        <DoctorSummaryTeaser
          summary={clinicalSummary}
          onOpenModal={() => setIsSummaryModalOpen(true)}
        />
      </div>

      {/* Modal Dossier */}
      <ClinicalSummaryModal summary={clinicalSummary} />
    </div>
  );
}

