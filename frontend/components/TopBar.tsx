"use client";

import React, { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { Play, Pause, FastForward, RotateCcw } from "lucide-react";
import { useGlucoTwinStore } from "../lib/store";
import { getCohortBadgeText, getPatientCohort } from "../lib/types";
import { getPatients } from "../lib/api";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";

interface TopBarProps {
  onStepForward?: () => void;
  onResetReplay?: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({ onStepForward, onResetReplay }) => {
  const pathname = usePathname();
  const router = useRouter();
  const isTwinPage = pathname.startsWith("/twin/");
  const isSummaryPage = pathname.startsWith("/summary/");
  const routePatientId = pathname.match(/^\/(?:twin|summary)\/([^/]+)/)?.[1];

  const {
    patients,
    setPatients,
    selectedPatient,
    setSelectedPatient,
    mode,
    setMode,
    isPlaying,
    setIsPlaying,
    replaySpeed,
    setReplaySpeed,
    scrubberIndex,
    maxScrubberIndex,
    clinicalSummary,
    cgmHistory,
  } = useGlucoTwinStore();

  useEffect(() => {
    if (patients.length === 0) {
      getPatients()
        .then((data) => {
          setPatients(data);
          if (routePatientId) {
            const match = data.find((p) => p.id === routePatientId);
            if (match) {
              setSelectedPatient(match);
              return;
            }
          }
          if (!selectedPatient && data.length > 0) {
            setSelectedPatient(data[0]);
          }
        })
        .catch((err) => console.error("Error loading patients in TopBar:", err));
    }
  }, [patients.length, routePatientId, selectedPatient, setPatients, setSelectedPatient]);

  const prevRouteIdRef = React.useRef<string | null>(null);

  // Synchronize patient selector when route patient ID changes
  useEffect(() => {
    if (routePatientId && patients.length > 0 && routePatientId !== prevRouteIdRef.current) {
      prevRouteIdRef.current = routePatientId;
      const match = patients.find((p) => p.id === routePatientId);
      if (match) {
        setSelectedPatient(match);
      }
    }
  }, [routePatientId, patients, setSelectedPatient]);

  const cohort = selectedPatient ? getPatientCohort(selectedPatient.id) : null;
  const cohortBadge = selectedPatient ? getCohortBadgeText(selectedPatient.id) : null;
  const hasSeriesLoaded = maxScrubberIndex > 0;

  // As of time formatting for summary page
  const asOfTimestamp =
    clinicalSummary?.generated_at ||
    (cgmHistory && cgmHistory.length > 0
      ? cgmHistory[cgmHistory.length - 1].timestamp
      : null);
  const formattedAsOf = asOfTimestamp
    ? new Date(asOfTimestamp).toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "Live / Latest";

  return (
    <header className="h-16 border-b border-border bg-surface px-4 lg:px-6 flex items-center justify-between sticky top-0 z-30 whitespace-nowrap overflow-x-hidden">
      {/* Left: Patient Selector & Cohort Badge */}
      <div className="flex items-center gap-3 min-w-0 flex-shrink">
        <div className="flex items-center gap-2 min-w-0 flex-shrink">
          <label htmlFor="patient-select" className="text-xs font-semibold text-text-secondary uppercase tracking-tight shrink-0">
            Patient:
          </label>
          <select
            id="patient-select"
            value={selectedPatient?.id || ""}
            onChange={(e) => {
              const found = patients.find((p) => p.id === e.target.value);
              if (found) {
                setSelectedPatient(found);
                if (pathname.startsWith("/twin/")) {
                  router.push(`/twin/${found.id}`);
                } else if (pathname.startsWith("/summary/")) {
                  router.push(`/summary/${found.id}`);
                }
              }
            }}
            className="text-xs bg-surface-muted border border-border text-text-primary rounded-md px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-navy font-medium max-w-[240px] truncate"
          >
            {patients.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name} ({p.id})
              </option>
            ))}
          </select>
        </div>

        {cohortBadge && (
          <Badge
            variant={
              cohort === "cgmacros_research"
                ? "purple"
                : cohort === "synthetic_persona"
                ? "info"
                : "neutral"
            }
            className="shrink-0"
          >
            {cohortBadge}
          </Badge>
        )}
      </div>

      {/* Right: Replay Controls ONLY on Twin page */}
      {isTwinPage && (
        <div className="flex items-center gap-2.5 shrink-0">
          {/* Mode Selector */}
          <div className="flex items-center bg-surface-muted p-1 rounded-md border border-border">
            <button
              onClick={() => {
                setMode("replay");
                setIsPlaying(false);
              }}
              className={`px-2 py-1 text-xs rounded font-medium transition-colors ${
                mode === "replay"
                  ? "bg-surface text-text-primary border border-border"
                  : "text-text-secondary hover:text-text-primary"
              }`}
            >
              Replay
            </button>
            <button
              onClick={() => {
                setMode("stream");
                setIsPlaying(false);
              }}
              className={`px-2 py-1 text-xs rounded font-medium transition-colors ${
                mode === "stream"
                  ? "bg-surface text-text-primary border border-border"
                  : "text-text-secondary hover:text-text-primary"
              }`}
            >
              Live
            </button>
          </div>

          {/* Replay Controls (Only active in replay mode) */}
          {mode === "replay" ? (
            <div className="flex items-center gap-1.5 border-l border-border pl-2.5">
              <Button
                size="sm"
                variant="outline"
                onClick={() => setIsPlaying(!isPlaying)}
                className="flex items-center gap-1.5 px-2.5"
              >
                {isPlaying ? (
                  <>
                    <Pause size={16} strokeWidth={1.75} className="text-brand" />
                    <span>Pause</span>
                  </>
                ) : (
                  <>
                    <Play size={16} strokeWidth={1.75} className="text-brand" />
                    <span>Play</span>
                  </>
                )}
              </Button>

              {onStepForward && (
                <Button size="sm" variant="outline" onClick={onStepForward} title="Step 1 reading forward">
                  <FastForward size={16} strokeWidth={1.75} className="text-text-secondary" />
                </Button>
              )}

              {onResetReplay && (
                <Button size="sm" variant="outline" onClick={onResetReplay} title="Reset replay to beginning">
                  <RotateCcw size={16} strokeWidth={1.75} className="text-text-secondary" />
                </Button>
              )}

              {/* Speed selector: 1x, 5x, 15x, 30x, 60x */}
              <div className="flex items-center gap-1 text-xs text-text-secondary">
                <span className="text-[11px] text-text-muted">Speed:</span>
                {[1, 5, 15, 30, 60].map((s) => (
                  <button
                    key={s}
                    onClick={() => setReplaySpeed(s)}
                    className={`px-1.5 py-0.5 rounded text-[11px] tnum border ${
                      replaySpeed === s
                        ? "bg-brand text-white border-brand"
                        : "bg-surface-muted text-text-secondary border-border hover:bg-surface"
                    }`}
                  >
                    {s}x
                  </button>
                ))}
              </div>

              {/* Step counter: show ONLY when a series is loaded */}
              {hasSeriesLoaded && (
                <span className="text-[11px] text-text-muted tnum pl-1">
                  Step {scrubberIndex}/{maxScrubberIndex}
                </span>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-2 border-l border-border pl-3 text-xs text-text-secondary">
              <span className="h-2 w-2 rounded-full bg-in-range animate-pulse" />
              <span>WebSocket Live Stream Active</span>
            </div>
          )}
        </div>
      )}

      {/* Right: As of time ONLY on Summary page */}
      {isSummaryPage && (
        <div className="flex items-center gap-2 text-xs bg-surface-muted px-3 py-1.5 rounded-lg border border-border shrink-0">
          <span className="text-[11px] text-text-muted">As of:</span>
          <span className="font-semibold text-text-primary tnum">{formattedAsOf}</span>
        </div>
      )}
    </header>
  );
};
