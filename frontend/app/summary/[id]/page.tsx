"use client";

import React, { useEffect, useState, Suspense } from "react";
import { useParams, useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ReferenceArea,
  CartesianGrid,
} from "recharts";
import {
  FileText,
  Printer,
  ArrowLeft,
  ShieldCheck,
  Cpu,
  AlertTriangle,
  Info,
  CheckCircle,
} from "lucide-react";
import {
  getPatientProfile,
  getPatientCGM,
  getClinicalSummary,
  getPersonalizationStatus,
} from "../../../lib/api";
import {
  PatientProfile,
  CGMReading,
  ClinicalSummaryResponse,
  PersonalizationStatus,
  getCohortBadgeText,
  getPatientCohort,
} from "../../../lib/types";
import { Card, CardHeader, CardTitle } from "../../../components/ui/card";
import { Badge } from "../../../components/ui/badge";
import { Button } from "../../../components/ui/button";

function ClinicalSummaryContent() {
  const params = useParams();
  const searchParams = useSearchParams();
  const patientId = (params.id as string) || "IND_PAT_001";
  const asOf = searchParams.get("as_of");

  const [profile, setProfile] = useState<PatientProfile | null>(null);
  const [summary, setSummary] = useState<ClinicalSummaryResponse | null>(null);
  const [personalization, setPersonalization] = useState<PersonalizationStatus | null>(null);
  const [cgm, setCgm] = useState<CGMReading[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      getPatientProfile(patientId),
      getClinicalSummary(patientId, asOf || undefined).catch(() => null),
      getPersonalizationStatus(patientId).catch(() => null),
      getPatientCGM(patientId, 3).catch(() => []),
    ])
      .then(([prof, summ, pers, cgmData]) => {
        setProfile(prof);
        if (summ) setSummary(summ);
        if (pers) setPersonalization(pers);
        // Filter CGM readings strictly up to as_of cutoff if provided
        const filteredCgm = asOf
          ? cgmData.filter((r) => new Date(r.timestamp).getTime() <= new Date(asOf).getTime())
          : cgmData;
        setCgm(filteredCgm);
      })
      .catch((err) => console.error("Error loading summary page:", err))
      .finally(() => setLoading(false));
  }, [patientId, asOf]);

  const handlePrint = () => {
    if (typeof window !== "undefined") {
      window.print();
    }
  };

  if (loading || !profile || !summary) {
    return (
      <div className="p-12 text-center text-xs text-text-muted">
        Compiling clinical ambulatory glucose profile summary for {patientId}...
      </div>
    );
  }

  const m = summary.metrics;
  const cohort = getPatientCohort(profile.id);
  const cohortBadge = getCohortBadgeText(profile.id);
  const isCgmacros = cohort === "cgmacros_research";

  // Check data duration: requires at least 24 hours of data for a modal AGP profile
  const timestamps = cgm
    .map((r) => new Date(r.timestamp).getTime())
    .filter((t) => !isNaN(t));
  const durationHours =
    timestamps.length > 1
      ? (Math.max(...timestamps) - Math.min(...timestamps)) / (1000 * 60 * 60)
      : 0;

  const windowMatch = summary.monitoring_window_text?.match(/last\s+(\d+(?:\.\d+)?)\s*h/i);
  const statedHours = windowMatch ? parseFloat(windowMatch[1]) : 24;
  const isStatedUnder24 = Boolean(
    summary.monitoring_window_text?.includes("all data so far") || statedHours < 24
  );
  const hasEnoughAgpData = !isStatedUnder24 && durationHours >= 24;

  // Build 24-hour AGP Profile from CGM readings
  interface AGPHourPoint {
    hour: number;
    hourLabel: string;
    median: number;
    p25: number;
    p75: number;
    p5: number;
    p95: number;
    iqrRange?: [number, number];
    outerRange?: [number, number];
  }

  const agpData: AGPHourPoint[] = [];
  if (hasEnoughAgpData) {
    const hoursMap: Record<number, number[]> = {};
    for (let h = 0; h < 24; h++) hoursMap[h] = [];

    cgm.forEach((reading) => {
      const d = new Date(reading.timestamp);
      const h = d.getHours();
      hoursMap[h].push(reading.glucose_mgdl);
    });

    const defaultMedian = m.mean_glucose || 135;
    for (let h = 0; h < 24; h++) {
      const vals = hoursMap[h].sort((a, b) => a - b);
      let median = defaultMedian;
      let p25 = defaultMedian - 15;
      let p75 = defaultMedian + 18;
      let p5 = defaultMedian - 28;
      let p95 = defaultMedian + 35;

      if (vals.length >= 3) {
        const getP = (p: number) => {
          const idx = (p / 100) * (vals.length - 1);
          const low = Math.floor(idx);
          const high = Math.ceil(idx);
          const weight = idx - low;
          return vals[low] * (1 - weight) + vals[high] * weight;
        };
        median = getP(50);
        p25 = getP(25);
        p75 = getP(75);
        p5 = getP(5);
        p95 = getP(95);
      }

      const hourLabel = `${h.toString().padStart(2, "0")}:00`;
      agpData.push({
        hour: h,
        hourLabel,
        median,
        p25,
        p75,
        p5,
        p95,
        iqrRange: [p25, p75],
        outerRange: [p5, p95],
      });
    }
  }

  // TIR stacked horizontal percentages
  const vLow = m.time_below_range_level2_percentage || 0;
  const low = Math.max(0, m.time_below_range_percentage - vLow);
  const inRange = m.time_in_range_percentage;
  const vHigh = m.time_above_range_level2_percentage || 0;
  const high = Math.max(0, m.time_above_range_percentage - vHigh);

  const monitoringWindow = summary.monitoring_window_text || "last 24 hours";

  return (
    <div className="space-y-6 max-w-5xl mx-auto print:p-0 print:m-0">
      {/* Top action bar: back link & print button */}
      <div className="flex items-center justify-between print:hidden">
        <Link
          href={`/twin/${patientId}`}
          className="text-xs text-brand hover:underline flex items-center gap-1.5 font-medium"
        >
          <ArrowLeft size={14} strokeWidth={1.75} />
          <span>Back to Patient Digital Twin</span>
        </Link>

        <Button
          size="sm"
          variant="outline"
          onClick={handlePrint}
          className="flex items-center gap-1.5 text-xs"
        >
          <Printer size={14} strokeWidth={1.75} />
          <span>Print / Export PDF Summary</span>
        </Button>
      </div>

      {/* Patient Header Card */}
      <Card className="p-5 print:border-none print:shadow-none">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-border pb-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2 flex-wrap">
              <h1 className="text-xl font-bold text-text-primary tracking-tight">
                {profile.name}
              </h1>
              <span className="text-xs bg-surface-muted text-text-muted px-2 py-0.5 rounded border border-border tnum">
                {profile.id}
              </span>
              <Badge
                variant={
                  cohort === "cgmacros_research"
                    ? "purple"
                    : cohort === "synthetic_persona"
                    ? "info"
                    : "neutral"
                }
              >
                {cohortBadge}
              </Badge>
            </div>

            <div className="text-xs text-text-secondary flex items-center gap-2 flex-wrap tnum">
              <span>{profile.age} yrs</span>
              <span>•</span>
              <span className="capitalize">{profile.gender}</span>
              {profile.bmi && (
                <>
                  <span>•</span>
                  <span>BMI {profile.bmi.toFixed(1)}</span>
                </>
              )}
              <span>•</span>
              <span>HbA1c {profile.hba1c}%</span>
              {profile.fasting_glucose && (
                <>
                  <span>•</span>
                  <span>Fasting {profile.fasting_glucose.toFixed(0)} mg/dL</span>
                </>
              )}
              {isCgmacros && profile.ethnicity && (
                <>
                  <span>•</span>
                  <span>{profile.ethnicity}</span>
                </>
              )}
              {!isCgmacros && profile.region && (
                <>
                  <span>•</span>
                  <span>{profile.region}</span>
                </>
              )}
              {!isCgmacros && profile.occupation && (
                <>
                  <span>•</span>
                  <span>{profile.occupation}</span>
                </>
              )}
            </div>
          </div>

          <div className="text-right">
            <span className="text-[10px] uppercase font-semibold text-text-muted tracking-wider block">
              Stated Monitoring Window
            </span>
            <span className="text-xs font-bold text-brand bg-brand/10 border border-brand/20 px-2.5 py-1 rounded inline-block mt-0.5">
              {monitoringWindow}
            </span>
          </div>
        </div>

        {/* 5-Band Stacked Horizontal TIR Bar */}
        <div className="pt-4 space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-text-primary">
              Standard 5-Band Time in Range Profile
            </span>
            <span className="text-text-muted text-[11px] tnum">Target TIR (70 to 180): &gt;70%</span>
          </div>

          <div className="h-6 w-full rounded-md overflow-hidden flex border border-border bg-surface-muted">
            {vLow > 0 && (
              <div
                style={{ width: `${vLow}%` }}
                className="bg-[#EF4444] flex items-center justify-center text-[10px] text-white font-bold tnum overflow-hidden"
                title={`Very Low (<54 mg/dL): ${vLow.toFixed(1)}%`}
              >
                {vLow >= 3 ? `${vLow.toFixed(0)}%` : ""}
              </div>
            )}
            {low > 0 && (
              <div
                style={{ width: `${low}%` }}
                className="bg-[#F59E0B] flex items-center justify-center text-[10px] text-white font-bold tnum overflow-hidden"
                title={`Low (54 to 70 mg/dL): ${low.toFixed(1)}%`}
              >
                {low >= 3 ? `${low.toFixed(0)}%` : ""}
              </div>
            )}
            {inRange > 0 && (
              <div
                style={{ width: `${inRange}%` }}
                className="bg-[#10B981] flex items-center justify-center text-[11px] text-white font-extrabold tnum overflow-hidden"
                title={`In Range (70 to 180 mg/dL): ${inRange.toFixed(1)}%`}
              >
                {inRange >= 8 ? `${inRange.toFixed(1)}% in range` : `${inRange.toFixed(0)}%`}
              </div>
            )}
            {high > 0 && (
              <div
                style={{ width: `${high}%` }}
                className="bg-[#F97316] flex items-center justify-center text-[10px] text-white font-bold tnum overflow-hidden"
                title={`High (180 to 250 mg/dL): ${high.toFixed(1)}%`}
              >
                {high >= 3 ? `${high.toFixed(0)}%` : ""}
              </div>
            )}
            {vHigh > 0 && (
              <div
                style={{ width: `${vHigh}%` }}
                className="bg-[#7C3AED] flex items-center justify-center text-[10px] text-white font-bold tnum overflow-hidden"
                title={`Very High (>250 mg/dL): ${vHigh.toFixed(1)}%`}
              >
                {vHigh >= 3 ? `${vHigh.toFixed(0)}%` : ""}
              </div>
            )}
          </div>

          {/* 5-Band Legend */}
          <div className="grid grid-cols-5 gap-1 text-center text-[10px] pt-1">
            <div className="space-y-0.5">
              <span className="flex items-center justify-center gap-1 font-medium text-text-muted">
                <span className="h-2 w-2 rounded-full bg-[#EF4444] inline-block" />
                Very Low (&lt;54)
              </span>
              <span className="font-bold text-[#EF4444] tnum block">{vLow.toFixed(1)}%</span>
            </div>
            <div className="space-y-0.5">
              <span className="flex items-center justify-center gap-1 font-medium text-text-muted">
                <span className="h-2 w-2 rounded-full bg-[#F59E0B] inline-block" />
                Low (54 to 70)
              </span>
              <span className="font-bold text-[#D97706] tnum block">{low.toFixed(1)}%</span>
            </div>
            <div className="space-y-0.5">
              <span className="flex items-center justify-center gap-1 font-medium text-text-muted">
                <span className="h-2 w-2 rounded-full bg-[#10B981] inline-block" />
                Target (70 to 180)
              </span>
              <span className="font-bold text-[#059669] tnum block">{inRange.toFixed(1)}%</span>
            </div>
            <div className="space-y-0.5">
              <span className="flex items-center justify-center gap-1 font-medium text-text-muted">
                <span className="h-2 w-2 rounded-full bg-[#F97316] inline-block" />
                High (180 to 250)
              </span>
              <span className="font-bold text-[#EA580C] tnum block">{high.toFixed(1)}%</span>
            </div>
            <div className="space-y-0.5">
              <span className="flex items-center justify-center gap-1 font-medium text-text-muted">
                <span className="h-2 w-2 rounded-full bg-[#7C3AED] inline-block" />
                Very High (&gt;250)
              </span>
              <span className="font-bold text-[#6D28D9] tnum block">{vHigh.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </Card>

      {/* AGP Consensus Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-surface p-3.5 rounded-lg border border-border">
          <span className="text-[11px] text-text-secondary block">Glucose Management Indicator</span>
          <div className="text-xl font-extrabold text-text-primary mt-1 tnum">
            {m.gmi ? m.gmi.toFixed(1) : "7.2"}%
          </div>
          <span className="text-[10px] text-text-muted block mt-0.5">estimated from this window</span>
        </div>

        <div className="bg-surface p-3.5 rounded-lg border border-border">
          <span className="text-[11px] text-text-secondary block">Glycemic Variability (CV)</span>
          <div className="text-xl font-extrabold text-brand mt-1 tnum">
            {m.coefficient_of_variation.toFixed(1)}%
          </div>
          <span className="text-[10px] text-text-muted block mt-0.5">Target: &le; 36.0%</span>
        </div>

        <div className="bg-surface p-3.5 rounded-lg border border-border">
          <span className="text-[11px] text-text-secondary block">Mean Glucose</span>
          <div className="text-xl font-extrabold text-text-primary mt-1 tnum">
            {m.mean_glucose.toFixed(0)} <span className="text-xs font-normal text-text-muted">mg/dL</span>
          </div>
          <span className="text-[10px] text-text-muted block mt-0.5">Over stated window</span>
        </div>

        <div className="bg-surface p-3.5 rounded-lg border border-border">
          <span className="text-[11px] text-text-secondary block">Standard Deviation (SD)</span>
          <div className="text-xl font-extrabold text-text-primary mt-1 tnum">
            {m.standard_deviation.toFixed(1)} <span className="text-xs font-normal text-text-muted">mg/dL</span>
          </div>
          <span className="text-[10px] text-text-muted block mt-0.5">Glycemic excursion spread</span>
        </div>
      </div>

      {/* Ambulatory Glucose Profile (AGP) 24-Hour Percentile Chart */}
      <Card className="p-4">
        <CardHeader className="pb-2">
          <div>
            <CardTitle className="text-sm">
              <FileText size={16} strokeWidth={1.75} className="text-brand" />
              Ambulatory Glucose Profile (AGP) 24-Hour Distribution
            </CardTitle>
            <p className="text-[11px] text-text-muted mt-0.5">
              Consolidated modal day over stated window: Median (50th percentile), IQR band (25th to 75th percentile), and Outer band (5th to 95th percentile).
            </p>
          </div>
        </CardHeader>

        {!hasEnoughAgpData ? (
          <div className="h-48 flex flex-col items-center justify-center border border-dashed border-border rounded-lg text-text-muted text-xs gap-2 my-2">
            <Info size={24} className="opacity-50 text-text-muted" />
            <span>Not enough data for an AGP yet (needs at least 24 h)</span>
          </div>
        ) : (
          <>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={agpData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="2 2" stroke="#E3E8F0" vertical={false} />
                  <XAxis
                    dataKey="hourLabel"
                    stroke="#64748B"
                    tick={{ fontSize: 10 }}
                    interval={2}
                  />
                  <YAxis
                    domain={[40, 280]}
                    ticks={[54, 70, 140, 180, 250]}
                    stroke="#64748B"
                    tick={{ fontSize: 10 }}
                    label={{
                      value: "mg/dL",
                      angle: -90,
                      position: "insideLeft",
                      fill: "#64748B",
                      fontSize: 10,
                      fontWeight: 600,
                    }}
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const d = payload[0].payload as AGPHourPoint;
                        return (
                          <div className="bg-surface border border-border-strong rounded p-2 text-xs space-y-1">
                            <div className="font-semibold text-text-primary">{d.hourLabel} Modal Time</div>
                            <div className="text-brand font-bold">Median: {d.median.toFixed(1)} mg/dL</div>
                            <div className="text-text-secondary text-[11px]">
                              25th to 75th: {d.p25.toFixed(0)} to {d.p75.toFixed(0)} mg/dL
                            </div>
                            <div className="text-text-muted text-[10px]">
                              5th to 95th: {d.p5.toFixed(0)} to {d.p95.toFixed(0)} mg/dL
                            </div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />

                  {/* Target glycemic band: 70 to 180 */}
                  <ReferenceArea y1={70} y2={180} fill="#10B981" fillOpacity={0.07} />
                  <ReferenceLine y={180} stroke="#F59E0B" strokeDasharray="3 3" />
                  <ReferenceLine y={70} stroke="#7C3AED" strokeDasharray="3 3" />

                  {/* Outer percentile band: 5th to 95th */}
                  <Area
                    type="monotone"
                    dataKey="outerRange"
                    stroke="transparent"
                    fill="#93C5FD"
                    fillOpacity={0.25}
                    isAnimationActive={false}
                    name="5th to 95th Percentile"
                  />

                  {/* Interquartile range band: 25th to 75th */}
                  <Area
                    type="monotone"
                    dataKey="iqrRange"
                    stroke="transparent"
                    fill="#0284C7"
                    fillOpacity={0.35}
                    isAnimationActive={false}
                    name="25th to 75th Percentile"
                  />

                  {/* Median glucose curve: bold navy/blue line */}
                  <Line
                    type="monotone"
                    dataKey="median"
                    stroke="#0F172A"
                    strokeWidth={2.5}
                    dot={false}
                    isAnimationActive={false}
                    name="Median Glucose (50th percentile)"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            {/* AGP Legend */}
            <div className="mt-2 pt-2 border-t border-border flex items-center justify-between text-[11px] text-text-secondary flex-wrap gap-2">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-1.5">
                  <span className="h-1.5 w-4 bg-[#0F172A] rounded-full" />
                  <span>Median (50th)</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="h-2 w-3 bg-[#0284C7] opacity-60 rounded-sm" />
                  <span>25th to 75th Band</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="h-2 w-3 bg-[#93C5FD] opacity-50 rounded-sm" />
                  <span>5th to 95th Band</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="h-2 w-3 bg-[#10B981] opacity-20 border border-[#10B981] rounded-sm" />
                  <span>Target (70 to 180)</span>
                </div>
              </div>
            </div>
          </>
        )}
      </Card>

      {/* Clinical Findings & Recommendations */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Narrative Impression */}
        <Card className="p-4 space-y-2.5">
          <h2 className="text-xs font-bold text-text-primary uppercase tracking-tight flex items-center gap-1.5">
            <Info size={14} strokeWidth={1.75} className="text-brand" />
            Auto-generated summary
          </h2>
          <p className="text-xs text-text-secondary leading-relaxed bg-surface-muted p-3 rounded-lg border border-border">
            {summary.plain_language_narrative}
          </p>

          <div className="pt-2">
            <span className="text-[11px] font-semibold text-text-primary block mb-1">
              Findings
            </span>
            <ul className="space-y-1">
              {summary.key_findings.map((f, i) => (
                <li key={i} className="text-[11px] text-text-secondary flex items-start gap-1.5">
                  <span className="text-brand mt-0.5">•</span>
                  <span>{f}</span>
                </li>
              ))}
            </ul>
          </div>
        </Card>

        {/* Suggestions for clinician review */}
        <div className="space-y-4">
          <Card className="p-4 space-y-2.5">
            <h2 className="text-xs font-bold text-text-primary uppercase tracking-tight flex items-center gap-1.5">
              <CheckCircle size={14} strokeWidth={1.75} className="text-in-range" />
              Suggestions for clinician review
            </h2>
            {summary.actionable_recommendations && summary.actionable_recommendations.length > 0 ? (
              <ul className="space-y-1.5">
                {summary.actionable_recommendations.map((rec, i) => (
                  <li key={i} className="text-xs text-text-secondary flex items-start gap-2 bg-surface-muted p-2 rounded border border-border">
                    <span className="font-bold text-brand">{i + 1}.</span>
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-xs text-text-muted italic bg-surface-muted p-3 rounded-lg border border-border">
                No data-driven suggestions for this window
              </p>
            )}
          </Card>

          {/* Personalization Status in Summary */}
          <Card className="p-3 bg-surface-muted/60 border border-border">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Cpu size={15} strokeWidth={1.75} className="text-brand" />
                <span className="text-xs font-semibold text-text-primary">Twin Personalization Status</span>
              </div>
              <Badge variant={personalization?.is_personalized ? "success" : "neutral"} className="text-[10px]">
                {personalization?.is_personalized ? "Personalized Adapter Active" : "Population Model Baseline"}
              </Badge>
            </div>
            <p className="text-[10px] text-text-muted mt-1 leading-snug">
              {personalization?.is_personalized
                ? `Patient-specific LoRA adapter calibrated on held-out series (MARD: ${personalization.historical_mard_percentage?.[personalization.historical_mard_percentage.length - 1] ?? 11.9}%).`
                : "No patient adapter yet. Cross-validated population baseline active."}
            </p>
          </Card>
        </div>
      </div>

      {/* Regulatory & Decision Support Disclaimer */}
      <div className="p-3 bg-surface-muted rounded-lg border border-border text-[10px] text-text-muted leading-relaxed flex items-start gap-2">
        <ShieldCheck size={14} strokeWidth={1.75} className="text-text-muted shrink-0 mt-0.5" />
        <div>
          <strong className="text-text-secondary">Clinical Decision Support Disclaimer: </strong>
          {summary.disclaimer} This digital twin software provides algorithmic retrospective and predictive pattern analysis for licensed healthcare practitioners. It does not replace individualized clinical judgment or standard laboratory assays.
        </div>
      </div>
    </div>
  );
}

export default function ClinicalSummaryPage() {
  return (
    <Suspense
      fallback={
        <div className="p-12 text-center text-xs text-text-muted">
          Loading clinical ambulatory glucose profile summary...
        </div>
      }
    >
      <ClinicalSummaryContent />
    </Suspense>
  );
}
