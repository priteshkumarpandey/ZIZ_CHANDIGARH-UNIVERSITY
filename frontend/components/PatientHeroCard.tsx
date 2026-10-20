import React from "react";
import { ArrowUp, ArrowDown, ArrowRight } from "lucide-react";
import { PatientProfile, TwinStateResponse, CGMReading, getCohortBadgeText, getPatientCohort } from "../lib/types";
import { Card } from "./ui/card";
import { Badge } from "./ui/badge";

interface PatientHeroCardProps {
  profile: PatientProfile;
  twinState: TwinStateResponse | null;
  servingFold?: number;
  cgmHistory?: CGMReading[];
}

export const PatientHeroCard: React.FC<PatientHeroCardProps> = ({
  profile,
  twinState,
  servingFold,
  cgmHistory,
}) => {
  const cohort = getPatientCohort(profile.id);
  const cohortBadge = getCohortBadgeText(profile.id);
  const isCgmacros = cohort === "cgmacros_research";
  const currentGlucose = twinState?.current_glucose || profile.fasting_glucose || 120.0;
  const trendArrow = twinState?.trend_arrow || "Flat";

  // Real TIR calculation over displayed window
  const inRangeCount =
    cgmHistory && cgmHistory.length > 0
      ? cgmHistory.filter((r) => r.glucose_mgdl >= 70 && r.glucose_mgdl <= 180).length
      : 0;
  const tirPercent =
    cgmHistory && cgmHistory.length > 0
      ? Math.round((inRangeCount / cgmHistory.length) * 100)
      : 78;
  const radius = 24;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (tirPercent / 100) * circumference;

  const renderTrendIcon = () => {
    switch (trendArrow) {
      case "Rising quickly":
      case "Rising":
        return <ArrowUp size={20} strokeWidth={1.75} className="text-severe" />;
      case "Falling quickly":
      case "Falling":
        return <ArrowDown size={20} strokeWidth={1.75} className="text-low" />;
      default:
        return <ArrowRight size={20} strokeWidth={1.75} className="text-in-range" />;
    }
  };

  const foldText =
    servingFold !== undefined
      ? isCgmacros
        ? `This participant was held out when the serving model was trained (fold ${servingFold}).`
        : `Synthetic demo patient, never part of any training data. Served by the fold ${servingFold} model.`
      : null;

  return (
    <Card className="p-4">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* Patient Identity & Badges */}
        <div className="space-y-1.5 flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h1 className="text-lg font-bold text-text-primary tracking-tight truncate">{profile.name}</h1>
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

          {/* Honest clinical demographics: only real fields for CGMacros */}
          <div className="text-xs text-text-secondary tnum flex items-center gap-2 flex-wrap">
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
            {/* Ethnicity if present for real participants */}
            {isCgmacros && profile.ethnicity && (
              <>
                <span>•</span>
                <span>{profile.ethnicity}</span>
              </>
            )}
            {/* Region & Occupation shown ONLY for synthetic personas / demo patients */}
            {!isCgmacros && profile.occupation && (
              <>
                <span>•</span>
                <span>{profile.occupation}</span>
              </>
            )}
            {!isCgmacros && profile.region && (
              <>
                <span>•</span>
                <span>{profile.region}</span>
              </>
            )}
          </div>

          {profile.phenotype_notes && !isCgmacros && (
            <p className="text-[11px] text-text-muted line-clamp-1 leading-snug">
              {profile.phenotype_notes}
            </p>
          )}

          {foldText && (
            <div className="text-[11px] text-text-muted italic">
              {foldText}
            </div>
          )}
        </div>

        {/* Current Glucose Telemetry & TIR Ring */}
        <div className="flex items-center gap-5 border-t md:border-t-0 md:border-l border-border pt-3 md:pt-0 md:pl-5 shrink-0">
          {/* Current Glucose Hero */}
          <div className="space-y-0.5">
            <span className="text-[10px] uppercase font-semibold text-text-secondary tracking-wider block">
              Current Telemetry
            </span>
            <div className="flex items-center gap-1.5">
              <span
                data-testid="hero-current-glucose"
                className="text-4xl font-extrabold text-text-primary tracking-tight tnum leading-none"
              >
                {currentGlucose.toFixed(1)}
              </span>
              <div className="flex flex-col items-center">
                {renderTrendIcon()}
                <span className="text-[9px] text-text-muted uppercase font-bold tracking-tight">mg/dL</span>
              </div>
            </div>
            <span className="text-[10px] text-text-muted block">
              Trend: {trendArrow}
            </span>
          </div>

          {/* SVG TIR Ring Gauge */}
          <div className="flex items-center gap-2 pl-1">
            <div className="relative w-12 h-12 flex items-center justify-center shrink-0">
              <svg className="w-12 h-12 transform -rotate-90" viewBox="0 0 56 56">
                <circle
                  cx="28"
                  cy="28"
                  r={radius}
                  stroke="#E3E8F0"
                  strokeWidth="4"
                  fill="transparent"
                />
                <circle
                  cx="28"
                  cy="28"
                  r={radius}
                  stroke="#10B981"
                  strokeWidth="4"
                  strokeDasharray={circumference}
                  strokeDashoffset={strokeDashoffset}
                  strokeLinecap="round"
                  fill="transparent"
                />
              </svg>
              <div className="absolute text-center">
                <span className="text-[11px] font-bold text-text-primary tnum">{tirPercent}%</span>
              </div>
            </div>
            <div className="space-y-0.5">
              <span className="text-[11px] font-semibold text-text-primary block leading-tight">Time in Range (last 6 h)</span>
              <span className="text-[9px] text-text-muted block">70 to 180 mg/dL</span>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};
