import React from "react";
import { User, MapPin, Pill } from "lucide-react";
import { PatientProfile, getCohortBadgeText, getPatientCohort } from "../lib/types";
import { Badge } from "./ui/badge";

interface PatientSelectorProps {
  patients: PatientProfile[];
  selectedPatient: PatientProfile | null;
  onSelect: (patient: PatientProfile) => void;
}

export const PatientSelector: React.FC<PatientSelectorProps> = ({
  patients,
  selectedPatient,
  onSelect,
}) => {
  return (
    <div className="bg-surface border border-border rounded-card p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-text-secondary flex items-center gap-1.5">
          <User size={16} strokeWidth={1.75} className="text-brand" />
          Patient Directory
        </h2>
        <span className="text-[11px] text-text-muted tnum">{patients.length} Registered Patients</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        {patients.map((p) => {
          const isSelected = selectedPatient?.id === p.id;
          const cohort = getPatientCohort(p.id);
          const badgeText = getCohortBadgeText(p.id);

          return (
            <button
              key={p.id}
              onClick={() => onSelect(p)}
              className={`text-left p-3.5 rounded-lg border transition-colors ${
                isSelected
                  ? "bg-brand-subtle border-brand ring-1 ring-inset ring-brand"
                  : "bg-surface border-border hover:border-border-strong hover:bg-surface-muted"
              }`}
            >
              <div className="flex items-start justify-between gap-1 mb-1">
                <div>
                  <h3 className="text-sm font-semibold text-text-primary tracking-tight">{p.name}</h3>
                  <p className="text-xs text-text-secondary tnum">
                    {p.age} yrs • {p.gender} • {p.occupation}
                  </p>
                </div>
                <Badge variant={p.hba1c > 8.0 ? "danger" : "warning"}>
                  <span className="tnum">HbA1c {p.hba1c}%</span>
                </Badge>
              </div>

              <div className="my-2">
                <Badge variant={cohort === "cgmacros_research" ? "purple" : cohort === "synthetic_persona" ? "info" : "neutral"}>
                  {badgeText}
                </Badge>
              </div>

              <div className="space-y-1 text-xs text-text-secondary">
                <div className="flex items-center gap-1.5">
                  <MapPin size={16} strokeWidth={1.75} className="text-brand shrink-0" />
                  <span>{p.region}</span>
                  <span className="text-text-muted">•</span>
                  <span className="truncate">{p.dietary_habit}</span>
                </div>
                {p.active_medications && p.active_medications.length > 0 && (
                  <div className="flex items-center gap-1.5 truncate">
                    <Pill size={16} strokeWidth={1.75} className="text-high shrink-0" />
                    <span className="truncate">{p.active_medications.join(", ")}</span>
                  </div>
                )}
              </div>

              {p.phenotype_notes && (
                <p className="mt-2 text-[11px] text-text-muted line-clamp-2 border-t border-border pt-2 leading-relaxed">
                  {p.phenotype_notes}
                </p>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
};

