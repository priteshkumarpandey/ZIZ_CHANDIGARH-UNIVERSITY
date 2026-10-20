"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { Users, MapPin, Pill, ArrowRight } from "lucide-react";
import { getPatients } from "../../lib/api";
import { PatientProfile, getCohortBadgeText, getPatientCohort } from "../../lib/types";
import { Badge } from "../../components/ui/badge";

import { useGlucoTwinStore } from "../../lib/store";

export default function PatientsPage() {
  const { setPatients: setStorePatients } = useGlucoTwinStore();
  const [patients, setPatients] = useState<PatientProfile[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    getPatients()
      .then((data) => {
        setPatients(data);
        setStorePatients(data);
      })
      .catch((err) => console.error("Error loading personas", err))
      .finally(() => setLoading(false));
  }, [setStorePatients]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-bold text-text-primary flex items-center gap-2">
          <Users size={20} strokeWidth={1.75} className="text-brand" />
          Patient Persona Directory
        </h1>
        <p className="text-xs text-text-secondary mt-1">
          Synthetic personas and de-identified clinical research participants representing diverse glycemic phenotypes.
        </p>
      </div>

      {loading ? (
        <div className="p-8 text-center text-xs text-text-muted">Loading patient directory from API...</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {patients.map((p) => {
            const isSynthetic = Boolean(p.is_synthetic);
            const cohort = getPatientCohort(p.id);
            const cohortBadge = getCohortBadgeText(p.id);

            return (
              <div key={p.id} className="bg-surface border border-border rounded-card p-5 space-y-3">
                <div className="flex items-start justify-between">
                  <div>
                    <h2 className="text-base font-semibold text-text-primary tracking-tight">
                      {isSynthetic ? p.name : p.id}
                    </h2>
                    <p className="text-xs text-text-secondary tnum">
                      {p.age} years • {p.gender}
                      {isSynthetic && p.occupation ? ` • ${p.occupation}` : ""}
                    </p>
                  </div>
                  <Badge variant={p.hba1c > 8.0 ? "danger" : "warning"}>
                    <span className="tnum">HbA1c {p.hba1c}%</span>
                  </Badge>
                </div>

                <div>
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

                {isSynthetic ? (
                  /* Synthetic personas & demo patients: show regional dietary habits & postprandial values */
                  <div className="grid grid-cols-2 gap-2 text-xs text-text-secondary">
                    {p.region && p.dietary_habit && (
                      <div className="flex items-center gap-1.5">
                        <MapPin size={16} strokeWidth={1.75} className="text-brand shrink-0" />
                        <span>{p.region} ({p.dietary_habit})</span>
                      </div>
                    )}
                    <div>
                      <span className="text-text-muted">BMI: </span>
                      <span className="font-medium text-text-primary tnum">{p.bmi} kg/m²</span>
                    </div>
                    <div>
                      <span className="text-text-muted">Fasting: </span>
                      <span className="font-medium text-text-primary tnum">{p.fasting_glucose} mg/dL</span>
                    </div>
                    {p.postprandial_glucose != null && (
                      <div>
                        <span className="text-text-muted">Postprandial: </span>
                        <span className="font-medium text-text-primary tnum">{p.postprandial_glucose} mg/dL</span>
                      </div>
                    )}
                  </div>
                ) : (
                  /* Real research participants: show ONLY fields that exist in their data (age, sex, BMI, HbA1c, fasting glucose, ethnicity) */
                  <div className="grid grid-cols-2 gap-2 text-xs text-text-secondary">
                    <div>
                      <span className="text-text-muted">BMI: </span>
                      <span className="font-medium text-text-primary tnum">{p.bmi} kg/m²</span>
                    </div>
                    <div>
                      <span className="text-text-muted">Fasting Glucose: </span>
                      <span className="font-medium text-text-primary tnum">{p.fasting_glucose} mg/dL</span>
                    </div>
                    <div>
                      <span className="text-text-muted">Participant ID: </span>
                      <span className="font-medium text-text-primary tnum">{p.id}</span>
                    </div>
                    {p.ethnicity && (
                      <div>
                        <span className="text-text-muted">Ethnicity: </span>
                        <span className="font-medium text-text-primary">{p.ethnicity}</span>
                      </div>
                    )}
                  </div>
                )}

                {isSynthetic && p.active_medications && p.active_medications.length > 0 && (
                  <div className="text-xs text-text-secondary border-t border-border pt-2.5 space-y-1">
                    <div className="flex items-center gap-1.5 text-high">
                      <Pill size={16} strokeWidth={1.75} className="shrink-0" />
                      <span className="font-medium">Active Medications:</span>
                    </div>
                    <p className="text-text-secondary pl-5">{p.active_medications.join(", ")}</p>
                  </div>
                )}

                {p.phenotype_notes && (
                  <p className="text-xs text-text-muted bg-surface-muted p-2.5 rounded-lg border border-border leading-relaxed">
                    {p.phenotype_notes}
                  </p>
                )}

                <div className="pt-2 flex justify-end">
                  <Link
                    href={`/twin/${p.id}`}
                    className="inline-flex items-center gap-1.5 text-xs font-semibold text-brand hover:text-brand-line transition-colors"
                  >
                    Launch Digital Twin
                    <ArrowRight size={16} strokeWidth={1.75} />
                  </Link>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

