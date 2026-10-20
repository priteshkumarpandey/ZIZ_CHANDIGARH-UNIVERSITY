"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { Users, FlaskConical, TrendingUp, ShieldCheck, ArrowRight } from "lucide-react";
import { getPatients } from "../lib/api";
import { PatientProfile, getCohortBadgeText, getPatientCohort } from "../lib/types";
import { Badge } from "../components/ui/badge";

export default function HomePage() {
  const [patients, setPatients] = useState<PatientProfile[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    getPatients()
      .then((data) => setPatients(data))
      .catch((err) => console.error("Could not load personas", err))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      {/* Welcome Hero Banner */}
      <div className="bg-surface border border-border rounded-card p-6 relative overflow-hidden">
        <div className="max-w-3xl space-y-3">
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-brand-subtle border border-brand/20 text-xs text-brand font-semibold">
            <ShieldCheck size={16} strokeWidth={1.75} className="text-brand" />
            Decision Support System
          </div>
          <h1 className="text-2xl font-bold text-text-primary tracking-tight">
            GlucoTwin Doctor Dashboard
          </h1>
          <p className="text-xs sm:text-sm text-text-secondary leading-relaxed">
            Multimodal physiological digital twin for proactive Type 2 Diabetes care. Continuously fuses wearable CGM streams with clinical priors to anticipate adverse glycemic spikes and nocturnal hypoglycemic events.
          </p>
          <div className="pt-2 flex flex-wrap gap-3">
            <Link
              href="/twin/IND_PAT_001"
              className="inline-flex items-center gap-2 bg-brand hover:bg-brand-line text-white text-xs font-semibold px-4 py-2 rounded-lg border border-brand transition-colors"
            >
              Launch Rajesh Sharma Twin
              <ArrowRight size={16} strokeWidth={1.75} />
            </Link>
            <Link
              href="/what-if"
              className="inline-flex items-center gap-2 bg-surface hover:bg-surface-muted text-text-primary text-xs font-medium px-4 py-2 rounded-lg border border-border transition-colors"
            >
              <FlaskConical size={16} strokeWidth={1.75} className="text-high" />
              What-If Counterfactual Simulator
            </Link>
          </div>
        </div>
      </div>

      {/* Feature Pillars */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-surface border border-border rounded-card p-5 space-y-2">
          <div className="h-8 w-8 rounded-lg bg-brand-subtle text-brand flex items-center justify-center border border-brand/20">
            <TrendingUp size={16} strokeWidth={1.75} />
          </div>
          <h2 className="text-sm font-semibold text-text-primary">Multi-Horizon Forecasting</h2>
          <p className="text-xs text-text-secondary leading-relaxed">
            30, 60, and 120-minute predictions with multi-quantile uncertainty fan (q10, q50, q90) and operating point validation.
          </p>
        </div>

        <div className="bg-surface border border-border rounded-card p-5 space-y-2">
          <div className="h-8 w-8 rounded-lg bg-high-subtle text-high flex items-center justify-center border border-high/20">
            <FlaskConical size={16} strokeWidth={1.75} />
          </div>
          <h2 className="text-sm font-semibold text-text-primary">Indian Meal What-If Engine</h2>
          <p className="text-xs text-text-secondary leading-relaxed">
            Physiology-constrained simulation calibrated against Indian meal library with approximate nutrient values.
          </p>
        </div>

        <div className="bg-surface border border-border rounded-card p-5 space-y-2">
          <div className="h-8 w-8 rounded-lg bg-low-subtle text-low flex items-center justify-center border border-low/20">
            <Users size={16} strokeWidth={1.75} />
          </div>
          <h2 className="text-sm font-semibold text-text-primary">Patient Persona Library</h2>
          <p className="text-xs text-text-secondary leading-relaxed">
            Calibrated patient profiles with diverse regional dietary habits, medications, and glycemic phenotypes.
          </p>
        </div>
      </div>

      {/* Patient Directory Grid */}
      <div className="bg-surface border border-border rounded-card p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-border pb-3">
          <h2 className="text-sm font-semibold text-text-primary flex items-center gap-2">
            <Users size={16} strokeWidth={1.75} className="text-brand" />
            Registered Patients
          </h2>
          <Link href="/patients" className="text-xs text-brand hover:underline font-medium">
            View All ({patients.length}) &rarr;
          </Link>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {patients.map((p) => {
            const cohort = getPatientCohort(p.id);
            const cohortBadge = getCohortBadgeText(p.id);

            return (
              <Link
                key={p.id}
                href={`/twin/${p.id}`}
                className="p-3.5 rounded-lg bg-surface border border-border hover:border-brand hover:bg-brand-subtle/30 transition-colors text-left block space-y-2"
              >
                <div className="flex justify-between items-start">
                  <span className="text-xs font-semibold text-text-primary">{p.name}</span>
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
                <p className="text-[11px] text-text-secondary tnum">
                  {p.age}y • {p.gender}
                  {p.region ? ` • ${p.region}` : p.bmi ? ` • BMI ${p.bmi.toFixed(1)}` : ""}
                </p>
                <p className="text-[10px] text-text-muted truncate">
                  Meds: {p.active_medications && p.active_medications.length > 0 ? p.active_medications.join(", ") : "None reported"}
                </p>
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
}

