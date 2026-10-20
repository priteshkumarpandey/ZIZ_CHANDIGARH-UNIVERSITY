import React from "react";
import { X, CheckCircle, AlertTriangle, Stethoscope, Shield } from "lucide-react";
import { ClinicalSummaryResponse } from "../lib/types";
import { useGlucoTwinStore } from "../lib/store";
import { Button } from "./ui/button";

interface ClinicalSummaryModalProps {
  summary: ClinicalSummaryResponse | null;
}

export const ClinicalSummaryModal: React.FC<ClinicalSummaryModalProps> = ({ summary }) => {
  const { isSummaryModalOpen, setIsSummaryModalOpen } = useGlucoTwinStore();

  if (!isSummaryModalOpen || !summary) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4">
      <div className="bg-surface border border-border-strong rounded-card max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-5 border-b border-border flex items-center justify-between sticky top-0 bg-surface z-10">
          <div className="flex items-center gap-2.5">
            <div className="h-8 w-8 rounded-lg bg-brand-subtle text-brand flex items-center justify-center border border-brand/30">
              <Stethoscope size={16} strokeWidth={1.75} />
            </div>
            <div>
              <h2 className="text-base font-bold text-text-primary tracking-tight">Physician Clinical Summary</h2>
              <p className="text-xs text-text-secondary tnum">
                Patient: {summary.patient_name} ({summary.patient_id})
              </p>
            </div>
          </div>

          <button
            onClick={() => setIsSummaryModalOpen(false)}
            className="text-text-muted hover:text-text-primary transition-colors p-1.5 rounded-lg hover:bg-surface-muted"
          >
            <X size={20} strokeWidth={1.75} />
          </button>
        </div>

        <div className="p-6 space-y-5">
          <div className="bg-surface-muted p-4 rounded-lg border border-border">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2">
              Clinical Narrative Synthesis
            </h3>
            <p className="text-xs text-text-primary leading-relaxed">
              {summary.plain_language_narrative}
            </p>
          </div>

          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2.5 flex items-center gap-1.5">
              <AlertTriangle size={16} strokeWidth={1.75} className="text-high" />
              Key Glycemic Observations
            </h3>
            <ul className="space-y-2">
              {summary.key_findings.map((finding, idx) => (
                <li
                  key={idx}
                  className="text-xs text-text-secondary bg-surface-muted border border-border p-2.5 rounded-lg flex items-start gap-2"
                >
                  <span className="h-1.5 w-1.5 rounded-full bg-high mt-1.5 shrink-0" />
                  <span>{finding}</span>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-2.5 flex items-center gap-1.5">
              <CheckCircle size={16} strokeWidth={1.75} className="text-in-range" />
              Suggestions for clinician review
            </h3>
            <ul className="space-y-2">
              {summary.actionable_recommendations.map((rec, idx) => (
                <li
                  key={idx}
                  className="text-xs text-text-secondary bg-green-subtle border border-green/30 p-2.5 rounded-lg flex items-start gap-2"
                >
                  <span className="h-1.5 w-1.5 rounded-full bg-in-range mt-1.5 shrink-0" />
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="bg-surface-muted border border-border p-3.5 rounded-lg flex items-start gap-2.5 text-xs text-text-secondary">
            <Shield size={16} strokeWidth={1.75} className="shrink-0 text-brand mt-0.5" />
            <div>
              <span className="font-semibold text-text-primary">Clinical Guidance: </span>
              <span>{summary.disclaimer || "Decision support, not a medical device"}</span>
            </div>
          </div>
        </div>

        <div className="p-4 border-t border-border flex justify-end bg-surface-muted">
          <Button onClick={() => setIsSummaryModalOpen(false)}>
            Close Dossier
          </Button>
        </div>
      </div>
    </div>
  );
};

