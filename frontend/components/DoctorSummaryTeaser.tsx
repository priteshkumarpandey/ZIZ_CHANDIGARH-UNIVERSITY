import React from "react";
import Link from "next/link";
import { FileText, ArrowRight } from "lucide-react";
import { ClinicalSummaryResponse } from "../lib/types";
import { Card, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";

interface DoctorSummaryTeaserProps {
  summary: ClinicalSummaryResponse | null;
  onOpenModal?: () => void;
}

export const DoctorSummaryTeaser: React.FC<DoctorSummaryTeaserProps> = ({
  summary,
  onOpenModal,
}) => {
  if (!summary) return null;

  const m = summary.metrics;
  const patientId = summary.patient_id || "IND_PAT_001";

  return (
    <Card className="w-full">
      <CardHeader>
        <div>
          <CardTitle>
            <FileText size={16} strokeWidth={1.75} className="text-brand" />
            Glucose profile summary
          </CardTitle>
          <p className="text-xs text-text-secondary mt-0.5">
            Consolidated AGP consensus metrics and synthesized physician impressions
          </p>
        </div>

        <Link href={`/summary/${patientId}`}>
          <Button size="sm" variant="outline" className="flex items-center gap-1.5">
            <span>Full summary</span>
            <ArrowRight size={16} strokeWidth={1.75} />
          </Button>
        </Link>
      </CardHeader>

      <div className="space-y-4">
        {/* Core Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <div className="bg-surface-muted p-3 rounded-lg border border-border">
            <span className="text-text-secondary">
              {summary.monitoring_window_text
                ? `Time in Range (${summary.monitoring_window_text})`
                : "Time in Range (last 24 hours)"}
            </span>
            <div className="text-lg font-bold text-in-range mt-0.5 tnum">
              {m.time_in_range_percentage.toFixed(1)}%
            </div>
            <span className="text-[10px] text-text-muted">Target: 70 to 180 mg/dL</span>
          </div>

          <div className="bg-surface-muted p-3 rounded-lg border border-border">
            <span className="text-text-secondary">Time Above Range (&gt;180)</span>
            <div className="text-lg font-bold text-high mt-0.5 tnum">
              {m.time_above_range_percentage.toFixed(1)}%
            </div>
            <span className="text-[10px] text-text-muted">Target &lt;25%</span>
          </div>

          <div className="bg-surface-muted p-3 rounded-lg border border-border">
            <span className="text-text-secondary">Time Below Range (&lt;70)</span>
            <div className="text-lg font-bold text-low mt-0.5 tnum">
              {m.time_below_range_percentage.toFixed(1)}%
            </div>
            <span className="text-[10px] text-text-muted">Target &lt;4%</span>
          </div>

          <div className="bg-surface-muted p-3 rounded-lg border border-border">
            <span className="text-text-secondary">Glucose Mgmt Indicator</span>
            <div className="text-lg font-bold text-text-primary mt-0.5 tnum">
              {m.gmi ? m.gmi.toFixed(1) : "7.2"}%
            </div>
            <span className="text-[10px] text-text-muted">Estimated HbA1c</span>
          </div>
        </div>

        {/* Narrative Excerpt */}
        {summary.plain_language_narrative && (
          <div className="text-xs text-text-secondary bg-surface-muted p-3 rounded-lg border border-border leading-relaxed">
            <strong className="text-text-primary block mb-1">Impression Excerpt:</strong>
            {summary.plain_language_narrative}
          </div>
        )}
      </div>
    </Card>
  );
};
