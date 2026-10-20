import React, { useState } from "react";
import { AlertCircle, AlertTriangle, ShieldCheck, HelpCircle, Activity, ChevronDown, ChevronUp } from "lucide-react";
import { ForecastResponse } from "../lib/types";
import { Card, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";

interface AlertsPanelProps {
  forecast: ForecastResponse | null;
  currentGlucose?: number;
}

export const AlertsPanel: React.FC<AlertsPanelProps> = ({ forecast, currentGlucose }) => {
  const [evidenceExpanded, setEvidenceExpanded] = useState<Record<number, boolean>>({});
  const [riskListExpanded, setRiskListExpanded] = useState<boolean>(false);

  if (!forecast || !forecast.forecast_horizons) {
    return (
      <Card className="p-3">
        <CardHeader className="pb-1">
          <CardTitle className="text-xs">
            <ShieldCheck size={16} strokeWidth={1.75} className="text-in-range" />
            Adverse Glycemic Events and Risk Estimates
          </CardTitle>
        </CardHeader>
        <p className="text-xs text-text-muted">Awaiting forecast data...</p>
      </Card>
    );
  }

  const effectiveCurrentGlucose =
    currentGlucose !== undefined ? currentGlucose : forecast.current_glucose;

  const isCurrentlyHyper =
    typeof effectiveCurrentGlucose === "number" && effectiveCurrentGlucose >= 180.0;
  const isCurrentlyHypo =
    typeof effectiveCurrentGlucose === "number" && effectiveCurrentGlucose <= 70.0;

  const isAlertUsable = (h: any) => {
    if (!h.alert_usable) return false;
    if (typeof h.alert_usable === "boolean") return h.alert_usable;
    if (typeof h.alert_usable === "object") {
      const ev = (h.predicted_event || "").toLowerCase();
      if (ev.includes("hyperglycemia_level_2") && h.alert_usable.severe_hyperglycemia !== undefined) {
        return Boolean(h.alert_usable.severe_hyperglycemia);
      }
      if (ev.includes("hyper") && h.alert_usable.hyperglycemia !== undefined) {
        return Boolean(h.alert_usable.hyperglycemia);
      }
      if (ev.includes("hypo") && h.alert_usable.hypoglycemia !== undefined) {
        return Boolean(h.alert_usable.hypoglycemia);
      }
      if (ev.includes("spike") && h.alert_usable.spike !== undefined) {
        return Boolean(h.alert_usable.spike);
      }
      return Object.values(h.alert_usable).some((v) => Boolean(v));
    }
    return false;
  };

  const horizons = forecast.forecast_horizons;

  // Filter for actionable alerts strictly on event ONSET
  // If reading has already crossed threshold, do not fire new alert (it is an event in progress)
  const onsetAlerts = horizons.filter((h) => {
    if (!isAlertUsable(h) || h.predicted_event === "none") return false;
    const ev = (h.predicted_event || "").toLowerCase();
    if (ev.includes("hyper") && isCurrentlyHyper) {
      return false; // Event in progress, not onset
    }
    if (ev.includes("hypo") && isCurrentlyHypo) {
      return false; // Event in progress, not onset
    }
    return true;
  });

  // At most two alert cards shown
  const displayedOnsetAlerts = onsetAlerts.slice(0, 2);

  const riskEstimates = horizons.filter(
    (h) => !isAlertUsable(h) && ((h.spike_risk_probability || 0) > 0.05 || (h.hypo_risk_probability || 0) > 0.05)
  );

  const toggleEvidence = (horizonMin: number) => {
    setEvidenceExpanded((prev) => ({
      ...prev,
      [horizonMin]: !prev[horizonMin],
    }));
  };

  const ruleBasedAlerts = [...(forecast.rule_based_alerts || [])];
  if (ruleBasedAlerts.length === 0 && effectiveCurrentGlucose !== undefined) {
    if (effectiveCurrentGlucose < 54.0) {
      ruleBasedAlerts.push({
        alert_type: "urgent_low",
        severity: "critical",
        title: "Urgent Low Glucose (<54 mg/dL)",
        message: `Observed glucose is ${effectiveCurrentGlucose.toFixed(1)} mg/dL, crossing below the critical 54 mg/dL safety floor.`,
        observed_glucose: effectiveCurrentGlucose,
        threshold_mgdl: 54.0,
        is_rule_based: true,
        provenance: "Rule-based alert (observed, not predicted)",
      });
    } else if (effectiveCurrentGlucose < 70.0) {
      ruleBasedAlerts.push({
        alert_type: "low",
        severity: "warning",
        title: "Low Glucose (<70 mg/dL)",
        message: `Observed glucose is ${effectiveCurrentGlucose.toFixed(1)} mg/dL, crossing below the 70 mg/dL safety floor.`,
        observed_glucose: effectiveCurrentGlucose,
        threshold_mgdl: 70.0,
        is_rule_based: true,
        provenance: "Rule-based alert (observed, not predicted)",
      });
    }
  }

  return (
    <Card className="p-3">
      <CardHeader className="pb-1.5">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-xs">
            <AlertCircle size={15} strokeWidth={1.75} className="text-brand" />
            Adverse Events & Risk Estimates
          </CardTitle>
          <div className="flex items-center gap-1.5 flex-wrap">
            {isCurrentlyHyper && (
              <Badge variant="warning">
                Currently above 180 mg/dL
              </Badge>
            )}
            {isCurrentlyHypo && (
              <Badge variant="danger">
                Currently below 70 mg/dL
              </Badge>
            )}
            {ruleBasedAlerts.length > 0 && (
              <Badge variant="danger">
                {ruleBasedAlerts.length} Safety Alert{ruleBasedAlerts.length > 1 ? "s" : ""}
              </Badge>
            )}
            {displayedOnsetAlerts.length > 0 ? (
              <Badge variant="danger">
                {displayedOnsetAlerts.length} Actionable Alert{displayedOnsetAlerts.length > 1 ? "s" : ""}
              </Badge>
            ) : !isCurrentlyHyper && !isCurrentlyHypo && ruleBasedAlerts.length === 0 ? (
              <Badge variant="success">No Actionable Event</Badge>
            ) : null}
          </div>
        </div>
      </CardHeader>

      <div className="space-y-2 pt-0.5">
        {/* Rule-based Safety Floor Alerts (Observed, Not Predicted) */}
        {ruleBasedAlerts.length > 0 && (
          <div className="space-y-1.5">
            {ruleBasedAlerts.map((alert, idx) => {
              const isUrgent = alert.severity === "critical" || alert.alert_type === "urgent_low";
              return (
                <div
                  key={`safety-rule-${idx}`}
                  className={`rounded-lg border p-2.5 space-y-1.5 text-xs ${
                    isUrgent
                      ? "border-red bg-red-subtle/80 text-red"
                      : "border-amber/80 bg-amber-subtle/80 text-text-primary"
                  }`}
                >
                  <div className="flex items-center justify-between gap-1.5">
                    <div className="flex items-center gap-1.5 font-bold">
                      <AlertTriangle
                        size={15}
                        strokeWidth={2}
                        className={isUrgent ? "text-severe shrink-0" : "text-amber shrink-0"}
                      />
                      <span className="text-xs">{alert.title}</span>
                    </div>
                    <Badge variant={isUrgent ? "danger" : "warning"} className="text-[10px] shrink-0 font-medium">
                      {alert.provenance || "Rule-based alert (observed, not predicted)"}
                    </Badge>
                  </div>
                  <p className="text-[11px] text-text-secondary leading-snug">
                    {alert.message}
                  </p>
                  <div className="flex items-center justify-between text-[10px] text-text-muted tnum pt-0.5 border-t border-border/50">
                    <span>
                      Observed: <strong className="text-text-primary">{alert.observed_glucose.toFixed(1)} mg/dL</strong>{" "}
                      (Threshold: {alert.threshold_mgdl.toFixed(0)} mg/dL)
                    </span>
                    {alert.rate_mgdl_per_min !== undefined && alert.rate_mgdl_per_min !== null && (
                      <span>
                        Rate: <strong className="text-text-primary">{alert.rate_mgdl_per_min.toFixed(1)} mg/dL/min</strong>
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Event In Progress Status (when currently above 180 or below 70) */}
        {isCurrentlyHyper && (
          <div className="rounded border border-amber/30 bg-amber-subtle/50 p-2 text-xs text-text-secondary flex items-start gap-1.5">
            <Activity size={14} strokeWidth={1.75} className="text-amber shrink-0 mt-0.5" />
            <div className="space-y-0.5">
              <span className="font-semibold text-text-primary block text-[11px]">
                Excursion in Progress ({effectiveCurrentGlucose?.toFixed(0)} mg/dL)
              </span>
              <p className="text-[10px] text-text-secondary">
                Patient reading already exceeds 180 mg/dL. Alert cards trigger strictly at event onset.
              </p>
            </div>
          </div>
        )}

        {/* Actionable Alerts strictly for event onset (at most two shown, compact format) */}
        <div className="space-y-1.5">
          {displayedOnsetAlerts.length === 0 ? (
            <div className="rounded border border-border bg-surface-muted p-2 text-xs text-text-secondary flex items-center gap-2">
              <ShieldCheck size={14} strokeWidth={1.75} className="text-in-range shrink-0" />
              <span className="text-[11px]">
                {isCurrentlyHyper
                  ? "No additional adverse onset predicted outside ongoing excursion."
                  : ruleBasedAlerts.length > 0
                  ? "Predictive horizons normal. Acute safety alerts managed by safety floor rules above."
                  : "All predictive horizons are within safe operating limits."}
              </span>
            </div>
          ) : (
            displayedOnsetAlerts.map((h) => {
              const evKey = (h.predicted_event || "").toLowerCase().includes("hypo")
                ? "hypoglycemia"
                : (h.predicted_event || "").toLowerCase().includes("severe")
                ? "severe_hyperglycemia"
                : "hyperglycemia";
              const rawMetrics = h.operating_metrics as Record<string, any> | undefined;
              const metrics = (rawMetrics?.[evKey] || rawMetrics?.hyperglycemia) as
                | { recall: number; precision: number; false_alerts_per_patient_day: number }
                | undefined;

              const isHypo = evKey.includes("hypo");
              const eventTitle = isHypo
                ? `Hypoglycemia risk in the next ${h.horizon_minutes} min`
                : evKey.includes("severe")
                ? `Severe hyperglycemia risk in the next ${h.horizon_minutes} min`
                : `Hyperglycemia risk in the next ${h.horizon_minutes} min`;

              const riskProb = isHypo
                ? h.hypo_risk_probability || 0.0
                : h.spike_risk_probability || h.hyper_risk_probability || 0.0;

              const isExpanded = Boolean(evidenceExpanded[h.horizon_minutes]);

              return (
                <div
                  key={h.horizon_minutes}
                  className="rounded-lg border border-red/60 bg-red-subtle/70 p-2.5 space-y-1.5 text-xs"
                >
                  {/* Compact Header: Title with expected time + Probability */}
                  <div className="flex items-center justify-between gap-1.5">
                    <div className="flex items-center gap-1.5 font-semibold text-red">
                      <AlertTriangle size={15} strokeWidth={1.75} className="shrink-0" />
                      <span className="text-xs leading-tight">{eventTitle}</span>
                    </div>
                    <Badge variant="danger" className="text-[10px] shrink-0 font-medium">
                      {(riskProb * 100).toFixed(0)}% risk
                    </Badge>
                  </div>

                    {/* Compact Evidence toggle button */}
                    <div className="pt-0.5">
                      <button
                        type="button"
                        onClick={() => toggleEvidence(h.horizon_minutes)}
                        className="w-full text-left text-[11px] text-text-secondary hover:text-text-primary flex items-center justify-between py-1 px-2 rounded bg-surface/80 border border-red/20 transition-colors"
                      >
                        <span className="font-medium text-text-primary">
                          Evidence
                        </span>
                        {isExpanded ? (
                          <ChevronUp size={13} strokeWidth={1.75} className="shrink-0 text-text-muted" />
                        ) : (
                          <ChevronDown size={13} strokeWidth={1.75} className="shrink-0 text-text-muted" />
                        )}
                      </button>

                      {isExpanded && (
                        <div className="mt-1.5 p-2 rounded bg-surface border border-red/20 space-y-2 text-[11px] text-text-secondary">
                          {/* Evidence Metrics: Predicted glucose, threshold, lead time, uncertainty */}
                          <div className="grid grid-cols-2 gap-1.5 p-1.5 bg-surface-muted rounded border border-border text-[10px] tnum">
                            <div>
                              <span className="text-text-muted block">Predicted Glucose:</span>
                              <strong className="text-text-primary text-[11px]">{h.q50.toFixed(0)} mg/dL</strong>
                            </div>
                            <div>
                              <span className="text-text-muted block">Alert Threshold:</span>
                              <strong className="text-text-primary text-[11px]">{isHypo ? "\u2264 70 mg/dL" : "\u2265 180 mg/dL"}</strong>
                            </div>
                            <div>
                              <span className="text-text-muted block">Lead Time:</span>
                              <strong className="text-text-primary text-[11px]">+{h.horizon_minutes} min</strong>
                            </div>
                            <div>
                              <span className="text-text-muted block">Uncertainty (q10-q90):</span>
                              <strong className="text-text-primary text-[11px]">{h.q10.toFixed(0)} to {h.q90.toFixed(0)} mg/dL</strong>
                            </div>
                          </div>

                          {/* Operating Metrics from Cross-Validation */}
                          {metrics ? (() => {
                            const m = metrics as Record<string, any>;
                            const recallMean = m.recall_mean ?? m.recall;
                            const recallStd = m.recall_std;
                            const precMean = m.precision_mean ?? m.precision;
                            const precStd = m.precision_std;
                            const faMean = m.false_alerts_per_patient_day_mean ?? m.false_alerts_per_patient_day;
                            const faStd = m.false_alerts_per_patient_day_std;

                            const recallText = recallStd !== undefined && recallStd > 0
                              ? `${(recallMean * 100).toFixed(0)} \u00b1 ${(recallStd * 100).toFixed(0)}%`
                              : `${(recallMean * 100).toFixed(0)}%`;

                            const precText = precStd !== undefined && precStd > 0
                              ? `${(precMean * 100).toFixed(0)} \u00b1 ${(precStd * 100).toFixed(0)}%`
                              : `${(precMean * 100).toFixed(0)}%`;

                            const faText = faStd !== undefined && faStd > 0
                              ? `${faMean.toFixed(1)} \u00b1 ${faStd.toFixed(1)}/day`
                              : `${faMean.toFixed(1)}/day`;

                            return (
                              <div className="space-y-1">
                                <div className="grid grid-cols-3 gap-1 tnum text-[10px] text-center">
                                  <div className="bg-surface p-1 rounded border border-border">
                                    <span className="text-text-muted block text-[9px]">Held-out Recall:</span>
                                    <strong className="text-text-primary">{recallText}</strong>
                                  </div>
                                  <div className="bg-surface p-1 rounded border border-border">
                                    <span className="text-text-muted block text-[9px]">Precision:</span>
                                    <strong className="text-text-primary">{precText}</strong>
                                  </div>
                                  <div className="bg-surface p-1 rounded border border-border">
                                    <span className="text-text-muted block text-[9px]">False alerts:</span>
                                    <strong className="text-text-primary">{faText}</strong>
                                  </div>
                                </div>
                                <div className="text-[9px] text-text-muted text-center">
                                  Cross-validated across 5 validation folds (mean ± SD)
                                </div>
                              </div>
                            );
                          })() : (
                            <div className="text-[10px] text-text-muted italic">Operating metrics unavailable from model service.</div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>

        {/* Collapsible Risk Estimates Only row */}
        {riskEstimates.length > 0 && (
          <div className="pt-1 border-t border-border">
            <button
              type="button"
              onClick={() => setRiskListExpanded(!riskListExpanded)}
              className="w-full flex items-center justify-between text-[11px] text-text-muted hover:text-text-secondary py-1 px-1.5 rounded hover:bg-surface-muted transition-colors"
            >
              <span className="flex items-center gap-1">
                <HelpCircle size={13} strokeWidth={1.75} />
                Risk estimates only ({riskEstimates.length} horizon{riskEstimates.length > 1 ? "s" : ""} below threshold criterion)
              </span>
              {riskListExpanded ? (
                <ChevronUp size={13} strokeWidth={1.75} />
              ) : (
                <ChevronDown size={13} strokeWidth={1.75} />
              )}
            </button>

            {riskListExpanded && (
              <div className="space-y-1 pt-1">
                {riskEstimates.map((h) => (
                  <div
                    key={h.horizon_minutes}
                    className="rounded border border-border bg-surface-muted p-1.5 flex items-center justify-between text-[11px] text-text-secondary"
                  >
                    <span>+{h.horizon_minutes}m horizon</span>
                    <span className="text-text-muted tnum text-[10px]">
                      Hypo: {((h.hypo_risk_probability || 0) * 100).toFixed(0)}%, Spike: {((h.spike_risk_probability || 0) * 100).toFixed(0)}%
                    </span>
                    <span className="text-[9px] text-text-muted bg-surface px-1 py-0.5 rounded border border-border">
                      Below threshold criterion
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </Card>
  );
};
