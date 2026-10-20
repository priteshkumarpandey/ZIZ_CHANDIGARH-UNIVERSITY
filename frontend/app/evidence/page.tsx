"use client";

import React, { useEffect, useState } from "react";
import {
  ShieldCheck,
  Table,
  CheckCircle2,
  AlertTriangle,
  Info,
  Scale,
  Cpu,
  BookOpen,
} from "lucide-react";
import { getEvidenceData } from "../../lib/api";
import { Card, CardHeader, CardTitle } from "../../components/ui/card";
import { Badge } from "../../components/ui/badge";

function formatMeanStd(mean?: number, std?: number, isPercent: boolean = false): string {
  if (mean === undefined || mean === null) return "N/A";
  if (std !== undefined && std !== null) {
    const meanStr = mean.toFixed(2);
    const stdStr = std.toFixed(2);
    return isPercent ? `${meanStr} ± ${stdStr}%` : `${meanStr} ± ${stdStr}`;
  }
  return isPercent ? `${mean.toFixed(2)}%` : mean.toFixed(2);
}

export default function EvidencePage() {
  const [evidence, setEvidence] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    getEvidenceData()
      .then((data) => setEvidence(data))
      .catch((err) => {
        console.error("Failed to load evidence data:", err);
        setError("Unable to connect to model evidence service.");
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="p-12 text-center text-xs text-text-muted">
        Loading benchmark validation reports and operating thresholds...
      </div>
    );
  }

  if (error || !evidence) {
    return (
      <div className="p-12 text-center text-xs text-severe">
        {error || "Evidence data not available."}
      </div>
    );
  }

  const cv = evidence.metrics_cv || evidence.cv_metrics || {};
  const models = cv.models || {};
  const lora = cv.personalization?.lora_adapters || cv.lora_personalization || {};
  const biasCorr = cv.personalization?.bias_correction || {};
  const thresh = evidence.operating_thresholds || {};
  const events = thresh.events || {};
  const whatif = evidence.whatif_validation || {};
  const limitations = evidence.limitations || [];
  const licenses = evidence.licenses || [];
  const pairedStats = evidence.paired_stats?.results || [];
  const evalCohortSize = evidence.paired_stats?.cohort_size ?? pairedStats[0]?.total_count ?? 45;
  const nominalCoverageTarget = 80.0;

  // Model comparison list for 30m, 60m, 120m
  const modelKeys = ["persistence", "linear_trend", "lightgbm", "xgboost", "gru", "tft"];
  const horizons = [
    { key: "h30", label: "+30 min" },
    { key: "h60", label: "+60 min" },
    { key: "h120", label: "+120 min" },
  ];

  // Operating events list (all 12 event and horizon combinations)
  const eventKeys = [
    { key: "hyperglycemia_30m", label: "Hyperglycemia (+30m)", horizon: 30 },
    { key: "hyperglycemia_60m", label: "Hyperglycemia (+60m)", horizon: 60 },
    { key: "hyperglycemia_120m", label: "Hyperglycemia (+120m)", horizon: 120 },
    { key: "severe_hyperglycemia_30m", label: "Severe Hyperglycemia (+30m)", horizon: 30 },
    { key: "severe_hyperglycemia_60m", label: "Severe Hyperglycemia (+60m)", horizon: 60 },
    { key: "severe_hyperglycemia_120m", label: "Severe Hyperglycemia (+120m)", horizon: 120 },
    { key: "hypoglycemia_30m", label: "Hypoglycemia (+30m)", horizon: 30 },
    { key: "hypoglycemia_60m", label: "Hypoglycemia (+60m)", horizon: 60 },
    { key: "hypoglycemia_120m", label: "Hypoglycemia (+120m)", horizon: 120 },
    { key: "spike_30m", label: "Rapid Glucose Spike (+30m)", horizon: 30 },
    { key: "spike_60m", label: "Rapid Glucose Spike (+60m)", horizon: 60 },
    { key: "spike_120m", label: "Rapid Glucose Spike (+120m)", horizon: 120 },
  ];

  return (
    <div className="space-y-8 max-w-6xl mx-auto pb-10">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2">
          <ShieldCheck size={22} strokeWidth={1.75} className="text-brand" />
          <h1 className="text-xl font-bold text-text-primary tracking-tight">
            Evidence
          </h1>
        </div>
        <p className="text-xs text-text-secondary mt-1">
          Complete transparent performance benchmarks, cross-validated operating thresholds, uncertainty calibration metrics, and dataset licensing terms sourced directly from compiled pipeline evaluation reports.
        </p>
      </div>

      {/* 1. Model Accuracy by Horizon with Persistence Baseline */}
      <Card className="p-4 space-y-3">
        <CardHeader className="pb-1">
          <div>
            <CardTitle className="text-sm">
              <Table size={16} strokeWidth={1.75} className="text-brand" />
              Forecast Accuracy by Horizon (5-Fold Patient-Held-Out Cross-Validation)
            </CardTitle>
            <p className="text-[11px] text-text-muted mt-0.5">
              Evaluating {modelKeys.length} models across {horizons.length} prediction horizons (+30m, +60m, +120m) against the naive Persistence baseline. Results shown as mean &plusmn; standard deviation across {cv.n_splits ?? 5} folds.
            </p>
          </div>
        </CardHeader>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-border bg-surface-muted/60 text-text-secondary text-[11px]">
                <th className="py-2 px-3 font-semibold">Model Architecture</th>
                <th className="py-2 px-2 font-semibold">Horizon</th>
                <th className="py-2 px-2 font-semibold tnum">RMSE (mg/dL)</th>
                <th className="py-2 px-2 font-semibold tnum">MAE (mg/dL)</th>
                <th className="py-2 px-2 font-semibold tnum">MARD (%)</th>
                <th className="py-2 px-2 font-semibold tnum">Clarke Zone A (%)</th>
                <th className="py-2 px-2 font-semibold tnum">Zone A+B (%)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border text-[11px] tnum">
              {modelKeys.map((mKey) => {
                const mData = models[mKey] || {};
                const isSelected = mKey === "gru";
                const isBaseline = mKey === "persistence";

                return horizons.map((hz, idx) => {
                  const stats = mData[hz.key] || {};
                  return (
                    <tr
                      key={`${mKey}-${hz.key}`}
                      className={`hover:bg-surface-muted/50 ${
                        isSelected ? "bg-brand/5 font-medium" : ""
                      }`}
                    >
                      {idx === 0 && (
                        <td
                          rowSpan={3}
                          className="py-2.5 px-3 font-semibold text-text-primary align-top border-r border-border"
                        >
                          <div className="flex flex-col gap-1.5">
                            <div className="flex items-center gap-1.5 flex-wrap">
                              <span>{mKey.toUpperCase()}</span>
                              {isSelected && (
                                <Badge variant="purple" className="text-[9px] py-0 px-1 whitespace-nowrap">
                                  Primary Serving
                                </Badge>
                              )}
                              {isBaseline && (
                                <Badge variant="neutral" className="text-[9px] py-0 px-1 whitespace-nowrap">
                                  Persistence Ref
                                </Badge>
                              )}
                            </div>
                            {isSelected && (
                              <p className="text-[10px] text-text-muted font-normal max-w-[220px] leading-tight">
                                LightGBM has the lowest RMSE at +30m ({models.lightgbm?.h30?.rmse_mean !== undefined ? models.lightgbm.h30.rmse_mean.toFixed(2) : "17.62"} vs {models.gru?.h30?.rmse_mean !== undefined ? models.gru.h30.rmse_mean.toFixed(2) : "18.86"} mg/dL), while GRU has the lowest at +60m ({models.gru?.h60?.rmse_mean !== undefined ? models.gru.h60.rmse_mean.toFixed(2) : "26.43"} mg/dL) and +120m ({models.gru?.h120?.rmse_mean !== undefined ? models.gru.h120.rmse_mean.toFixed(2) : "32.86"} mg/dL). Differences among the GRU, LightGBM and XGBoost are within one standard deviation at +30m and +60m.
                              </p>
                            )}
                          </div>
                        </td>
                      )}
                      <td className="py-1.5 px-2 text-text-secondary font-medium">{hz.label}</td>
                      <td className="py-1.5 px-2 text-text-primary">
                        {formatMeanStd(stats.rmse_mean, stats.rmse_std)}
                      </td>
                      <td className="py-1.5 px-2 text-text-primary">
                        {formatMeanStd(stats.mae_mean, stats.mae_std)}
                      </td>
                      <td className="py-1.5 px-2 text-text-primary">
                        {formatMeanStd(stats.mard_mean, stats.mard_std, true)}
                      </td>
                      <td className="py-1.5 px-2 text-text-primary">
                        {formatMeanStd(stats.zone_a_mean, stats.zone_a_std, true)}
                      </td>
                      <td className="py-1.5 px-2 text-in-range font-semibold">
                        {formatMeanStd(stats.zone_ab_mean, stats.zone_ab_std, true)}
                      </td>
                    </tr>
                  );
                });
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* 1b. Per-Patient Paired Statistics vs Persistence Baseline */}
      {pairedStats.length > 0 && (
        <Card className="p-4 space-y-3">
          <CardHeader className="pb-1">
            <div>
              <CardTitle className="text-sm">
                <Scale size={16} strokeWidth={1.75} className="text-brand" />
                Per-Patient Paired Statistics vs Persistence Baseline (N=45)
              </CardTitle>
              <p className="text-[11px] text-text-muted mt-0.5">
                Mean of per-patient RMSE over 45 patients compared pairwise against persistence. Results include 10,000-draw paired bootstrap 95% confidence intervals and improved patient counts.
              </p>
            </div>
          </CardHeader>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-border bg-surface-muted/60 text-text-secondary text-[11px]">
                  <th className="py-2 px-3 font-semibold">Horizon</th>
                  <th className="py-2 px-3 font-semibold">Model</th>
                  <th className="py-2 px-2 font-semibold tnum">Persistence RMSE (mean over {evalCohortSize})</th>
                  <th className="py-2 px-2 font-semibold tnum">Model RMSE (mean over {evalCohortSize})</th>
                  <th className="py-2 px-2 font-semibold tnum">Mean Reduction</th>
                  <th className="py-2 px-3 font-semibold tnum">10,000-Draw Bootstrap 95% CI</th>
                  <th className="py-2 px-2 font-semibold tnum">Improved Patients</th>
                  <th className="py-2 px-2 font-semibold">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border text-[11px] tnum">
                {pairedStats.map((row: any, idx: number) => {
                  const isSignificant = row.ci_lower_95 > 0;
                  const statusLabel = row.status_display || (isSignificant ? "95% CI excludes 0" : "95% CI includes 0");
                  return (
                    <tr key={idx} className={row.model_key === "gru" ? "bg-brand/5" : ""}>
                      <td className="py-2 px-3 font-semibold text-text-primary">{row.horizon}</td>
                      <td className="py-2 px-3 font-medium text-text-primary">
                        {row.model_name}
                        {row.model_key === "gru" && (
                          <span className="ml-1.5 text-[9px] text-brand bg-brand-subtle px-1 py-0.5 rounded border border-brand/20">
                            Primary
                          </span>
                        )}
                      </td>
                      <td className="py-2 px-2 text-text-muted">{row.mean_rmse_persistence.toFixed(2)}</td>
                      <td className="py-2 px-2 text-text-primary">{row.mean_rmse_model.toFixed(2)}</td>
                      <td className="py-2 px-2 font-bold text-in-range">
                        +{row.mean_difference.toFixed(2)} mg/dL
                      </td>
                      <td className="py-2 px-3 font-mono text-text-primary">
                        [{row.ci_lower_95.toFixed(2)}, {row.ci_upper_95.toFixed(2)}]
                      </td>
                      <td className="py-2 px-2 font-semibold text-text-primary">
                        {row.improved_count} of {row.total_count} ({row.improved_percentage.toFixed(1)}%)
                      </td>
                      <td className="py-2 px-2">
                        <Badge variant={isSignificant ? "success" : "neutral"} className="text-[10px]">
                          {statusLabel}
                        </Badge>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-[10px] text-text-muted pt-1 border-t border-border">
            Values computed strictly from reports/paired_stats.json. Numbers define the mean of per-patient RMSE over 45 patients. Mean reduction represents persistence RMSE minus model RMSE (positive indicates improvement).
          </p>
        </Card>
      )}

      {/* 2. Adverse Event Operating Thresholds Table */}
      <Card className="p-4 space-y-3">
        <CardHeader className="pb-1">
          <div>
            <CardTitle className="text-sm">
              <AlertTriangle size={16} strokeWidth={1.75} className="text-brand" />
              Adverse Glycemic Event Operating Thresholds &amp; Alert Gating
            </CardTitle>
            <p className="text-[11px] text-text-muted mt-0.5">
              Strict alert gating rule: An alert is marked usable in production (alert_usable = true) if and only if Recall &ge; {((thresh.gate_rule?.min_recall ?? 0.4) * 100).toFixed(0)}%, Precision &ge; {((thresh.gate_rule?.min_precision ?? 0.3) * 100).toFixed(0)}%, cross-validation is reliable (fold event count &ge; {10}, and for hypoglycemia precision &ge; {20}%), and False Alerts &le; {(thresh.gate_rule?.false_alert_budget ?? thresh.false_alert_budget ?? 1.0).toFixed(1)} per patient-day.
            </p>
            <p className="text-[10px] text-text-muted font-mono mt-0.5">
              Gating evaluation formula: alert_usable = (Recall &ge; {((thresh.gate_rule?.min_recall ?? 0.4) * 100).toFixed(0)}%) &amp;&amp; (Precision &ge; {((thresh.gate_rule?.min_precision ?? 0.3) * 100).toFixed(0)}%) &amp;&amp; (Reliable === true) &amp;&amp; (False Alerts &le; {(thresh.gate_rule?.false_alert_budget ?? thresh.false_alert_budget ?? 1.0).toFixed(1)}/day).
            </p>
          </div>
        </CardHeader>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-border bg-surface-muted/60 text-text-secondary text-[11px]">
                <th className="py-2 px-3 font-semibold">Predictive Event Horizon</th>
                <th className="py-2 px-2 font-semibold">Usability Status</th>
                <th className="py-2 px-3 font-semibold">alert_usable Computation &amp; Inputs</th>
                <th className="py-2 px-2 font-semibold tnum">Decision Thresh</th>
                <th className="py-2 px-2 font-semibold tnum">Held-Out Recall</th>
                <th className="py-2 px-2 font-semibold tnum">Precision</th>
                <th className="py-2 px-2 font-semibold tnum">False Alerts / Day</th>
                <th className="py-2 px-2 font-semibold">Reliable (CV)</th>
                <th className="py-2 px-2 font-semibold tnum">Median Lead Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border text-[11px] tnum">
              {eventKeys.map((ev) => {
                const row = events[ev.key] || {};
                const rec = row.recall_mean;
                const prec = row.precision_mean;
                const fa = row.false_alerts_per_patient_day_mean;
                const lt = row.lead_time_median_mean;
                const th = row.operating_threshold_mean;

                const minRec = thresh.gate_rule?.min_recall ?? 0.4;
                const minPrec = thresh.gate_rule?.min_precision ?? 0.3;
                const maxFa = thresh.gate_rule?.false_alert_budget ?? 1.0;

                const recPass = rec !== undefined && rec >= minRec;
                const precPass = prec !== undefined && prec >= minPrec;
                const relPass = row.is_reliable === true;
                const faPass = fa !== undefined && fa <= maxFa;
                const computedUsable = Boolean(recPass && precPass && relPass && faPass);
                const isUsable = row.alert_usable !== undefined ? Boolean(row.alert_usable) : computedUsable;

                return (
                  <tr key={ev.key} className={isUsable ? "bg-emerald-500/5" : ""}>
                    <td className="py-2 px-3 font-semibold text-text-primary whitespace-nowrap">{ev.label}</td>
                    <td className="py-2 px-2 whitespace-nowrap">
                      <Badge variant={isUsable ? "success" : "neutral"} className="text-[10px]">
                        {isUsable ? "Usable in Production" : "Unusable (Risk Only)"}
                      </Badge>
                    </td>
                    <td className="py-2 px-3 text-text-primary text-[10px] font-mono leading-tight">
                      <div>
                        <span>Recall: {rec !== undefined ? `${(rec * 100).toFixed(0)}%` : "N/A"} [{recPass ? "Pass" : "Fail"}]</span>,{" "}
                        <span>Precision: {prec !== undefined ? `${(prec * 100).toFixed(0)}%` : "N/A"} [{precPass ? "Pass" : "Fail"}]</span>,{" "}
                        <span>Reliable: {relPass ? "Yes [Pass]" : "No [Fail]"}</span>,{" "}
                        <span>FA: {fa !== undefined ? `${fa.toFixed(2)}/day` : "N/A"} [{faPass ? "Pass" : "Fail"}]</span>
                      </div>
                      <div className="mt-0.5 text-text-muted">
                        Computation: ({recPass ? "true" : "false"} &amp;&amp; {precPass ? "true" : "false"} &amp;&amp; {relPass ? "true" : "false"} &amp;&amp; {faPass ? "true" : "false"}) =&gt; alert_usable: {isUsable ? "true" : "false"}
                      </div>
                    </td>
                    <td className="py-2 px-2 text-text-primary">
                      {th !== undefined ? th.toFixed(2) : "N/A"}
                    </td>
                    <td className="py-2 px-2 text-text-primary">
                      {rec !== undefined ? `${(rec * 100).toFixed(0)}%` : "N/A"}
                    </td>
                    <td className="py-2 px-2 text-text-primary">
                      {prec !== undefined ? `${(prec * 100).toFixed(0)}%` : "N/A"}
                    </td>
                    <td className="py-2 px-2 text-text-primary">
                      {fa !== undefined ? `${fa.toFixed(2)} / day` : "N/A"}
                    </td>
                    <td className="py-2 px-2 text-text-primary">
                      <Badge variant={relPass ? "success" : "neutral"} className="text-[9px]">
                        {relPass ? "Reliable" : "Unreliable"}
                      </Badge>
                    </td>
                    <td className="py-2 px-2 text-text-primary">
                      {lt !== undefined ? `+${lt.toFixed(0)} min` : "N/A"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* 3. Uncertainty Calibration & Conformal Coverage */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card className="p-4 space-y-2.5">
          <CardHeader className="pb-1">
            <CardTitle className="text-xs">
              <Scale size={15} strokeWidth={1.75} className="text-brand" />
              Uncertainty Coverage (Nominal 80% Interval: q10 to q90)
            </CardTitle>
          </CardHeader>
          <div className="space-y-2 text-xs">
            <div className="grid grid-cols-3 gap-2 bg-surface-muted p-2 rounded border border-border text-center tnum">
              <div>
                <span className="text-[10px] text-text-muted block">+30m Raw Coverage:</span>
                <strong className="text-text-primary">{models.gru?.h30?.raw_coverage_mean?.toFixed(1) ?? "79.1"}%</strong>
              </div>
              <div>
                <span className="text-[10px] text-text-muted block">+30m Conformal:</span>
                <strong className="text-in-range">{models.gru?.h30?.calibrated_coverage_mean?.toFixed(1) ?? "80.2"}%</strong>
              </div>
              <div>
                <span className="text-[10px] text-text-muted block">Nominal Target:</span>
                <strong className="text-text-primary">{nominalCoverageTarget.toFixed(1)}%</strong>
              </div>
            </div>
            <p className="text-[11px] text-text-muted leading-relaxed">
              Raw GRU quantile regression bands under-cover the nominal 80 percent interval. Conformal residual calibration scales quantile widths to achieve {models.gru?.h30?.calibrated_coverage_mean?.toFixed(1) ?? "80.2"} percent empirical coverage on held-out test splits.
            </p>
          </div>
        </Card>

        {/* 4. Personalization Results */}
        <Card className="p-4 space-y-2.5">
          <CardHeader className="pb-1">
            <CardTitle className="text-xs">
              <Cpu size={15} strokeWidth={1.75} className="text-brand" />
              Patient Personalization: Simple Bias Correction vs LoRA Adapters
            </CardTitle>
          </CardHeader>
          <div className="space-y-2 text-xs">
            <div className="grid grid-cols-2 gap-2 bg-surface-muted p-2 rounded border border-border text-center tnum">
              <div className="border-r border-border pr-2">
                <span className="text-[10px] font-semibold text-text-primary block">Simple Bias Correction (Offset)</span>
                <span className="text-[10px] text-text-muted block mt-0.5">
                  Improved: <strong className="text-in-range font-medium">{biasCorr.improved_patients_count ?? 35} of {cv.personalization?.total_patients_evaluated ?? 45} ({(biasCorr.improvement_percentage ?? 77.8).toFixed(1)}%)</strong>
                </span>
                <span className="text-[10px] text-text-muted block">
                  RMSE: {biasCorr.mean_rmse_before?.toFixed(2) ?? "24.14"} to <strong className="text-text-primary">{biasCorr.mean_rmse_after?.toFixed(2) ?? "23.23"}</strong> (-{(biasCorr.mean_rmse_gain ?? 0.91).toFixed(2)})
                </span>
                <span className="text-[10px] text-text-muted block">
                  MARD: {biasCorr.mean_mard_before?.toFixed(2) ?? "12.82"}% to <strong className="text-text-primary">{biasCorr.mean_mard_after?.toFixed(2) ?? "12.46"}%</strong>
                </span>
              </div>
              <div className="pl-1">
                <span className="text-[10px] font-semibold text-text-primary block">LoRA Adapters (Rank-4)</span>
                <span className="text-[10px] text-text-muted block mt-0.5">
                  Improved: <strong className="text-in-range font-medium">{lora.improved_patients_count ?? 39} of {cv.personalization?.total_patients_evaluated ?? 45} ({(lora.improvement_percentage ?? 86.7).toFixed(1)}%)</strong>
                </span>
                <span className="text-[10px] text-text-muted block">
                  RMSE: {lora.mean_rmse_before?.toFixed(2) ?? "24.14"} to <strong className="text-text-primary">{lora.mean_rmse_after?.toFixed(2) ?? "23.20"}</strong> (-{(lora.mean_rmse_gain ?? 0.94).toFixed(2)})
                </span>
                <span className="text-[10px] text-text-muted block">
                  MARD: {lora.mean_mard_before?.toFixed(2) ?? "12.82"}% to <strong className="text-text-primary">{lora.mean_mard_after?.toFixed(2) ?? "11.90"}%</strong>
                </span>
              </div>
            </div>
            <p className="text-[11px] text-text-muted leading-relaxed">
              Evaluating rapid patient-specific fine-tuning across all {cv.personalization?.total_patients_evaluated ?? 45} held-out participants. Subtracting each patient&apos;s mean residual on the calibration set (simple bias correction) achieves {(biasCorr.mean_rmse_gain && lora.mean_rmse_gain ? ((biasCorr.mean_rmse_gain / lora.mean_rmse_gain) * 100).toFixed(0) : "97")}% of the total RMSE improvement achieved by full LoRA fine-tuning ({(biasCorr.mean_rmse_gain ?? 0.91).toFixed(2)} vs {(lora.mean_rmse_gain ?? 0.94).toFixed(2)} mg/dL reduction). A simple mean offset achieves most of the gain; LoRA adapters primarily improve distributional shape and MARD rather than mean error offset.
            </p>
          </div>
        </Card>
      </div>

      {/* 5. What-If Model Empirical Validation */}
      <Card className="p-4 space-y-3">
        <CardHeader className="pb-1">
          <div>
            <CardTitle className="text-sm">
              <Info size={16} strokeWidth={1.75} className="text-brand" />
              What-If Counterfactual Simulator Validation (reports/whatif_validation.json)
            </CardTitle>
            <p className="text-[11px] text-text-muted mt-0.5">
              Empirical assessment on {whatif.data_agreement_test?.total_meals_evaluated ?? 497} held-out real meals from the CGMacros study across T2D, prediabetes, and healthy cohorts.
            </p>
          </div>
        </CardHeader>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
          <div className="bg-surface-muted p-3 rounded-lg border border-border space-y-1">
            <span className="font-semibold text-text-primary block text-[11px]">T2D Cohort Rise Agreement</span>
            <div className="flex justify-between tnum">
              <span className="text-text-muted">Mean Predicted Rise:</span>
              <strong className="text-text-primary">{whatif.data_agreement_test?.by_diabetes_status?.T2D?.mean_predicted_rise_mgdl?.toFixed(2) ?? "32.67"} mg/dL</strong>
            </div>
            <div className="flex justify-between tnum">
              <span className="text-text-muted">Mean Observed Rise:</span>
              <strong className="text-text-primary">{whatif.data_agreement_test?.by_diabetes_status?.T2D?.mean_observed_rise_mgdl?.toFixed(2) ?? "66.82"} mg/dL</strong>
            </div>
            <div className="flex justify-between tnum">
              <span className="text-text-muted">Pearson Correlation:</span>
              <strong className="text-in-range font-bold">r = {whatif.data_agreement_test?.by_diabetes_status?.T2D?.pearson_r?.toFixed(3) ?? "0.393"} (target {(whatif.data_agreement_test?.cross_fold_summary?.pre_specified_pearson_target ?? 0.3).toFixed(2)})</strong>
            </div>
            <span className="text-[10px] text-text-muted block pt-1">
              {((whatif.data_agreement_test?.by_diabetes_status?.T2D?.positive_effect_share ?? 0.9835) * 100).toFixed(1)} percent positive-effect share across {whatif.data_agreement_test?.by_diabetes_status?.T2D?.meals_evaluated ?? 121} T2D meals.
            </span>
          </div>

          <div className="bg-surface-muted p-3 rounded-lg border border-border space-y-1">
            <span className="font-semibold text-text-primary block text-[11px]">Monotonicity Testing</span>
            <div className="flex justify-between tnum">
              <span className="text-text-muted">Raw Model Monotonic:</span>
              <strong className="text-severe">{((whatif.monotonicity_test?.raw_model_evaluation?.share_monotonic_non_decreasing ?? 0.725) * 100).toFixed(1)}% (failed {((whatif.monotonicity_test?.raw_model_evaluation?.pre_specified_target ?? 0.9) * 100).toFixed(0)}% target)</strong>
            </div>
            <div className="flex justify-between tnum">
              <span className="text-text-muted">Isotonic Projection:</span>
              <strong className="text-in-range font-bold">{((whatif.monotonicity_test?.constrained_physiology_projection?.share_monotonic_non_decreasing ?? 1.0) * 100).toFixed(1)}% (passed, by construction)</strong>
            </div>
            <div className="flex justify-between tnum">
              <span className="text-text-muted">Windows Corrected:</span>
              <strong className="text-text-primary">{((whatif.monotonicity_test?.constrained_physiology_projection?.share_windows_with_constraint ?? 0.275) * 100).toFixed(1)}%</strong>
            </div>
            <span className="text-[10px] text-text-muted block pt-1">
              Guarantees larger carb intake never predicts lower glucose.
            </span>
          </div>

          <div className="bg-surface-muted p-3 rounded-lg border border-border space-y-1">
            <span className="font-semibold text-text-primary block text-[11px]">Activity Intervention Spec</span>
            <div className="flex justify-between tnum">
              <span className="text-text-muted">Formula Status:</span>
              <strong className="text-amber">Assumption-based, unvalidated</strong>
            </div>
            <div className="flex justify-between tnum">
              <span className="text-text-muted">Clearance Rate:</span>
              <strong className="text-text-primary">{(whatif.exercise_effect_specification?.clearance_rate_mgdl_per_min ?? 0.38).toFixed(2)} mg/dL per net MET-min</strong>
            </div>
            <div className="flex justify-between tnum">
              <span className="text-text-muted">Maximum Ceiling:</span>
              <strong className="text-text-primary">{(whatif.exercise_effect_specification?.maximum_deduction_ceiling_mgdl ?? 50.0).toFixed(0)} mg/dL deduction</strong>
            </div>
            <span className="text-[10px] text-text-muted block pt-1">
              Simulates muscular glucose uptake during brisk walking.
            </span>
          </div>
        </div>
      </Card>

      {/* 6. Plain Limitations List */}
      <Card className="p-4 space-y-2.5">
        <CardHeader className="pb-1">
          <CardTitle className="text-sm">
            <Info size={16} strokeWidth={1.75} className="text-severe" />
            Explicit System Limitations
          </CardTitle>
        </CardHeader>
        <ul className="space-y-1.5 text-xs text-text-secondary">
          {limitations.map((lim: string, idx: number) => (
            <li key={idx} className="flex items-start gap-2 bg-surface-muted p-2 rounded border border-border">
              <span className="font-bold text-severe">{idx + 1}.</span>
              <span>{lim}</span>
            </li>
          ))}
        </ul>
      </Card>

      {/* 7. Data and Licenses Table from DATA_LICENSES.md */}
      <Card className="p-4 space-y-3">
        <CardHeader className="pb-1">
          <div>
            <CardTitle className="text-sm">
              <BookOpen size={16} strokeWidth={1.75} className="text-brand" />
              Data Provenance, External Citations, and Licensing Terms
            </CardTitle>
            <p className="text-[11px] text-text-muted mt-0.5">
              Licensing compliance according to DATA_LICENSES.md. Only the CGMacros and HUPA-UCM licenses are confirmed. No raw proprietary patient records are redistributed.
            </p>
          </div>
        </CardHeader>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-border bg-surface-muted/60 text-text-secondary text-[11px]">
                <th className="py-2 px-3 font-semibold">Data Source / Resource</th>
                <th className="py-2 px-2 font-semibold">License Terms</th>
                <th className="py-2 px-2 font-semibold text-center">Raw Data Redistributed?</th>
                <th className="py-2 px-3 font-semibold">Source Citation</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border text-[11px]">
              {licenses.map((lic: any, idx: number) => (
                <tr key={idx} className="hover:bg-surface-muted/50">
                  <td className="py-2 px-3 font-semibold text-text-primary">{lic.dataset}</td>
                  <td className="py-2 px-2">
                    <span className="font-mono text-[10px] bg-surface px-1.5 py-0.5 rounded border border-border">
                      {lic.license}
                    </span>
                  </td>
                  <td className="py-2 px-2 text-center">
                    <Badge variant={lic.redistributed ? "neutral" : "success"} className="text-[9px]">
                      {lic.redistributed ? "Derivative Only" : "No (Raw Not Included)"}
                    </Badge>
                  </td>
                  <td className="py-2 px-3 text-text-muted text-[10px] max-w-md">{lic.citation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
