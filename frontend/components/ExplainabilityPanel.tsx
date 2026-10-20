import React, { useState } from "react";
import { Brain } from "lucide-react";
import { ExplainabilityResponse, FeatureAttribution } from "../lib/types";
import { Card, CardHeader, CardTitle } from "./ui/card";
import { Tabs } from "./ui/tabs";
import { Badge } from "./ui/badge";

interface ExplainabilityPanelProps {
  explainability: ExplainabilityResponse | null;
  asOf?: string;
  currentGlucose?: number;
}

const FEATURE_UNITS: Record<string, string> = {
  glucose_mgdl: "mg/dL",
  glucose_diff_5m: "mg/dL",
  glucose_diff_15m: "mg/dL",
  glucose_diff_30m: "mg/dL",
  glucose_diff_60m: "mg/dL",
  glucose_roll_mean_30m: "mg/dL",
  glucose_roll_std_30m: "mg/dL",
  heart_rate: "bpm",
  steps: "steps",
  activity_level: "METs",
  carbs_g: "g",
  carbs_last_2h: "g",
  time_since_meal_mins: "min",
  age: "yrs",
  bmi: "kg/m\u00b2",
  hba1c: "%",
  fasting_glucose: "mg/dL",
  systolic_bp: "mmHg",
  diastolic_bp: "mmHg",
  cholesterol: "mg/dL",
  triglycerides: "mg/dL",
  hdl: "mg/dL",
  ldl: "mg/dL",
  creatinine: "mg/dL",
  insulin: "uU/mL",
  height_cm: "cm",
  weight_kg: "kg",
};

const FEATURE_DESCRIPTIONS: Record<string, string> = {
  glucose_mgdl: "Current interstitial glucose reading",
  glucose_diff_5m: "5-minute velocity of glucose change",
  glucose_diff_15m: "15-minute glycemic momentum",
  glucose_diff_30m: "30-minute excursion trajectory",
  glucose_diff_60m: "60-minute postprandial acceleration",
  glucose_roll_mean_30m: "Rolling 30-minute average glucose",
  glucose_roll_std_30m: "Rolling 30-minute glycemic variability index",
  heart_rate: "Continuous heart rate reflecting metabolic strain",
  steps: "Step cadence during activity window",
  activity_level: "Physical activity metabolic equivalents (METs)",
  carbs_g: "Carbohydrate intake from recent nutrition",
  carbs_last_2h: "Nutritional carbohydrate load in past 2 hours",
  time_since_meal_mins: "Elapsed time since last recorded nutritional intake",
  age: "Patient chronological age",
  bmi: "Body Mass Index",
  hba1c: "Glycated hemoglobin indicating 3-month glycemic control",
  fasting_glucose: "Baseline morning fasting glucose",
  systolic_bp: "Resting systolic blood pressure",
  diastolic_bp: "Resting diastolic blood pressure",
};

export const ExplainabilityPanel: React.FC<ExplainabilityPanelProps> = ({
  explainability,
  asOf,
  currentGlucose,
}) => {
  const [activeTab, setActiveTab] = useState<string>("gru");

  if (!explainability) return null;

  const displayAsOf = asOf || explainability.as_of;
  const timeStr = displayAsOf
    ? displayAsOf.includes("T")
      ? displayAsOf.split("T")[1]?.slice(0, 5) || displayAsOf
      : displayAsOf
    : null;

  const gruFeatures: FeatureAttribution[] =
    explainability.gru_permutation_importance && explainability.gru_permutation_importance.length > 0
      ? explainability.gru_permutation_importance
      : explainability.top_features || [];

  const lgbFeatures: FeatureAttribution[] =
    explainability.lightgbm_shap_values && explainability.lightgbm_shap_values.length > 0
      ? explainability.lightgbm_shap_values
      : explainability.top_features || [];

  // Strictly show top 4 features to prevent height overflow
  const displayFeatures = (activeTab === "gru" ? gruFeatures : lgbFeatures).slice(0, 4);
  const allZero =
    displayFeatures.length === 0 ||
    displayFeatures.every((f) => Math.abs(f.shap_value || 0) < 0.001);

  return (
    <Card className="w-full p-2.5">
      <CardHeader className="pb-1 flex-row items-center justify-between">
        <div>
          <CardTitle className="text-xs">
            <Brain size={15} strokeWidth={1.75} className="text-low" />
            Why this forecast
          </CardTitle>
          <p className="text-[10px] text-text-muted" data-testid="why-panel-subtitle">
            Top feature impacts for +{explainability.horizon_minutes}m horizon
            {timeStr ? ` \u2022 as of ${timeStr}` : ""}
          </p>
        </div>

        <Tabs
          tabs={[
            { id: "gru", label: "GRU Perm" },
            { id: "shap", label: "SHAP" },
          ]}
          activeTab={activeTab}
          onTabChange={setActiveTab}
        />
      </CardHeader>

      <div className="space-y-1 pt-0.5">
        {explainability.variation_note ? (
          <p className="text-[11px] text-text-muted italic py-1.5 text-center">
            {explainability.variation_note}
          </p>
        ) : allZero ? (
          <p className="text-[11px] text-text-muted italic py-1.5 text-center">
            little variation in this window
          </p>
        ) : (
          displayFeatures.map((feat) => {
            const val = feat.shap_value || 0;
            const isPositive = val >= 0;
            const magnitude = Math.min(100, Math.abs(val) * 22);
            const unit = FEATURE_UNITS[feat.feature_name] || "";
            const desc =
              feat.description || FEATURE_DESCRIPTIONS[feat.feature_name] || feat.feature_name;
            const isGlucoseFeat = feat.feature_name === "glucose_mgdl";
            const displayedUserVal =
              isGlucoseFeat && currentGlucose !== undefined
                ? Number(currentGlucose.toFixed(1))
                : feat.user_value;

            return (
              <div key={feat.feature_name} className="space-y-0.5" title={desc}>
                <div className="flex justify-between items-center text-[11px]">
                  <span
                    className="font-medium text-text-primary truncate max-w-[170px] cursor-help"
                    title={desc}
                  >
                    {feat.description || feat.feature_name}
                  </span>
                  <div className="flex items-center gap-1.5 shrink-0">
                    <span
                      className="text-[10px] text-text-muted tnum"
                      data-testid={isGlucoseFeat ? "why-panel-current-glucose" : undefined}
                    >
                      {displayedUserVal}
                      {unit ? ` ${unit}` : ""}
                    </span>
                    <Badge variant={isPositive ? "danger" : "info"} className="text-[9px] px-1 py-0">
                      <span className="tnum">
                        {isPositive ? `+${val.toFixed(1)}` : `${val.toFixed(1)}`}
                      </span>
                    </Badge>
                  </div>
                </div>

                <div className="h-1 w-full bg-surface-muted rounded-full overflow-hidden flex border border-border">
                  <div
                    className={`h-full rounded-full ${isPositive ? "bg-severe" : "bg-brand-line"}`}
                    style={{ width: `${Math.max(4, magnitude)}%` }}
                  />
                </div>
              </div>
            );
          })
        )}
      </div>
    </Card>
  );
};
