import type { components } from "./api-schema";

export type Schema<T extends keyof components["schemas"]> = components["schemas"][T];

export type PatientProfile = Schema<"PatientProfile">;

export interface RuleBasedAlert {
  alert_type: string;
  severity: string;
  title: string;
  message: string;
  observed_glucose: number;
  threshold_mgdl: number;
  rate_mgdl_per_min?: number | null;
  is_rule_based: boolean;
  provenance: string;
}

export type TwinStateResponse = Schema<"TwinStateResponse"> & {
  rule_based_alerts?: RuleBasedAlert[];
};
export type ForecastResponse = Schema<"ForecastResponse"> & {
  rule_based_alerts?: RuleBasedAlert[];
};
export type ForecastPoint = Schema<"ForecastPoint">;
export type CGMReading = Schema<"CGMReading">;
export type FoodItem = Schema<"FoodItem">;
export type SimulationResponse = Schema<"SimulationResponse"> & {
  as_of?: string | null;
  starting_glucose?: number | null;
};
export type ClinicalSummaryResponse = Schema<"ClinicalSummaryResponse"> & {
  monitoring_window_text?: string | null;
};
export type ClinicalMetrics = Schema<"ClinicalMetrics">;
export type ExplainabilityResponse = Schema<"ExplainabilityResponse"> & {
  variation_note?: string | null;
  as_of?: string | null;
  current_glucose?: number | null;
};
export type FeatureAttribution = Schema<"FeatureAttribution">;
export type PredictRequest = Schema<"PredictRequest">;
export type SimulateRequest = Schema<"SimulateRequest">;

export interface OperatingPoint {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  usable: boolean;
}

export interface OperatingThresholdsSummary {
  [key: string]: any;
}

export interface PersonalizationStatus {
  patient_id?: string;
  is_personalized?: boolean;
  adapter_version?: string;
  historical_mard_percentage?: number[];
  historical_rmse_mgdl?: number[];
  num_samples_seen?: number;
  last_updated?: string;
}


export interface HealthStatus {
  status: string;
  project?: string;
  environment?: string;
  version?: string;
}

export type PatientCohortCategory = "synthetic_demo" | "synthetic_persona" | "cgmacros_research";

export function getPatientCohort(patientId: string): PatientCohortCategory {
  if (patientId.startsWith("DEMO_PAT")) {
    return "synthetic_demo";
  }
  if (patientId.startsWith("IND_PAT")) {
    return "synthetic_persona";
  }
  return "cgmacros_research";
}

export function getCohortBadgeText(patientId: string): string {
  const cohort = getPatientCohort(patientId);
  switch (cohort) {
    case "synthetic_demo":
      return "Synthetic demo data";
    case "synthetic_persona":
      return "Synthetic persona (synthetic linkage)";
    case "cgmacros_research":
      return "De-identified research data";
  }
}

