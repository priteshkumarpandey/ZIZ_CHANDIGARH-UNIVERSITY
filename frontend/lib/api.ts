import {
  HealthStatus,
  PatientProfile,
  TwinStateResponse,
  CGMReading,
  FoodItem,
  ForecastResponse,
  SimulationResponse,
  ClinicalSummaryResponse,
  ExplainabilityResponse,
  PersonalizationStatus,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function checkApiHealth(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error("API is unreachable");
  return res.json();
}

export async function getPatients(): Promise<PatientProfile[]> {
  const res = await fetch(`${API_BASE}/api/v1/twin/patients`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch patient personas");
  return res.json();
}

export async function getPatientProfile(patientId: string): Promise<PatientProfile> {
  const patients = await getPatients();
  const found = patients.find((p) => p.id === patientId);
  if (!found) throw new Error(`Patient ${patientId} not found`);
  return found;
}

export async function getTwinState(patientId: string): Promise<TwinStateResponse> {
  const res = await fetch(`${API_BASE}/api/v1/twin/${patientId}/state`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch twin state for ${patientId}`);
  return res.json();
}

export async function getPatientCGM(patientId: string, days: number = 2): Promise<CGMReading[]> {
  const res = await fetch(`${API_BASE}/api/v1/twin/${patientId}/cgm?days=${days}`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch CGM history");
  return res.json();
}

export async function getPersonalizationStatus(patientId: string): Promise<PersonalizationStatus> {
  const res = await fetch(`${API_BASE}/api/v1/twin/${patientId}/personalization`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch personalization status");
  return res.json();
}

export async function getExplainability(
  patientId: string,
  horizon: number = 60,
  asOf?: string,
  signal?: AbortSignal
): Promise<ExplainabilityResponse> {
  const url = new URL(`${API_BASE}/api/v1/explain/${patientId}`);
  url.searchParams.set("horizon", String(horizon));
  if (asOf) url.searchParams.set("as_of", asOf);
  const res = await fetch(url.toString(), { cache: "no-store", signal });
  if (!res.ok) throw new Error("Failed to fetch explainability");
  return res.json();
}

export async function predictGlucose(
  patientId: string,
  cgmSeries: number[],
  carbs: number = 0.0,
  gi: number = 0.0,
  asOf?: string,
  signal?: AbortSignal
): Promise<ForecastResponse> {
  const body: Record<string, any> = {
    patient_id: patientId,
    recent_cgm_series: cgmSeries,
    recent_carbs_g: carbs,
    recent_glycemic_index: gi,
    recent_heart_rate: 72.0,
    recent_mets: 1.0,
  };
  if (asOf) {
    body.as_of = asOf;
  }
  const res = await fetch(`${API_BASE}/api/v1/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw new Error("Failed to generate glucose prediction");
  return res.json();
}

export async function getFoodCatalog(): Promise<FoodItem[]> {
  const res = await fetch(`${API_BASE}/api/v1/simulate/foods`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch food catalog");
  return res.json();
}

export const fetchPatients = getPatients;
export const fetchPatientCGM = getPatientCGM;
export const fetchFoods = getFoodCatalog;

export async function runSimulation(
  patientId: string,
  food?: FoodItem | string | null,
  servings: number = 1.0,
  activityMinutes: number = 0,
  calibrateMealResponse: boolean = false,
  asOf?: string
): Promise<SimulationResponse> {
  const foodObj = typeof food === "object" ? food : null;
  const foodName = foodObj ? foodObj.name : typeof food === "string" ? food : "Simulated Meal";
  const carbs = foodObj ? (foodObj.carbohydrates_g || 0) * servings : (food ? 50.0 : 0);
  const fiber = foodObj ? (foodObj.fiber_g || 0) * servings : (food ? 4.0 : 0);
  const protein = foodObj ? (foodObj.protein_g || 0) * servings : (food ? 10.0 : 0);
  const fat = foodObj ? (foodObj.fat_g || 0) * servings : (food ? 8.0 : 0);
  const gi = foodObj?.glycemic_index || 55;
  const gl = (gi * carbs) / 100;

  const res = await fetch(`${API_BASE}/api/v1/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      patient_id: patientId,
      as_of: asOf,
      calibrate_meal_response: calibrateMealResponse,
      simulation_duration_minutes: 180,
      planned_meal: food
        ? {
            timestamp: asOf || new Date().toISOString(),
            name: foodName,
            items: [],
            total_carbohydrates_g: carbs,
            total_fiber_g: fiber,
            total_protein_g: protein,
            total_fat_g: fat,
            effective_glycemic_load: gl,
          }
        : null,
      planned_activity:
        activityMinutes > 0
          ? {
              timestamp: asOf || new Date().toISOString(),
              activity_type: "Post-meal Brisk Walk",
              duration_minutes: activityMinutes,
              mets: 3.5,
              steps_estimate: activityMinutes * 100,
            }
          : null,
    }),
  });
  if (!res.ok) throw new Error("Simulation execution failed");
  return res.json();
}

export async function getClinicalSummary(
  patientId: string,
  asOf?: string,
  signal?: AbortSignal
): Promise<ClinicalSummaryResponse> {
  const url = new URL(`${API_BASE}/api/v1/summary/${patientId}`);
  if (asOf) url.searchParams.set("as_of", asOf);
  const res = await fetch(url.toString(), { cache: "no-store", signal });
  if (!res.ok) throw new Error("Failed to fetch clinical summary");
  return res.json();
}

export async function getEvidenceData(): Promise<any> {
  const res = await fetch(`${API_BASE}/api/v1/evidence`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch evidence reports");
  return res.json();
}

export async function triggerPersonalizationCalibration(patientId: string) {
  const res = await fetch(`${API_BASE}/api/v1/twin/${patientId}/personalize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) throw new Error("Personalization trigger failed");
  return res.json();
}

