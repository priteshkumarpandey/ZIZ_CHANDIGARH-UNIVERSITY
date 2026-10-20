"use client";

import React, { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ReferenceArea,
  CartesianGrid,
} from "recharts";
import {
  FlaskConical,
  Utensils,
  Footprints,
  Sparkles,
  ChevronRight,
  CheckCircle2,
  Sliders,
  Scale,
  Info,
} from "lucide-react";
import { getFoodCatalog, runSimulation, getPatients, getEvidenceData, getTwinState } from "../../lib/api";
import { FoodItem, SimulationResponse, PatientProfile } from "../../lib/types";
import { Card } from "../../components/ui/card";
import { Badge } from "../../components/ui/badge";
import { Button } from "../../components/ui/button";

export default function WhatIfPage() {
  const [patients, setPatients] = useState<PatientProfile[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>("IND_PAT_001");
  const [foods, setFoods] = useState<FoodItem[]>([]);
  const [selectedFoodId, setSelectedFoodId] = useState<string>("ifct_002");
  const [servings, setServings] = useState<number>(1.0);
  const [activityMinutes, setActivityMinutes] = useState<number>(0);
  const [calibrateModel, setCalibrateModel] = useState<boolean>(false);
  const [compareMode, setCompareMode] = useState<boolean>(false);

  // Scenario B controls for comparison mode
  const [foodBId, setFoodBId] = useState<string>("ifct_001");
  const [servingsB, setServingsB] = useState<number>(1.0);
  const [activityBMinutes, setActivityBMinutes] = useState<number>(20);

  const [patientAsOf, setPatientAsOf] = useState<string | null>(null);
  const [simulationA, setSimulationA] = useState<SimulationResponse | null>(null);
  const [simulationB, setSimulationB] = useState<SimulationResponse | null>(null);
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [whatifValidation, setWhatifValidation] = useState<any>(null);

  useEffect(() => {
    Promise.all([getPatients(), getFoodCatalog(), getEvidenceData().catch(() => null)])
      .then(([pts, fds, ev]) => {
        setPatients(pts);
        setFoods(fds);
        if (fds && fds.length > 0) {
          if (!fds.some((f) => f.id === selectedFoodId)) {
            setSelectedFoodId(fds[0].id);
          }
          if (fds.length > 1 && !fds.some((f) => f.id === foodBId)) {
            setFoodBId(fds[0].id);
          }
        }
        if (ev && ev.whatif_validation) {
          setWhatifValidation(ev.whatif_validation);
        }
      })
      .catch((err) => console.error("Error loading simulation data:", err));
  }, []);

  useEffect(() => {
    getTwinState(selectedPatientId)
      .then((state) => {
        if (state?.last_updated) {
          setPatientAsOf(state.last_updated);
        }
      })
      .catch(() => null);
  }, [selectedPatientId]);

  const handleSimulate = async () => {
    setIsSimulating(true);
    try {
      const foodA = foods.find((f) => f.id === selectedFoodId) || null;
      const resA = await runSimulation(
        selectedPatientId,
        foodA,
        servings,
        activityMinutes,
        calibrateModel,
        patientAsOf || undefined
      );
      setSimulationA(resA);

      if (compareMode) {
        const foodB = foods.find((f) => f.id === foodBId) || null;
        const resB = await runSimulation(
          selectedPatientId,
          foodB,
          servingsB,
          activityBMinutes,
          calibrateModel,
          patientAsOf || undefined
        );
        setSimulationB(resB);
      } else {
        setSimulationB(null);
      }
    } catch (err) {
      console.error("Simulation error", err);
    } finally {
      setIsSimulating(false);
    }
  };

  const selectedFood = foods.find((f) => f.id === selectedFoodId);
  const selectedFoodB = foods.find((f) => f.id === foodBId);

  // Dynamic validation values from reports/whatif_validation.json
  const cfMean =
    whatifValidation?.data_agreement_test?.cross_fold_summary?.pearson_r_mean !== undefined
      ? Number(whatifValidation.data_agreement_test.cross_fold_summary.pearson_r_mean).toFixed(4)
      : "0.3994";
  const cfStd =
    whatifValidation?.data_agreement_test?.cross_fold_summary?.pearson_r_std !== undefined
      ? Number(whatifValidation.data_agreement_test.cross_fold_summary.pearson_r_std).toFixed(4)
      : "0.1102";
  const targetPearson =
    whatifValidation?.data_agreement_test?.cross_fold_summary?.pre_specified_pearson_target !== undefined
      ? Number(whatifValidation.data_agreement_test.cross_fold_summary.pre_specified_pearson_target).toFixed(2)
      : "0.30";
  const pooledPearson =
    whatifValidation?.data_agreement_test?.pooled_results?.pearson_r !== undefined
      ? Number(whatifValidation.data_agreement_test.pooled_results.pearson_r).toFixed(4)
      : "0.3777";
  const t2dPearson =
    whatifValidation?.data_agreement_test?.by_diabetes_status?.T2D?.pearson_r !== undefined
      ? Number(whatifValidation.data_agreement_test.by_diabetes_status.T2D.pearson_r).toFixed(4)
      : "0.3931";
  const t2dPredRise =
    whatifValidation?.data_agreement_test?.by_diabetes_status?.T2D?.mean_predicted_rise_mgdl !== undefined
      ? Number(whatifValidation.data_agreement_test.by_diabetes_status.T2D.mean_predicted_rise_mgdl).toFixed(2)
      : "32.67";
  const t2dObsRise =
    whatifValidation?.data_agreement_test?.by_diabetes_status?.T2D?.mean_observed_rise_mgdl !== undefined
      ? Number(whatifValidation.data_agreement_test.by_diabetes_status.T2D.mean_observed_rise_mgdl).toFixed(2)
      : "66.82";
  const t2dMeals =
    whatifValidation?.data_agreement_test?.by_diabetes_status?.T2D?.meals_evaluated ?? 121;
  const t2dShare =
    whatifValidation?.data_agreement_test?.by_diabetes_status?.T2D?.positive_effect_share !== undefined
      ? (Number(whatifValidation.data_agreement_test.by_diabetes_status.T2D.positive_effect_share) * 100).toFixed(1)
      : "98.4";
  const clearanceRate =
    whatifValidation?.exercise_effect_specification?.clearance_rate_mgdl_per_min !== undefined
      ? Number(whatifValidation.exercise_effect_specification.clearance_rate_mgdl_per_min).toFixed(2)
      : "0.38";
  const clearanceStatus =
    whatifValidation?.exercise_effect_specification?.status ?? "assumption-based, unvalidated";

  // Build chart dataset from simulation curves
  interface SimulationChartPoint {
    minuteOffset: number;
    baseline?: number;
    scenarioA?: number;
    q10A?: number;
    q90A?: number;
    fanRangeA?: [number, number];
    scenarioB?: number;
    fanRangeB?: [number, number];
  }

  const chartData: SimulationChartPoint[] = [];
  if (simulationA && simulationA.scenario_trajectory) {
    const totalPoints = simulationA.scenario_trajectory.length;
    for (let i = 0; i < totalPoints; i++) {
      const min = i * 5;
      const baseVal = simulationA.baseline_trajectory ? simulationA.baseline_trajectory[i] : undefined;
      const scnVal = simulationA.scenario_trajectory[i];
      const q10 = simulationA.scenario_q10 ? simulationA.scenario_q10[i] : scnVal;
      const q90 = simulationA.scenario_q90 ? simulationA.scenario_q90[i] : scnVal;

      const scnBVal = simulationB && simulationB.scenario_trajectory ? simulationB.scenario_trajectory[i] : undefined;
      const q10B = simulationB && simulationB.scenario_q10 ? simulationB.scenario_q10[i] : scnBVal;
      const q90B = simulationB && simulationB.scenario_q90 ? simulationB.scenario_q90[i] : scnBVal;

      chartData.push({
        minuteOffset: min,
        baseline: baseVal,
        scenarioA: scnVal,
        q10A: q10,
        q90A: q90,
        fanRangeA: q10 !== undefined && q90 !== undefined ? [q10, q90] : undefined,
        scenarioB: scnBVal,
        fanRangeB: q10B !== undefined && q90B !== undefined ? [q10B, q90B] : undefined,
      });
    }
  }

  // Calculate horizon deltas (+30m, +60m, +120m, peak)
  const getHorizonDelta = (sim: SimulationResponse | null, min: number) => {
    if (!sim || !sim.scenario_trajectory || !sim.baseline_trajectory) return 0;
    const idx = Math.min(Math.floor(min / 5), sim.scenario_trajectory.length - 1);
    const scn = sim.scenario_trajectory[idx];
    const base = sim.baseline_trajectory[idx];
    return scn - base;
  };

  const delta30A = getHorizonDelta(simulationA, 30);
  const delta60A = getHorizonDelta(simulationA, 60);
  const delta120A = getHorizonDelta(simulationA, 120);
  const peakDeltaA = simulationA
    ? simulationA.scenario_peak_glucose - simulationA.baseline_peak_glucose
    : 0;

  const delta30B = getHorizonDelta(simulationB, 30);
  const delta60B = getHorizonDelta(simulationB, 60);
  const delta120B = getHorizonDelta(simulationB, 120);
  const peakDeltaB = simulationB
    ? simulationB.scenario_peak_glucose - simulationB.baseline_peak_glucose
    : 0;

  const delta30Diff = simulationA && simulationB ? delta30B - delta30A : 0;
  const delta60Diff = simulationA && simulationB ? delta60B - delta60A : 0;
  const delta120Diff = simulationA && simulationB ? delta120B - delta120A : 0;
  const peakDeltaDiff = simulationA && simulationB
    ? simulationB.scenario_peak_glucose - simulationA.scenario_peak_glucose
    : 0;

  return (
    <div className="space-y-6">
      {/* Page Title & Provenance Badges */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div>
          <h1 className="text-lg font-bold text-text-primary flex items-center gap-2">
            <FlaskConical size={20} strokeWidth={1.75} className="text-high" />
            Physiology-Constrained What-If Simulator
          </h1>
          <p className="text-xs text-text-secondary mt-0.5">
            Explore counterfactual postprandial glucose curves by testing regional Indian meal substitutions and physical activity interventions.
          </p>
        </div>

        {/* Required Provenance Badges */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <Badge variant="purple" className="text-[10px]">
            model-based
          </Badge>
          <Badge variant="neutral" className="text-[10px]">
            assumption-based, unvalidated
          </Badge>
          <Badge variant="info" className="text-[10px]">
            approximate nutrient values
          </Badge>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Controls Column (5 cols) */}
        <div className="lg:col-span-5 bg-surface border border-border rounded-card p-4 space-y-4">
          {/* Patient Selector */}
          <div>
            <label className="text-xs font-semibold text-text-secondary block mb-1 uppercase tracking-tight">
              Select Patient Digital Twin
            </label>
            <select
              value={selectedPatientId}
              onChange={(e) => setSelectedPatientId(e.target.value)}
              className="w-full bg-surface-muted border border-border rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:ring-1 focus:ring-inset focus:ring-brand"
            >
              {patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} ({p.id}) {p.region ? `(${p.region})` : ""} HbA1c {p.hba1c}%
                </option>
              ))}
            </select>
          </div>

          {/* Model Mode Toggle & Compare Mode Toggle */}
          <div className="grid grid-cols-2 gap-2 pt-1 border-t border-border">
            <div className="bg-surface-muted p-2 rounded-lg border border-border flex items-center justify-between">
              <div>
                <span className="text-[11px] font-semibold text-text-primary block">Compare 2 Scenarios</span>
                <span className="text-[9px] text-text-muted">Side-by-side curves</span>
              </div>
              <input
                type="checkbox"
                checked={compareMode}
                onChange={(e) => setCompareMode(e.target.checked)}
                className="h-4 w-4 rounded border-border text-brand focus:ring-brand accent-brand cursor-pointer"
              />
            </div>

            <div className="bg-surface-muted p-2 rounded-lg border border-border flex items-center justify-between">
              <div>
                <span className="text-[11px] font-semibold text-text-primary block">Linear Calibration</span>
                <span className="text-[9px] text-text-muted">Default: raw model</span>
              </div>
              <input
                type="checkbox"
                checked={calibrateModel}
                onChange={(e) => setCalibrateModel(e.target.checked)}
                className="h-4 w-4 rounded border-border text-brand focus:ring-brand accent-brand cursor-pointer"
              />
            </div>
          </div>

          {/* Scenario A Controls */}
          <div className="space-y-3 p-3 rounded-lg border border-border bg-surface-muted/40">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-text-primary flex items-center gap-1.5">
                <Utensils size={14} strokeWidth={1.75} className="text-brand" />
                {compareMode ? "Scenario A: Intervention" : "Intervention Meal & Activity"}
              </span>
              <span className="text-[10px] text-text-muted">Indian Meal Library</span>
            </div>

            <div>
              <label className="text-[11px] font-medium text-text-secondary block mb-1">
                Regional Dish
              </label>
              <select
                value={selectedFoodId}
                onChange={(e) => setSelectedFoodId(e.target.value)}
                className="w-full bg-surface border border-border rounded px-2.5 py-1.5 text-xs text-text-primary focus:outline-none focus:ring-1 focus:ring-brand"
              >
                {foods.map((food) => (
                  <option key={food.id} value={food.id}>
                    {food.name} ({food.regional_category}) GI: {food.glycemic_index}
                  </option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="text-[11px] font-medium text-text-secondary block mb-1">
                  Portion / Servings
                </label>
                <div className="flex items-center gap-1">
                  {[0.5, 1.0, 1.5, 2.0].map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => setServings(s)}
                      className={`flex-1 py-1 rounded text-[11px] font-medium border transition-colors ${
                        servings === s
                          ? "bg-brand text-white border-brand font-bold"
                          : "bg-surface text-text-secondary border-border hover:bg-surface-muted"
                      }`}
                    >
                      {s}x
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-[11px] font-medium text-text-secondary flex items-center gap-1">
                    <Footprints size={12} strokeWidth={1.75} className="text-brand" />
                    <span>Walk</span>
                    <span className="text-[10px] text-text-muted font-normal">(assumption-based)</span>
                  </label>
                  <span className="text-[11px] font-bold text-brand tnum">{activityMinutes}m</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={60}
                  step={5}
                  value={activityMinutes}
                  onChange={(e) => setActivityMinutes(Number(e.target.value))}
                  className="w-full h-1 bg-border rounded appearance-none cursor-pointer accent-brand mt-1.5"
                />
              </div>
            </div>

            {selectedFood && (
              <div className="bg-surface p-2 rounded border border-border text-[10px] grid grid-cols-4 gap-1 text-center tnum">
                <div>
                  <span className="text-text-muted block">Carbs:</span>
                  <strong className="text-text-primary">{((selectedFood.carbohydrates_g || 0) * servings).toFixed(1)}g</strong>
                </div>
                <div>
                  <span className="text-text-muted block">Fiber:</span>
                  <strong className="text-text-primary">{((selectedFood.fiber_g || 0) * servings).toFixed(1)}g</strong>
                </div>
                <div>
                  <span className="text-text-muted block">Protein:</span>
                  <strong className="text-text-primary">{((selectedFood.protein_g || 0) * servings).toFixed(1)}g</strong>
                </div>
                <div>
                  <span className="text-text-muted block">GI:</span>
                  <strong className="text-high">{selectedFood.glycemic_index}</strong>
                </div>
              </div>
            )}
          </div>

          {/* Scenario B Controls (Only visible in Compare Mode) */}
          {compareMode && (
            <div className="space-y-3 p-3 rounded-lg border border-amber/30 bg-amber-subtle/30">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-amber flex items-center gap-1.5">
                  <Scale size={14} strokeWidth={1.75} />
                  Scenario B: Alternative Comparison
                </span>
                <span className="text-[10px] text-text-muted">Alternative Option</span>
              </div>

              <div>
                <label className="text-[11px] font-medium text-text-secondary block mb-1">
                  Alternative Dish
                </label>
                <select
                  value={foodBId}
                  onChange={(e) => setFoodBId(e.target.value)}
                  className="w-full bg-surface border border-border rounded px-2.5 py-1.5 text-xs text-text-primary focus:outline-none focus:ring-1 focus:ring-brand"
                >
                  {foods.map((food) => (
                    <option key={food.id} value={food.id}>
                      {food.name} ({food.regional_category}) GI: {food.glycemic_index}
                    </option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-[11px] font-medium text-text-secondary block mb-1">
                    Portion / Servings
                  </label>
                  <div className="flex items-center gap-1">
                    {[0.5, 1.0, 1.5, 2.0].map((s) => (
                      <button
                        key={s}
                        type="button"
                        onClick={() => setServingsB(s)}
                        className={`flex-1 py-1 rounded text-[11px] font-medium border transition-colors ${
                          servingsB === s
                            ? "bg-amber text-white border-amber font-bold"
                            : "bg-surface text-text-secondary border-border hover:bg-surface-muted"
                        }`}
                      >
                        {s}x
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <div className="flex justify-between items-center mb-1">
                    <label className="text-[11px] font-medium text-text-secondary flex items-center gap-1">
                      <Footprints size={12} strokeWidth={1.75} className="text-amber" />
                      <span>Walk</span>
                      <span className="text-[10px] text-text-muted font-normal">(assumption-based)</span>
                    </label>
                    <span className="text-[11px] font-bold text-amber tnum">{activityBMinutes}m</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={60}
                    step={5}
                    value={activityBMinutes}
                    onChange={(e) => setActivityBMinutes(Number(e.target.value))}
                    className="w-full h-1 bg-border rounded appearance-none cursor-pointer accent-amber mt-1.5"
                  />
                </div>
              </div>

              {selectedFoodB && (
                <div className="bg-surface p-2 rounded border border-border text-[10px] grid grid-cols-4 gap-1 text-center tnum">
                  <div>
                    <span className="text-text-muted block">Carbs:</span>
                    <strong className="text-text-primary">{((selectedFoodB.carbohydrates_g || 0) * servingsB).toFixed(1)}g</strong>
                  </div>
                  <div>
                    <span className="text-text-muted block">Fiber:</span>
                    <strong className="text-text-primary">{((selectedFoodB.fiber_g || 0) * servingsB).toFixed(1)}g</strong>
                  </div>
                  <div>
                    <span className="text-text-muted block">Protein:</span>
                    <strong className="text-text-primary">{((selectedFoodB.protein_g || 0) * servingsB).toFixed(1)}g</strong>
                  </div>
                  <div>
                    <span className="text-text-muted block">GI:</span>
                    <strong className="text-high">{selectedFoodB.glycemic_index}</strong>
                  </div>
                </div>
              )}
            </div>
          )}

          <Button
            onClick={handleSimulate}
            disabled={isSimulating}
            className="w-full mt-2 flex items-center justify-center gap-2"
          >
            <span>{isSimulating ? "Simulating Dynamics..." : "Run What-If Simulation"}</span>
            <ChevronRight size={16} strokeWidth={1.75} />
          </Button>
        </div>

        {/* Results Column (7 cols) */}
        <div className="lg:col-span-7 bg-surface border border-border rounded-card p-4 space-y-4">
          <div className="flex items-center justify-between border-b border-border pb-3">
            <div>
              <h2 className="text-sm font-semibold text-text-primary">Counterfactual Trajectory Comparison</h2>
              <p className="text-[10px] text-text-muted">Postprandial 180-minute projection with isotonic constraint (passed, by construction)</p>
              {simulationA?.as_of && (
                <div className="flex items-center gap-3 text-[10px] text-text-secondary mt-1">
                  <span>Simulated as of: <strong className="text-text-primary tnum">{simulationA.as_of.includes("T") ? simulationA.as_of.replace("T", " ").slice(0, 16) : simulationA.as_of}</strong></span>
                  {simulationA.starting_glucose != null && (
                    <span>Starting glucose: <strong className="text-text-primary tnum">{simulationA.starting_glucose.toFixed(1)} mg/dL</strong></span>
                  )}
                </div>
              )}
            </div>
            {simulationA && (
              <Badge variant="success" className="flex items-center gap-1">
                <Sparkles size={14} strokeWidth={1.75} />
                <span className="tnum">
                  Peak Shift: {peakDeltaA >= 0 ? `+${peakDeltaA.toFixed(1)}` : peakDeltaA.toFixed(1)} mg/dL
                </span>
              </Badge>
            )}
          </div>

          {simulationA ? (
            <div className="space-y-4">
              {/* Delta Chips for Scenarios */}
              <div className="space-y-2.5">
                {/* Scenario A Row */}
                <div className="space-y-1">
                  <div className="text-[11px] font-semibold text-[#0284C7] flex items-center justify-between">
                    <span>Scenario A (vs Baseline):</span>
                    {activityMinutes > 0 && (
                      <span className="text-[10px] text-text-muted font-normal flex items-center gap-1">
                        <Footprints size={11} className="text-brand" /> Walk {activityMinutes}m (assumption-based)
                      </span>
                    )}
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    <div className="bg-surface-muted p-2 rounded-lg border border-border text-center">
                      <span className="text-[10px] text-text-muted block">Scenario A +30m Delta</span>
                      <span className={`text-xs font-bold tnum ${delta30A > 0 ? "text-severe" : "text-in-range"}`}>
                        {delta30A >= 0 ? `+${delta30A.toFixed(1)}` : delta30A.toFixed(1)} mg/dL
                      </span>
                    </div>
                    <div className="bg-surface-muted p-2 rounded-lg border border-border text-center">
                      <span className="text-[10px] text-text-muted block">Scenario A +60m Delta</span>
                      <span className={`text-xs font-bold tnum ${delta60A > 0 ? "text-severe" : "text-in-range"}`}>
                        {delta60A >= 0 ? `+${delta60A.toFixed(1)}` : delta60A.toFixed(1)} mg/dL
                      </span>
                    </div>
                    <div className="bg-surface-muted p-2 rounded-lg border border-border text-center">
                      <span className="text-[10px] text-text-muted block">Scenario A +120m Delta</span>
                      <span className={`text-xs font-bold tnum ${delta120A > 0 ? "text-severe" : "text-in-range"}`}>
                        {delta120A >= 0 ? `+${delta120A.toFixed(1)}` : delta120A.toFixed(1)} mg/dL
                      </span>
                    </div>
                    <div className="bg-surface-muted p-2 rounded-lg border border-border text-center">
                      <span className="text-[10px] text-text-muted block">Scenario A Peak Delta</span>
                      <span className={`text-xs font-bold tnum ${peakDeltaA > 0 ? "text-severe" : "text-in-range"}`}>
                        {peakDeltaA >= 0 ? `+${peakDeltaA.toFixed(1)}` : peakDeltaA.toFixed(1)} mg/dL
                      </span>
                    </div>
                  </div>
                </div>

                {/* Scenario B Row & B minus A Row (in compare mode) */}
                {compareMode && simulationB && (
                  <>
                    <div className="space-y-1">
                      <div className="text-[11px] font-semibold text-[#D97706] flex items-center justify-between">
                        <span>Scenario B (vs Baseline):</span>
                        {activityBMinutes > 0 && (
                          <span className="text-[10px] text-text-muted font-normal flex items-center gap-1">
                            <Footprints size={11} className="text-amber" /> Walk {activityBMinutes}m (assumption-based)
                          </span>
                        )}
                      </div>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                        <div className="bg-surface-muted p-2 rounded-lg border border-amber/30 text-center">
                          <span className="text-[10px] text-text-muted block">Scenario B +30m Delta</span>
                          <span className={`text-xs font-bold tnum ${delta30B > 0 ? "text-severe" : "text-in-range"}`}>
                            {delta30B >= 0 ? `+${delta30B.toFixed(1)}` : delta30B.toFixed(1)} mg/dL
                          </span>
                        </div>
                        <div className="bg-surface-muted p-2 rounded-lg border border-amber/30 text-center">
                          <span className="text-[10px] text-text-muted block">Scenario B +60m Delta</span>
                          <span className={`text-xs font-bold tnum ${delta60B > 0 ? "text-severe" : "text-in-range"}`}>
                            {delta60B >= 0 ? `+${delta60B.toFixed(1)}` : delta60B.toFixed(1)} mg/dL
                          </span>
                        </div>
                        <div className="bg-surface-muted p-2 rounded-lg border border-amber/30 text-center">
                          <span className="text-[10px] text-text-muted block">Scenario B +120m Delta</span>
                          <span className={`text-xs font-bold tnum ${delta120B > 0 ? "text-severe" : "text-in-range"}`}>
                            {delta120B >= 0 ? `+${delta120B.toFixed(1)}` : delta120B.toFixed(1)} mg/dL
                          </span>
                        </div>
                        <div className="bg-surface-muted p-2 rounded-lg border border-amber/30 text-center">
                          <span className="text-[10px] text-text-muted block">Scenario B Peak Delta</span>
                          <span className={`text-xs font-bold tnum ${peakDeltaB > 0 ? "text-severe" : "text-in-range"}`}>
                            {peakDeltaB >= 0 ? `+${peakDeltaB.toFixed(1)}` : peakDeltaB.toFixed(1)} mg/dL
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* B minus A Row */}
                    <div className="space-y-1 pt-1 border-t border-border">
                      <div className="text-[11px] font-semibold text-text-primary flex items-center justify-between">
                        <span>B minus A (Intervention Difference):</span>
                        <span className="text-[10px] text-amber font-normal">walking effect is the assumed deduction, not a learned effect</span>
                      </div>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                        <div className="bg-surface-muted p-2 rounded-lg border border-border text-center">
                          <span className="text-[10px] text-text-muted block">B minus A (+30m)</span>
                          <span className={`text-xs font-bold tnum ${delta30Diff < 0 ? "text-in-range" : "text-text-primary"}`}>
                            {delta30Diff >= 0 ? `+${delta30Diff.toFixed(1)}` : delta30Diff.toFixed(1)} mg/dL
                          </span>
                        </div>
                        <div className="bg-surface-muted p-2 rounded-lg border border-border text-center">
                          <span className="text-[10px] text-text-muted block">B minus A (+60m)</span>
                          <span className={`text-xs font-bold tnum ${delta60Diff < 0 ? "text-in-range" : "text-text-primary"}`}>
                            {delta60Diff >= 0 ? `+${delta60Diff.toFixed(1)}` : delta60Diff.toFixed(1)} mg/dL
                          </span>
                        </div>
                        <div className="bg-surface-muted p-2 rounded-lg border border-border text-center">
                          <span className="text-[10px] text-text-muted block">B minus A (+120m)</span>
                          <span className={`text-xs font-bold tnum ${delta120Diff < 0 ? "text-in-range" : "text-text-primary"}`}>
                            {delta120Diff >= 0 ? `+${delta120Diff.toFixed(1)}` : delta120Diff.toFixed(1)} mg/dL
                          </span>
                        </div>
                        <div className="bg-surface-muted p-2 rounded-lg border border-border text-center">
                          <span className="text-[10px] text-text-muted block">B minus A (Peak Shift)</span>
                          <span className={`text-xs font-bold tnum ${peakDeltaDiff < 0 ? "text-in-range" : "text-text-primary"}`}>
                            {peakDeltaDiff >= 0 ? `+${peakDeltaDiff.toFixed(1)}` : peakDeltaDiff.toFixed(1)} mg/dL
                          </span>
                        </div>
                      </div>
                    </div>
                  </>
                )}
              </div>

              {/* Curve and Fan Legend */}
              <div className="flex items-center justify-center gap-3 sm:gap-5 flex-wrap text-[11px] py-1.5 px-3 bg-surface-muted/60 rounded border border-border">
                <div className="flex items-center gap-1.5">
                  <span className="w-4 h-0 border-t-2 border-dashed border-slate-500 inline-block"></span>
                  <span className="text-text-secondary font-medium">Baseline</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-4 h-0 border-t-2 border-solid border-[#0284C7] inline-block"></span>
                  <span className="text-[#0284C7] font-medium">Scenario A</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-3.5 h-3 bg-[#0284C7]/20 border border-[#0284C7]/40 rounded-sm inline-block"></span>
                  <span className="text-text-muted">Scenario A Fan</span>
                </div>
                {compareMode && (
                  <>
                    <div className="flex items-center gap-1.5">
                      <span className="w-4 h-0 border-t-2 border-dashed border-[#D97706] inline-block"></span>
                      <span className="text-[#D97706] font-medium">Scenario B</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="w-3.5 h-3 bg-[#D97706]/20 border border-[#D97706]/40 rounded-sm inline-block"></span>
                      <span className="text-text-muted">Scenario B Fan</span>
                    </div>
                  </>
                )}
              </div>

              {/* Simulation Curve Chart */}
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart
                    data={chartData}
                    margin={{ top: 10, right: 30, left: 10, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="2 2" stroke="#E3E8F0" vertical={false} />
                    <XAxis
                      type="number"
                      dataKey="minuteOffset"
                      domain={[0, 180]}
                      ticks={[0, 30, 60, 90, 120, 150, 180]}
                      tickFormatter={(v) => (v === 0 ? "Meal" : `+${v}m`)}
                      stroke="#64748B"
                      tick={{ fontSize: 11 }}
                    />
                    <YAxis
                      domain={[60, 260]}
                      ticks={[70, 140, 180, 220, 260]}
                      stroke="#64748B"
                      tick={{ fontSize: 11 }}
                      label={{
                        value: "mg/dL",
                        angle: -90,
                        position: "insideLeft",
                        fill: "#64748B",
                        fontSize: 10,
                        fontWeight: 600,
                      }}
                    />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const d = payload[0].payload as SimulationChartPoint;
                          return (
                            <div className="bg-surface border border-border-strong rounded-lg p-2.5 text-xs space-y-1">
                              <div className="font-semibold text-text-primary">
                                +{d.minuteOffset} min Postprandial
                              </div>
                              {d.baseline !== undefined && (
                                <div className="text-text-muted">
                                  Baseline: <strong>{d.baseline.toFixed(1)} mg/dL</strong>
                                </div>
                              )}
                              {d.scenarioA !== undefined && (
                                <div className="text-brand">
                                  Scenario A: <strong>{d.scenarioA.toFixed(1)} mg/dL</strong>
                                </div>
                              )}
                              {d.fanRangeA && (
                                <div className="text-[10px] text-text-secondary">
                                  A Uncertainty (q10-q90): {d.fanRangeA[0].toFixed(0)} to {d.fanRangeA[1].toFixed(0)} mg/dL
                                </div>
                              )}
                              {d.scenarioB !== undefined && (
                                <div className="text-amber">
                                  Scenario B: <strong>{d.scenarioB.toFixed(1)} mg/dL</strong>
                                </div>
                              )}
                            </div>
                          );
                        }
                        return null;
                      }}
                    />

                    {/* Target range */}
                    <ReferenceArea y1={70} y2={180} fill="#10B981" fillOpacity={0.06} />
                    <ReferenceLine y={180} stroke="#F59E0B" strokeDasharray="3 3" />
                    <ReferenceLine y={70} stroke="#7C3AED" strokeDasharray="3 3" />

                    {/* Scenario A Quantile Fan */}
                    <Area
                      type="monotone"
                      dataKey="fanRangeA"
                      stroke="transparent"
                      fill="#0284C7"
                      fillOpacity={0.16}
                      isAnimationActive={false}
                      name="Scenario A Fan"
                    />

                    {/* Scenario B Quantile Fan if compareMode */}
                    {compareMode && (
                      <Area
                        type="monotone"
                        dataKey="fanRangeB"
                        stroke="transparent"
                        fill="#D97706"
                        fillOpacity={0.12}
                        isAnimationActive={false}
                        name="Scenario B Fan"
                      />
                    )}

                    {/* Baseline trajectory */}
                    <Line
                      type="monotone"
                      dataKey="baseline"
                      stroke="#64748B"
                      strokeDasharray="4 4"
                      strokeWidth={1.75}
                      dot={false}
                      isAnimationActive={false}
                      name="Baseline Trajectory"
                    />

                    {/* Scenario A trajectory */}
                    <Line
                      type="monotone"
                      dataKey="scenarioA"
                      stroke="#0284C7"
                      strokeWidth={2.5}
                      dot={false}
                      animationDuration={600}
                      name="Scenario A"
                    />

                    {/* Scenario B trajectory if compareMode */}
                    {compareMode && (
                      <Line
                        type="monotone"
                        dataKey="scenarioB"
                        stroke="#D97706"
                        strokeWidth={2}
                        strokeDasharray="2 2"
                        dot={false}
                        animationDuration={600}
                        name="Scenario B"
                      />
                    )}
                  </ComposedChart>
                </ResponsiveContainer>
              </div>

              {/* Takeaway message */}
              <div className="p-3 bg-brand/5 border border-brand/20 rounded-lg text-xs flex items-start gap-2">
                <CheckCircle2 size={16} strokeWidth={1.75} className="text-brand shrink-0 mt-0.5" />
                <div className="space-y-0.5">
                  <span className="font-semibold text-text-primary block">Takeaway:</span>
                  <p className="text-text-secondary leading-relaxed">{simulationA.clinical_takeaway}</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-64 flex flex-col items-center justify-center border border-dashed border-border rounded-lg text-text-muted text-xs gap-2">
              <FlaskConical size={24} strokeWidth={1.75} className="text-text-muted opacity-50" />
              <span>Select simulation parameters and click Run What-If Simulation</span>
            </div>
          )}

          {/* Dynamic What-If Validation Card (Real Numbers from reports/whatif_validation.json) */}
          <div className="border-t border-border pt-3 space-y-2 text-xs">
            <div className="flex items-center gap-1.5 font-semibold text-text-primary text-[11px]">
              <Info size={14} strokeWidth={1.75} className="text-brand" />
              What-If Model Empirical Validation (reports/whatif_validation.json)
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 bg-surface-muted p-2.5 rounded-lg border border-border text-[10px] tnum">
              <div>
                <span className="text-text-muted block">Cross-Fold Mean Pearson r:</span>
                <strong className="text-in-range text-[11px]">{cfMean} ± {cfStd}</strong>
                <span className="text-[9px] text-text-muted block">(target {targetPearson})</span>
              </div>
              <div>
                <span className="text-text-muted block">Pooled Pearson r:</span>
                <strong className="text-text-primary text-[11px]">{pooledPearson}</strong>
              </div>
              <div>
                <span className="text-text-muted block">T2D Cohort Pearson r:</span>
                <strong className="text-text-primary text-[11px]">{t2dPearson}</strong>
              </div>
              <div>
                <span className="text-text-muted block">T2D Pred vs Obs Rise:</span>
                <strong className="text-text-primary text-[11px]">{t2dPredRise} vs {t2dObsRise} mg/dL</strong>
              </div>
            </div>

            <p className="text-[10px] text-text-muted leading-relaxed">
              <strong>Underestimation notice:</strong> The raw ML model exhibits high-carbohydrate saturation and underestimates postprandial amplitude in T2D patients ({t2dPredRise} mg/dL predicted vs {t2dObsRise} mg/dL observed across {t2dMeals} meals). However, directional sensitivity is preserved with a {t2dShare} percent positive-effect share. Agreement figures: cross-fold mean Pearson r = {cfMean}, pooled Pearson r = {pooledPearson}, and T2D cohort Pearson r = {t2dPearson}. Exercise clearance is modeled at {clearanceRate} mg/dL per net MET-minute ({clearanceStatus}).
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
