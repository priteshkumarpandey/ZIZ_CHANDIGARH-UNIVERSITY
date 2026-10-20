import React, { useState, useEffect } from "react";
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
import { FlaskConical, Utensils, Footprints, Sparkles, CheckCircle2, ChevronRight } from "lucide-react";
import { FoodItem, SimulationResponse, PatientProfile } from "../lib/types";
import { fetchFoods, runSimulation } from "../lib/api";
import { useGlucoTwinStore } from "../lib/store";
import { Card, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Slider } from "./ui/slider";
import { Badge } from "./ui/badge";

export const WhatIfSimulator: React.FC = () => {
  const { selectedPatient, cgmHistory, simulation, setSimulation } = useGlucoTwinStore();
  const [foods, setFoods] = useState<FoodItem[]>([]);
  const [selectedFoodId, setSelectedFoodId] = useState<string>("ifct_002"); // White rice default
  const [activityMinutes, setActivityMinutes] = useState<number>(20);
  const [activityType, setActivityType] = useState<string>("Post-meal Brisk Walk");
  const [isSimulating, setIsSimulating] = useState<boolean>(false);

  useEffect(() => {
    fetchFoods()
      .then((items) => setFoods(items))
      .catch((err) => console.error("Could not load foods", err));
  }, []);

  const handleRunSimulation = async () => {
    if (!selectedPatient) return;
    setIsSimulating(true);
    try {
      const result = await runSimulation(
        selectedPatient.id,
        selectedFoodId,
        activityMinutes
      );
      setSimulation(result);
    } catch (err) {
      console.error("Simulation error", err);
    } finally {
      setIsSimulating(false);
    }
  };

  const selectedFood = foods.find((f) => f.id === selectedFoodId);

  // Format chart data
  const chartData = simulation
    ? simulation.timestamps.map((ts, idx) => ({
        time: `+${idx * 5}m`,
        baseline: simulation.baseline_trajectory[idx],
        scenario: simulation.scenario_trajectory[idx],
        q10: simulation.scenario_q10[idx],
        q90: simulation.scenario_q90[idx],
      }))
    : [];

  return (
    <Card className="w-full">
      <CardHeader>
        <div>
          <CardTitle>
            <FlaskConical size={16} strokeWidth={1.75} className="text-brand" />
            Physiology-Constrained What-If Simulator
          </CardTitle>
          <p className="text-xs text-text-secondary mt-0.5">
            Evaluate counterfactual Indian meal compositions and postprandial activity interventions
          </p>
        </div>

        {simulation && (
          <Badge variant="success" className="flex items-center gap-1">
            <Sparkles size={16} strokeWidth={1.75} />
            Peak Blunted by {simulation.peak_reduction_mgdl} mg/dL
          </Badge>
        )}
      </CardHeader>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls Column */}
        <div className="space-y-4 bg-surface-muted p-4 rounded-lg border border-border">
          <div>
            <label className="text-xs font-semibold text-text-primary flex items-center gap-1.5 mb-1.5">
              <Utensils size={16} strokeWidth={1.75} className="text-brand" />
              Select Regional Indian Meal (Indian Meal Library)
            </label>
            <select
              value={selectedFoodId}
              onChange={(e) => setSelectedFoodId(e.target.value)}
              className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:ring-1 focus:ring-inset focus:ring-brand font-medium"
            >
              {foods.map((food) => (
                <option key={food.id} value={food.id}>
                  {food.name} ({food.regional_category}) - GI: {food.glycemic_index}
                </option>
              ))}
            </select>
          </div>

          {selectedFood && (
            <div className="bg-surface p-3 rounded-md border border-border text-xs space-y-1.5 text-text-secondary">
              <div className="flex justify-between">
                <span className="text-text-muted">Serving Size:</span>
                <span className="font-medium text-text-primary">{selectedFood.serving_size_g}g</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Available Carbs:</span>
                <span className="font-medium text-text-primary">{selectedFood.carbohydrates_g}g</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Dietary Fiber:</span>
                <span className="font-medium text-text-primary">{selectedFood.fiber_g}g</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Glycemic Index (GI):</span>
                <Badge variant={selectedFood.glycemic_index > 70 ? "danger" : (selectedFood.glycemic_index > 55 ? "warning" : "success")}>
                  {selectedFood.glycemic_index}
                </Badge>
              </div>
            </div>
          )}

          <div className="pt-2 border-t border-border space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-text-primary flex items-center gap-1.5">
                <Footprints size={16} strokeWidth={1.75} className="text-brand" />
                Post-Meal Activity Intervention
              </span>
            </div>

            <Slider
              label="Brisk Walk Duration"
              valueDisplay={`${activityMinutes} mins`}
              min={0}
              max={60}
              step={5}
              value={activityMinutes}
              onChange={(e) => setActivityMinutes(Number(e.target.value))}
            />
          </div>

          <Button
            onClick={handleRunSimulation}
            disabled={isSimulating}
            className="w-full flex items-center justify-center gap-2 mt-4"
          >
            {isSimulating ? "Simulating Physiology..." : "Run Twin Simulation"}
            <ChevronRight size={16} strokeWidth={1.75} />
          </Button>
        </div>

        {/* Visualization Column */}
        <div className="lg:col-span-2 space-y-4">
          <div className="h-64 w-full">
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E3E8F0" vertical={false} />
                  <XAxis dataKey="time" stroke="#64748B" tick={{ fontSize: 11 }} />
                  <YAxis domain={[80, 260]} stroke="#64748B" tick={{ fontSize: 11 }} unit=" mg/dL" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#FFFFFF",
                      borderColor: "#CAD4E0",
                      borderRadius: "8px",
                      fontSize: "12px",
                      boxShadow: "none",
                      color: "#0F172A",
                    }}
                  />
                  <ReferenceArea y1={70} y2={180} fill="#10B981" fillOpacity={0.08} />
                  <ReferenceLine y={180} stroke="#DC2626" strokeDasharray="3 3" strokeOpacity={0.6} />

                  {/* Scenario 90% Quantile Ribbon */}
                  <Area
                    type="monotone"
                    dataKey="q90"
                    stroke="transparent"
                    fill="#10B981"
                    fillOpacity={0.15}
                    name="Scenario 90th %ile"
                  />
                  <Area
                    type="monotone"
                    dataKey="q10"
                    stroke="transparent"
                    fill="#10B981"
                    fillOpacity={0.0}
                    name="Scenario 10th %ile"
                  />

                  {/* Baseline curve (sedentary / unmodified) */}
                  <Line
                    type="monotone"
                    dataKey="baseline"
                    stroke="#DC2626"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={false}
                    name="Baseline Trajectory (mg/dL)"
                  />

                  {/* Simulated scenario curve */}
                  <Line
                    type="monotone"
                    dataKey="scenario"
                    stroke="#10B981"
                    strokeWidth={2.5}
                    dot={false}
                    name="Intervention Trajectory (mg/dL)"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex flex-col items-center justify-center border border-dashed border-border rounded-lg text-text-muted text-xs gap-2">
                <FlaskConical size={24} strokeWidth={1.75} className="text-text-muted" />
                <span>Select a meal and activity parameter, then click Run Twin Simulation</span>
              </div>
            )}
          </div>

          {simulation && (
            <div className="p-3.5 bg-brand-subtle border border-brand/30 rounded-lg text-xs flex items-start gap-2.5">
              <CheckCircle2 size={16} strokeWidth={1.75} className="text-in-range shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold text-text-primary">Clinical Decision Insight: </span>
                <span className="text-text-secondary">{simulation.clinical_takeaway}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
};
