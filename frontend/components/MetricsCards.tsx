import React from "react";
import { Activity, Target, ArrowUpRight, ArrowDownRight, Gauge, Zap } from "lucide-react";
import { ClinicalMetrics } from "../lib/types";
import { Card } from "./ui/card";
import { Badge } from "./ui/badge";

interface MetricsCardsProps {
  metrics: ClinicalMetrics | null;
}

export const MetricsCards: React.FC<MetricsCardsProps> = ({ metrics }) => {
  if (!metrics) return null;

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
      <Card className="p-3.5 flex flex-col justify-between">
        <div className="flex items-center justify-between text-text-secondary text-xs">
          <span>Time in Range</span>
          <Target size={16} strokeWidth={1.75} className="text-in-range" />
        </div>
        <div className="mt-2">
          <div className="text-xl font-bold text-in-range tnum">
            {metrics.time_in_range_percentage}%
          </div>
          <div className="text-[10px] text-text-muted mt-0.5">Target &gt;70% (70-180 mg/dL)</div>
        </div>
      </Card>

      <Card className="p-3.5 flex flex-col justify-between">
        <div className="flex items-center justify-between text-text-secondary text-xs">
          <span>Time Above Range</span>
          <ArrowUpRight size={16} strokeWidth={1.75} className="text-high" />
        </div>
        <div className="mt-2">
          <div className="text-xl font-bold text-high tnum">
            {metrics.time_above_range_percentage}%
          </div>
          <div className="text-[10px] text-text-muted mt-0.5">Target &lt;25% (&gt;180 mg/dL)</div>
        </div>
      </Card>

      <Card className="p-3.5 flex flex-col justify-between">
        <div className="flex items-center justify-between text-text-secondary text-xs">
          <span>Time Below Range</span>
          <ArrowDownRight size={16} strokeWidth={1.75} className="text-low" />
        </div>
        <div className="mt-2">
          <div className="text-xl font-bold text-low tnum">
            {metrics.time_below_range_percentage}%
          </div>
          <div className="text-[10px] text-text-muted mt-0.5">Target &lt;4% (&lt;70 mg/dL)</div>
        </div>
      </Card>

      <Card className="p-3.5 flex flex-col justify-between">
        <div className="flex items-center justify-between text-text-secondary text-xs">
          <span>Mean Glucose</span>
          <Activity size={16} strokeWidth={1.75} className="text-brand" />
        </div>
        <div className="mt-2">
          <div className="text-xl font-bold text-text-primary tnum">
            {metrics.mean_glucose} <span className="text-xs font-normal text-text-secondary">mg/dL</span>
          </div>
          <div className="text-[10px] text-text-muted mt-0.5">SD: ±{metrics.standard_deviation} mg/dL</div>
        </div>
      </Card>

      <Card className="p-3.5 flex flex-col justify-between">
        <div className="flex items-center justify-between text-text-secondary text-xs">
          <span>GMI (Est. HbA1c)</span>
          <Gauge size={16} strokeWidth={1.75} className="text-high" />
        </div>
        <div className="mt-2">
          <div className="text-xl font-bold text-high tnum">
            {metrics.gmi}%
          </div>
          <div className="text-[10px] text-text-muted mt-0.5">Glycemic Indicator</div>
        </div>
      </Card>

      <Card className="p-3.5 flex flex-col justify-between">
        <div className="flex items-center justify-between text-text-secondary text-xs">
          <span>Glycemic CV%</span>
          <Zap size={16} strokeWidth={1.75} className="text-low" />
        </div>
        <div className="mt-2">
          <div className={`text-xl font-bold tnum ${metrics.coefficient_of_variation > 36.0 ? 'text-high' : 'text-in-range'}`}>
            {metrics.coefficient_of_variation}%
          </div>
          <div className="text-[10px] text-text-muted mt-0.5">
            {metrics.coefficient_of_variation > 36.0 ? 'High Variability (>36%)' : 'Stable Variability (<=36%)'}
          </div>
        </div>
      </Card>
    </div>
  );
};
