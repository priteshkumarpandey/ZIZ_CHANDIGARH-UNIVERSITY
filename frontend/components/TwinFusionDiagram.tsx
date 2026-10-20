import React from "react";
import {
  GitMerge,
  FileText,
  Activity,
  Cpu,
  TrendingUp,
  AlertTriangle,
  FlaskConical,
} from "lucide-react";
import { Card, CardHeader, CardTitle } from "./ui/card";

export const TwinFusionDiagram: React.FC = () => {
  return (
    <Card className="w-full">
      <CardHeader>
        <div>
          <CardTitle>
            <GitMerge size={16} strokeWidth={1.75} className="text-brand" />
            Digital Twin Fusion Pipeline
          </CardTitle>
          <p className="text-xs text-text-secondary mt-0.5">
            Static EHR priors fused with real-time wearable streams and circadian time-step features
          </p>
        </div>
      </CardHeader>

      <div className="w-full flex justify-center py-2">
        <div className="w-full max-w-4xl overflow-x-auto px-2">
          <svg
            viewBox="0 0 880 185"
            className="w-full h-auto min-w-[760px] text-xs font-sans"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            {/* Static EHR Prior Box */}
            <rect x="20" y="20" width="215" height="60" rx="8" fill="#F1F4F9" stroke="#CAD4E0" strokeWidth="1" />
            <foreignObject x="32" y="28" width="20" height="20">
              <FileText size={16} strokeWidth={1.75} className="text-text-primary" />
            </foreignObject>
            <text x="56" y="42" fill="#0F172A" fontWeight="600" fontSize="12">EHR Demographics & Lab</text>
            <text x="32" y="62" fill="#64748B" fontSize="11">Age, BMI, HbA1c, Meds (16 feats)</text>

            {/* Wearable Stream Box */}
            <rect x="20" y="100" width="215" height="60" rx="8" fill="#F1F4F9" stroke="#CAD4E0" strokeWidth="1" />
            <foreignObject x="32" y="108" width="20" height="20">
              <Activity size={16} strokeWidth={1.75} className="text-[#0369A1]" />
            </foreignObject>
            <text x="56" y="122" fill="#0F172A" fontWeight="600" fontSize="12">Wearable Stream</text>
            <text x="32" y="142" fill="#64748B" fontSize="11">CGM (72 pts), HR, Steps, METs</text>

            {/* Connecting Arrows to Fusion Core */}
            <path d="M235 50 L320 82" stroke="#94A3B8" strokeWidth="1.5" strokeDasharray="3 3" />
            <path d="M235 130 L320 98" stroke="#94A3B8" strokeWidth="1.5" />

            {/* Fusion Core Box */}
            <rect x="320" y="55" width="220" height="70" rx="8" fill="#E0F2FE" stroke="#0369A1" strokeWidth="1.5" />
            <foreignObject x="334" y="66" width="24" height="24">
              <Cpu size={20} strokeWidth={1.75} className="text-[#0369A1]" />
            </foreignObject>
            <text x="362" y="80" fill="#0369A1" fontWeight="700" fontSize="13">Digital Twin Core</text>
            <text x="334" y="98" fill="#0284C7" fontSize="11">Multimodal GRU + Fourier Time</text>
            <text x="334" y="112" fill="#64748B" fontSize="10">Calibrated LoRA Adapter</text>

            {/* Connecting Arrows to Outputs */}
            <path d="M540 80 L620 43" stroke="#94A3B8" strokeWidth="1.5" />
            <path d="M540 90 L620 96" stroke="#94A3B8" strokeWidth="1.5" />
            <path d="M540 100 L620 149" stroke="#94A3B8" strokeWidth="1.5" />

            {/* Output 1: Quantile Trajectory (wide enough to contain all text) */}
            <rect x="620" y="20" width="240" height="46" rx="6" fill="#F1F4F9" stroke="#CAD4E0" strokeWidth="1" />
            <foreignObject x="630" y="26" width="20" height="20">
              <TrendingUp size={16} strokeWidth={1.75} className="text-[#0369A1]" />
            </foreignObject>
            <text x="654" y="38" fill="#0F172A" fontWeight="600" fontSize="10.5">Quantile Forecast (q10, q50, q90)</text>
            <text x="654" y="52" fill="#64748B" fontSize="10">+30m / +60m / +120m trajectories</text>

            {/* Output 2: Gated Alerts (Thresholds tuned on validation patients) */}
            <rect x="620" y="73" width="240" height="46" rx="6" fill="#FEF2F2" stroke="#DC2626" strokeWidth="1" />
            <foreignObject x="630" y="79" width="20" height="20">
              <AlertTriangle size={16} strokeWidth={1.75} className="text-[#DC2626]" />
            </foreignObject>
            <text x="654" y="91" fill="#DC2626" fontWeight="600" fontSize="10.5">Gated Adverse Event Alerts</text>
            <text x="654" y="105" fill="#64748B" fontSize="10">Thresholds tuned on validation patients</text>

            {/* Output 3: Counterfactual Simulation */}
            <rect x="620" y="126" width="240" height="46" rx="6" fill="#ECFDF5" stroke="#10B981" strokeWidth="1" />
            <foreignObject x="630" y="132" width="20" height="20">
              <FlaskConical size={16} strokeWidth={1.75} className="text-[#047857]" />
            </foreignObject>
            <text x="654" y="144" fill="#047857" fontWeight="600" fontSize="10.5">What-If Counterfactuals</text>
            <text x="654" y="158" fill="#64748B" fontSize="10">Indian meal library & activity</text>
          </svg>
        </div>
      </div>
    </Card>
  );
};
