import React from "react";
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
import { TrendingUp, AlertTriangle } from "lucide-react";
import { CGMReading, ForecastResponse } from "../lib/types";
import { Card, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";

interface GlucoseForecastChartProps {
  cgmHistory: CGMReading[];
  forecast: ForecastResponse | null;
  patientName: string;
  isPersonalized?: boolean;
}

export const GlucoseForecastChart: React.FC<GlucoseForecastChartProps> = ({
  cgmHistory,
  forecast,
  patientName,
  isPersonalized = false,
}) => {
  // Use last 72 readings = 6 hours of 5-min readings
  const recentHistory = cgmHistory.slice(-72);
  const historyLen = recentHistory.length;

  interface ChartPoint {
    minuteOffset: number;
    clockTime?: string;
    actualGlucose?: number;
    q50?: number;
    fanRange?: [number, number];
    isMarker?: boolean;
    markerLabel?: string;
  }

  // 1. History points: minuteOffset from -(historyLen - 1) * 5 up to 0
  const chartData: ChartPoint[] = recentHistory.map((item, idx) => {
    const offset = (idx - (historyLen - 1)) * 5;
    return {
      minuteOffset: offset,
      clockTime: new Date(item.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      actualGlucose: item.glucose_mgdl,
      q50: undefined,
      fanRange: undefined,
      isMarker: false,
    };
  });

  // 2. Forecast points: connect at minuteOffset 0 and extend to +120
  const lastActual = recentHistory.length > 0 ? recentHistory[recentHistory.length - 1].glucose_mgdl : 130;

  if (chartData.length > 0) {
    // Current point at minuteOffset 0 connects both actual and forecast
    const currentPoint = chartData[chartData.length - 1];
    currentPoint.q50 = lastActual;
    currentPoint.fanRange = [lastActual, lastActual];
  }

  if (forecast && forecast.forecast_horizons && forecast.forecast_horizons.length > 0) {
    const horizonsMap = new Map<number, { q10: number; q50: number; q90: number }>();
    horizonsMap.set(0, { q10: lastActual, q50: lastActual, q90: lastActual });
    forecast.forecast_horizons.forEach((h) => {
      horizonsMap.set(h.horizon_minutes, { q10: h.q10, q50: h.q50, q90: h.q90 });
    });

    const knownMins = Array.from(horizonsMap.keys()).sort((a, b) => a - b);

    // Interpolate in 5-minute increments from 5 to 120
    for (let m = 5; m <= 120; m += 5) {
      // Find bounding horizons for linear interpolation
      let lower = knownMins[0];
      let upper = knownMins[knownMins.length - 1];
      for (let k = 0; k < knownMins.length - 1; k++) {
        if (m >= knownMins[k] && m <= knownMins[k + 1]) {
          lower = knownMins[k];
          upper = knownMins[k + 1];
          break;
        }
      }

      const fraction = upper === lower ? 0 : (m - lower) / (upper - lower);
      const lowData = horizonsMap.get(lower)!;
      const upData = horizonsMap.get(upper)!;

      const q10Interp = lowData.q10 + fraction * (upData.q10 - lowData.q10);
      const q50Interp = lowData.q50 + fraction * (upData.q50 - lowData.q50);
      const q90Interp = lowData.q90 + fraction * (upData.q90 - lowData.q90);

      const isKeyMarker = m === 30 || m === 60 || m === 120;

      chartData.push({
        minuteOffset: m,
        clockTime: `+${m}m`,
        actualGlucose: undefined,
        q50: q50Interp,
        fanRange: [q10Interp, q90Interp],
        isMarker: isKeyMarker,
        markerLabel: isKeyMarker ? `${Math.round(q50Interp)}` : undefined,
      });
    }
  }

  // Determine earliest minute forecast crosses 180 threshold
  let hyperCrossingMin: number | null = null;
  if (lastActual < 180) {
    for (const pt of chartData) {
      if (pt.minuteOffset > 0 && pt.q50 !== undefined && pt.q50 >= 180) {
        hyperCrossingMin = pt.minuteOffset;
        break;
      }
    }
  }

  const usableAlerts = forecast?.forecast_horizons?.filter((h) => h.alert_usable && h.predicted_event !== "none") || [];
  const hasUsableAlert = usableAlerts.length > 0;

  return (
    <Card className="w-full">
      <CardHeader className="flex-col items-start gap-2 pb-3">
        <div>
          <CardTitle className="whitespace-nowrap">
            <TrendingUp size={16} strokeWidth={1.75} className="text-brand shrink-0" />
            <span>CGM and forecast</span>
          </CardTitle>
          <p className="text-xs text-text-muted mt-0.5">
            Continuous glucose telemetry with calibrated quantile fan (q10, q50, q90)
          </p>
        </div>

        <div className="flex items-center gap-2 flex-wrap pt-0.5">
          <Badge variant={isPersonalized ? "purple" : "neutral"} className="text-[10px] whitespace-nowrap">
            {isPersonalized ? "Personalized model active" : "Population model served (no adapter yet)"}
          </Badge>
          {hasUsableAlert ? (
            <Badge variant="danger" className="flex items-center gap-1 whitespace-nowrap">
              <AlertTriangle size={14} strokeWidth={1.75} />
              <span>Calibrated Alert Triggered</span>
            </Badge>
          ) : (
            <Badge variant="success" className="whitespace-nowrap">Trajectory Monitored</Badge>
          )}
        </div>
      </CardHeader>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={chartData}
            margin={{ top: 16, right: 85, left: 15, bottom: 4 }}
          >
            <CartesianGrid strokeDasharray="2 2" stroke="#E3E8F0" vertical={false} />

            {/* XAxis with real time scale: -360m to +120m (75% history, 25% forecast) */}
            <XAxis
              type="number"
              dataKey="minuteOffset"
              domain={[-360, 120]}
              ticks={[-360, -240, -120, 0, 30, 60, 120]}
              tickFormatter={(v) => {
                if (v < 0) return `${v / 60}h`;
                if (v === 0) return "Now";
                return `+${v}m`;
              }}
              stroke="#64748B"
              tick={{ fontSize: 11 }}
            />

            {/* YAxis with ticks aligned to reference lines, unit shown once as title */}
            <YAxis
              domain={[40, 300]}
              ticks={[40, 70, 180, 250, 300]}
              stroke="#64748B"
              tick={{ fontSize: 11 }}
              label={{
                value: "mg/dL",
                angle: -90,
                position: "insideLeft",
                offset: -2,
                fill: "#64748B",
                fontSize: 10,
                fontWeight: 600,
              }}
            />

            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload as ChartPoint;
                  return (
                    <div className="bg-surface border border-border-strong rounded-lg p-2.5 text-xs space-y-1">
                      <div className="font-semibold text-text-primary">
                        {data.minuteOffset <= 0
                          ? `${data.minuteOffset === 0 ? "Now" : `${data.minuteOffset}m`} (${data.clockTime})`
                          : `+${data.minuteOffset}m Forecast`}
                      </div>
                      {data.actualGlucose != null && (
                        <div className="text-text-primary">
                          Observed: <strong>{data.actualGlucose.toFixed(1)} mg/dL</strong>
                        </div>
                      )}
                      {data.q50 != null && (
                        <div className="text-brand">
                          Forecast Median: <strong>{data.q50.toFixed(1)} mg/dL</strong>
                        </div>
                      )}
                      {data.fanRange && (
                        <div className="text-text-secondary text-[11px]">
                          Fan (q10 - q90): {data.fanRange[0].toFixed(0)} - {data.fanRange[1].toFixed(0)} mg/dL
                        </div>
                      )}
                    </div>
                  );
                }
                return null;
              }}
            />

            {/* Target glycemic range zone: 70 to 180 mg/dL */}
            <ReferenceArea y1={70} y2={180} fill="#10B981" fillOpacity={0.08} />

            {/* Clinical reference lines with labels in right margin outside plotted area */}
            <ReferenceLine
              y={250}
              stroke="#DC2626"
              strokeDasharray="3 3"
              strokeOpacity={0.7}
              label={{ value: "VH (>250)", fill: "#DC2626", fontSize: 10, position: "right", offset: 8 }}
            />
            <ReferenceLine
              y={180}
              stroke="#F59E0B"
              strokeDasharray="3 3"
              strokeOpacity={0.7}
              label={{ value: "Hyper (>180)", fill: "#B45309", fontSize: 10, position: "right", offset: 8 }}
            />
            <ReferenceLine
              y={70}
              stroke="#7C3AED"
              strokeDasharray="3 3"
              strokeOpacity={0.7}
              label={{ value: "Hypo (<70)", fill: "#7C3AED", fontSize: 10, position: "right", offset: 8 }}
            />
            <ReferenceLine
              y={54}
              stroke="#DC2626"
              strokeDasharray="3 3"
              strokeOpacity={0.7}
              label={{ value: "VL (<54)", fill: "#DC2626", fontSize: 10, position: "right", offset: 8 }}
            />

            {/* Vertical forecast horizon markers */}
            <ReferenceLine x={30} stroke="#94A3B8" strokeDasharray="2 2" strokeOpacity={0.6} />
            <ReferenceLine x={60} stroke="#94A3B8" strokeDasharray="2 2" strokeOpacity={0.6} />
            <ReferenceLine x={120} stroke="#94A3B8" strokeDasharray="2 2" strokeOpacity={0.6} />

            {/* Threshold crossing marker when alert is active */}
            {hasUsableAlert && hyperCrossingMin !== null && (
              <ReferenceLine
                x={hyperCrossingMin}
                stroke="#EF4444"
                strokeDasharray="3 3"
                strokeWidth={1.5}
                label={{
                  value: `expected in about ${hyperCrossingMin} min`,
                  position: "top",
                  fill: "#DC2626",
                  fontSize: 10,
                  fontWeight: 600,
                }}
              />
            )}

            {/* Quantile Fan: filled band between q10 and q90 */}
            <Area
              type="monotone"
              dataKey="fanRange"
              stroke="transparent"
              fill="#0284C7"
              fillOpacity={0.16}
              isAnimationActive={false}
              name="Quantile Fan (q10 - q90)"
            />

            {/* Observed CGM line: 2px dark line (#0F172A) with dot ONLY on the latest point */}
            <Line
              type="monotone"
              dataKey="actualGlucose"
              stroke="#0F172A"
              strokeWidth={2}
              isAnimationActive={false}
              dot={(props: any) => {
                if (props.payload.minuteOffset === 0) {
                  return (
                    <circle
                      cx={props.cx}
                      cy={props.cy}
                      r={4}
                      fill="#0F172A"
                      stroke="#FFFFFF"
                      strokeWidth={2}
                      key="latest-cgm-marker"
                    />
                  );
                }
                return <React.Fragment key={`dot-${props.payload.minuteOffset}`} />;
              }}
              name="Observed CGM"
            />

            {/* Forecast Median Line: brand blue (#0284C7) with labeled markers at +30, +60, +120 */}
            <Line
              type="monotone"
              dataKey="q50"
              stroke="#0284C7"
              strokeWidth={2}
              strokeDasharray="4 4"
              isAnimationActive={false}
              dot={(props: any) => {
                if ([30, 60, 120].includes(props.payload.minuteOffset)) {
                  const is120 = props.payload.minuteOffset === 120;
                  return (
                    <g key={`marker-f-${props.payload.minuteOffset}`}>
                      <circle
                        cx={props.cx}
                        cy={props.cy}
                        r={3.5}
                        fill="#0284C7"
                        stroke="#FFFFFF"
                        strokeWidth={1.5}
                      />
                      <text
                        x={is120 ? props.cx - 8 : props.cx}
                        y={props.cy - 7}
                        textAnchor={is120 ? "end" : "middle"}
                        fill="#0284C7"
                        fontSize={10}
                        fontWeight={600}
                      >
                        {Math.round(props.payload.q50)}
                      </text>
                    </g>
                  );
                }
                return <React.Fragment key={`fdot-${props.payload.minuteOffset}`} />;
              }}
              name="Forecast Median (q50)"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Legend with distinct colors */}
      <div className="mt-4 pt-3 border-t border-border flex flex-wrap items-center justify-between gap-3 text-xs text-text-secondary">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-3 bg-[#0F172A] rounded-sm" />
            <span>Observed CGM</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-3 bg-[#0284C7] rounded-sm border border-dashed border-[#0284C7]" />
            <span>Forecast Median (q50)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-3 bg-[#BAE6FD] border border-[#0284C7]/40 rounded-sm" />
            <span>Quantile Fan (q10 - q90)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-3 bg-[#10B981]/20 border border-[#10B981]/40 rounded-sm" />
            <span>Target Range (70-180 mg/dL)</span>
          </div>
        </div>

        {forecast && (
          <div className="text-text-muted tnum">
            {forecast.summary_message}
          </div>
        )}
      </div>
    </Card>
  );
};

