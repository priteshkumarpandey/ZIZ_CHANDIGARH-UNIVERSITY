import React, { useEffect, useState } from "react";
import { Radio, AlertCircle, TrendingUp, TrendingDown, Minus } from "lucide-react";
import { useGlucoTwinStore } from "../lib/store";
import { RuleBasedAlert } from "../lib/types";
import { Card, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";

interface LiveStreamFeedProps {
  patientId: string;
}

export const LiveStreamFeed: React.FC<LiveStreamFeedProps> = ({ patientId }) => {
  const { appendCgmReading } = useGlucoTwinStore();
  const [latestReading, setLatestReading] = useState<{
    glucose_mgdl: number;
    trend_arrow: string;
    forecast_30m: number;
    adverse_event_warning: boolean;
    summary_message: string;
    rule_based_alerts?: RuleBasedAlert[];
  } | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);

  useEffect(() => {
    const wsUrl = (process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/api/v1/stream") + `/${patientId}`;
    let socket: WebSocket | null = null;

    try {
      socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        setIsConnected(true);
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLatestReading({
            glucose_mgdl: data.glucose_mgdl,
            trend_arrow: data.trend_arrow,
            forecast_30m: data.forecast_30m,
            adverse_event_warning: data.adverse_event_warning,
            summary_message: data.summary_message,
            rule_based_alerts: data.rule_based_alerts || [],
          });

          appendCgmReading({
            patient_id: patientId,
            timestamp: data.timestamp,
            glucose_mgdl: data.glucose_mgdl,
            trend_arrow: data.trend_arrow,
            steps: 0,
            mets: 1.0,
          });

        } catch (err) {
          console.error("Failed to parse websocket message", err);
        }
      };

      socket.onerror = () => {
        setIsConnected(false);
      };

      socket.onclose = () => {
        setIsConnected(false);
      };
    } catch (err) {
      console.error("WebSocket connection failed", err);
    }

    return () => {
      if (socket) {
        socket.close();
      }
    };
  }, [patientId, appendCgmReading]);

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Radio size={16} strokeWidth={1.75} className={isConnected ? "text-in-range animate-pulse" : "text-text-muted"} />
          <CardTitle>Real-Time Wearable Stream Feed</CardTitle>
        </div>

        <Badge variant={isConnected ? "success" : "neutral"}>
          {isConnected ? "Live Connected" : "Connecting"}
        </Badge>
      </CardHeader>

      {latestReading ? (
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 bg-surface-muted p-4 rounded-lg border border-border">
          <div className="flex items-center gap-4">
            <div>
              <div className="text-3xl font-extrabold text-text-primary flex items-center gap-2 tnum">
                {latestReading.glucose_mgdl}
                <span className="text-sm font-normal text-text-muted">mg/dL</span>
                {latestReading.trend_arrow === "Rising" && <TrendingUp size={20} strokeWidth={1.75} className="text-severe" />}
                {latestReading.trend_arrow === "Falling" && <TrendingDown size={20} strokeWidth={1.75} className="text-low" />}
                {latestReading.trend_arrow === "Flat" && <Minus size={20} strokeWidth={1.75} className="text-in-range" />}
              </div>
              <span className="text-xs text-text-muted">Latest Live CGM Reading</span>
            </div>
          </div>

          <div className="flex items-center gap-6 border-t sm:border-t-0 sm:border-l border-border pt-3 sm:pt-0 sm:pl-6 text-xs">
            <div>
              <span className="text-text-muted">Projected (+30m):</span>
              <div className="text-base font-bold text-high tnum">
                {latestReading.forecast_30m} mg/dL
              </div>
            </div>

            {latestReading.rule_based_alerts && latestReading.rule_based_alerts.length > 0 ? (
              <div className="flex flex-col gap-1">
                {latestReading.rule_based_alerts.map((alert, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-1.5 text-xs text-severe font-semibold bg-red-subtle px-2.5 py-1.5 rounded-lg border border-red/30"
                  >
                    <AlertCircle size={15} strokeWidth={1.75} className="shrink-0" />
                    <span>{alert.title}</span>
                    <Badge variant="danger" className="text-[10px] ml-1">
                      Rule-based alert (observed, not predicted)
                    </Badge>
                  </div>
                ))}
              </div>
            ) : latestReading.adverse_event_warning ? (
              <div className="flex items-center gap-1.5 text-severe font-semibold bg-red-subtle px-2.5 py-1.5 rounded-lg border border-red/30">
                <AlertCircle size={16} strokeWidth={1.75} />
                <span>Excursion Alert</span>
              </div>
            ) : null}
          </div>
        </div>
      ) : (
        <div className="p-4 text-center text-xs text-gray-500">
          Awaiting live telemetry packets...
        </div>
      )}
    </Card>
  );
};
