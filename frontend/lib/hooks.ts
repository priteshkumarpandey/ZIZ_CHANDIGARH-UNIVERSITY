import { useEffect, useState, useRef } from "react";
import { CGMReading } from "./types";

export function useLiveTelemetry(patientId: string) {
  const [reading, setReading] = useState<CGMReading | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!patientId) return;

    const wsUrl = (process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/api/v1/stream") + `/${patientId}`;
    
    try {
      const socket = new WebSocket(wsUrl);
      socketRef.current = socket;

      socket.onopen = () => setIsConnected(true);
      socket.onclose = () => setIsConnected(false);
      socket.onerror = () => setIsConnected(false);

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setReading(data);
        } catch (err) {
          console.error("Telemetry message parse error", err);
        }
      };
    } catch (err) {
      console.warn("WebSocket connection fallback:", err);
      setIsConnected(false);
    }

    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [patientId]);

  return { reading, isConnected };
}
