"use client";

import React, { useEffect, useState } from "react";
import { CheckCircle2, XCircle, Loader2 } from "lucide-react";
import { checkApiHealth } from "../lib/api";
import { HealthStatus } from "../lib/types";

export const ApiStatusBadge: React.FC = () => {
  const [status, setStatus] = useState<HealthStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    checkApiHealth()
      .then((data) => {
        setStatus(data);
        setError(null);
      })
      .catch((err) => {
        setError(err.message);
        setStatus(null);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center gap-1.5 text-xs text-text-muted bg-surface-muted border border-border px-2.5 py-1 rounded-full">
        <Loader2 size={16} strokeWidth={1.75} className="animate-spin text-brand" />
        <span>Checking API...</span>
      </div>
    );
  }

  if (error || !status) {
    return (
      <div className="flex items-center gap-1.5 text-xs text-severe bg-red-subtle border border-red/30 px-2.5 py-1 rounded-full">
        <XCircle size={16} strokeWidth={1.75} className="text-severe" />
        <span>API Offline</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-1.5 text-xs text-in-range-text bg-green-subtle border border-green/30 px-2.5 py-1 rounded-full">
      <CheckCircle2 size={16} strokeWidth={1.75} className="text-in-range" />
      <span>API Healthy ({status.environment})</span>
    </div>
  );
};
