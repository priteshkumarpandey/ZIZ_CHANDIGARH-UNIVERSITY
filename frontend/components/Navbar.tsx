import React from "react";
import { ApiStatusBadge } from "./ApiStatusBadge";

export const Navbar: React.FC = () => {
  return (
    <header className="h-16 border-b border-border bg-surface/80 backdrop-blur px-6 flex items-center justify-between sticky top-0 z-30">
      <div className="flex items-center gap-3">
        <span className="text-xs font-semibold text-text-muted uppercase tracking-wider">
          Happiest Health Digital Twin Challenge 2026
        </span>
      </div>

      <div className="flex items-center gap-4">
        <ApiStatusBadge />
      </div>
    </header>
  );
};
