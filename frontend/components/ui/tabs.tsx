import React from "react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

interface TabsProps {
  tabs: { id: string; label: string }[];
  activeTab: string;
  onTabChange: (id: string) => void;
  className?: string;
}

export const Tabs: React.FC<TabsProps> = ({
  tabs,
  activeTab,
  onTabChange,
  className,
}) => {
  return (
    <div className={twMerge(clsx("flex space-x-1 rounded-lg bg-surface-muted p-1 border border-border", className))}>
      {tabs.map((tab) => {
        const isActive = activeTab === tab.id;
        return (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={clsx(
              "flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
              isActive
                ? "bg-surface text-text-primary border border-border"
                : "text-text-secondary hover:text-text-primary hover:bg-surface/50"
            )}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
};

