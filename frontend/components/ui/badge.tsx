import React from "react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "success" | "warning" | "danger" | "info" | "neutral" | "purple";
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  className,
  variant = "neutral",
  ...props
}) => {
  const variants = {
    success: "bg-in-range-subtle text-in-range-text border-in-range/30",
    warning: "bg-high-subtle text-high-text border-high/30",
    danger: "bg-severe-subtle text-severe border-severe/30",
    info: "bg-brand-subtle text-brand border-brand/30",
    neutral: "bg-surface-muted text-text-secondary border-border",
    purple: "bg-low-subtle text-low border-low/30",
  };

  return (
    <span
      className={twMerge(
        clsx(
          "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium tracking-tight",
          variants[variant],
          className
        )
      )}
      {...props}
    >
      {children}
    </span>
  );
};

