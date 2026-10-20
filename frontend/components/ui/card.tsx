import React from "react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export const Card: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  children,
  className,
  ...props
}) => {
  return (
    <div
      className={twMerge(
        clsx(
          "bg-surface border border-border rounded-card p-5 text-text-primary transition-colors",
          className
        )
      )}
      {...props}
    >
      {children}
    </div>
  );
};

export const CardHeader: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  children,
  className,
  ...props
}) => {
  return (
    <div className={twMerge(clsx("mb-4 flex items-center justify-between border-b border-border pb-3", className))} {...props}>
      {children}
    </div>
  );
};

export const CardTitle: React.FC<React.HTMLAttributes<HTMLHeadingElement>> = ({
  children,
  className,
  ...props
}) => {
  return (
    <h3 className={twMerge(clsx("text-sm font-semibold text-text-primary flex items-center gap-2 tracking-tight", className))} {...props}>
      {children}
    </h3>
  );
};

