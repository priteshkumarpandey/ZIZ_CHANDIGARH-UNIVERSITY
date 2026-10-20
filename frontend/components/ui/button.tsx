import React from "react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "outline" | "danger";
  size?: "sm" | "md" | "lg";
}

export const Button: React.FC<ButtonProps> = ({
  children,
  className,
  variant = "primary",
  size = "md",
  ...props
}) => {
  const baseStyles = "inline-flex items-center justify-center font-medium rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-inset disabled:opacity-50 disabled:cursor-not-allowed";
  
  const variants = {
    primary: "bg-brand hover:bg-brand-line text-white focus:ring-brand-line border border-brand",
    secondary: "bg-surface-muted hover:bg-border text-text-primary border border-border focus:ring-brand-line",
    outline: "bg-surface border border-border hover:bg-surface-muted text-text-primary focus:ring-brand-line",
    danger: "bg-severe hover:bg-severe-subtle hover:text-severe text-white border border-severe focus:ring-severe",
  };

  const sizes = {
    sm: "px-2.5 py-1 text-xs",
    md: "px-3.5 py-1.5 text-sm",
    lg: "px-5 py-2.5 text-base",
  };

  return (
    <button
      className={twMerge(clsx(baseStyles, variants[variant], sizes[size], className))}
      {...props}
    >
      {children}
    </button>
  );
};

