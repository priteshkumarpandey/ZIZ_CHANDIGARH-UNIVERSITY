import React from "react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

interface SliderProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  valueDisplay?: string;
}

export const Slider: React.FC<SliderProps> = ({
  label,
  valueDisplay,
  className,
  ...props
}) => {
  return (
    <div className="space-y-1.5 w-full">
      {(label || valueDisplay) && (
        <div className="flex justify-between text-xs font-medium text-text-secondary">
          {label && <span>{label}</span>}
          {valueDisplay && <span className="text-brand font-semibold tnum">{valueDisplay}</span>}
        </div>
      )}
      <input
        type="range"
        className={twMerge(
          clsx(
            "w-full h-1.5 bg-border rounded-lg appearance-none cursor-pointer accent-brand",
            className
          )
        )}
        {...props}
      />
    </div>
  );
};

