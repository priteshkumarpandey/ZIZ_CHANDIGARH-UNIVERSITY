import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        surface: {
          DEFAULT: "var(--surface)",
          muted: "var(--surface-muted)",
        },
        border: {
          DEFAULT: "var(--border)",
          strong: "var(--border-strong)",
        },
        text: {
          primary: "var(--text-primary)",
          secondary: "var(--text-secondary)",
          muted: "var(--text-muted)",
        },
        brand: {
          DEFAULT: "var(--brand)",
          line: "var(--brand-line)",
          subtle: "var(--brand-subtle)",
        },
        high: {
          DEFAULT: "var(--high)",
          text: "var(--high-text)",
          subtle: "var(--high-subtle)",
        },
        severe: {
          DEFAULT: "var(--severe)",
          subtle: "var(--severe-subtle)",
        },
        "in-range": {
          DEFAULT: "var(--in-range)",
          text: "var(--in-range-text)",
          subtle: "var(--in-range-subtle)",
        },
        low: {
          DEFAULT: "var(--low)",
          subtle: "var(--low-subtle)",
        },
        navy: {
          DEFAULT: "var(--navy)",
          light: "var(--navy-light)",
          subtle: "var(--navy-subtle)",
        },
        red: {
          DEFAULT: "var(--red)",
          subtle: "var(--red-subtle)",
        },
        amber: {
          DEFAULT: "var(--amber)",
          subtle: "var(--amber-subtle)",
        },
        green: {
          DEFAULT: "var(--green)",
          subtle: "var(--green-subtle)",
        },
        purple: {
          DEFAULT: "var(--purple)",
          subtle: "var(--purple-subtle)",
        },
      },
      fontFamily: {
        sans: ["Inter Variable", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
      },
      borderRadius: {
        card: "12px",
      },
    },
  },
  plugins: [],
};

export default config;

