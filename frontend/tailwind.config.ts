import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        lab: {
          bg: "#0a0a0a",
          surface: "#111111",
          elevated: "#1a1a1a",
          border: "#2a2a2a",
          accent: "#f59e0b",
          muted: "#737373",
        },
        risk: {
          low: "#22c55e",
          medium: "#f59e0b",
          high: "#ef4444",
        },
      },
      fontFamily: {
        sans: ["var(--font-geist-sans)", "system-ui", "sans-serif"],
        serif: ["var(--font-instrument-serif)", "ui-serif", "Georgia", "serif"],
        mono: ["var(--font-geist-mono)", "ui-monospace", "monospace"],
      },
      boxShadow: {
        "risk-glow-low": "0 0 24px rgba(34, 197, 94, 0.45)",
        "risk-glow-medium": "0 0 24px rgba(245, 158, 11, 0.5)",
        "risk-glow-high": "0 0 24px rgba(239, 68, 68, 0.5)",
      },
      keyframes: {
        "scan-line": {
          "0%": { top: "0%" },
          "100%": { top: "100%" },
        },
      },
      animation: {
        "scan-line": "scan-line 1.8s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
