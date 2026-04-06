"use client";

import { motion } from "framer-motion";
import type { RiskLevel } from "@/types/analysis";
import { cn } from "@/lib/utils";

const styles: Record<RiskLevel, { text: string; border: string; pulse: string[] }> = {
  LOW: {
    text: "text-risk-low",
    border: "border-risk-low/60",
    pulse: [
      "0 0 16px rgba(34, 197, 94, 0.25)",
      "0 0 40px rgba(34, 197, 94, 0.5)",
      "0 0 16px rgba(34, 197, 94, 0.25)",
    ],
  },
  MEDIUM: {
    text: "text-risk-medium",
    border: "border-risk-medium/60",
    pulse: [
      "0 0 16px rgba(245, 158, 11, 0.25)",
      "0 0 40px rgba(245, 158, 11, 0.55)",
      "0 0 16px rgba(245, 158, 11, 0.25)",
    ],
  },
  HIGH: {
    text: "text-risk-high",
    border: "border-risk-high/60",
    pulse: [
      "0 0 16px rgba(239, 68, 68, 0.25)",
      "0 0 40px rgba(239, 68, 68, 0.55)",
      "0 0 16px rgba(239, 68, 68, 0.25)",
    ],
  },
};

export function RiskBadge({ level }: { level: RiskLevel }) {
  const s = styles[level];

  return (
    <div className="flex justify-center">
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <motion.div
          className={cn(
            "border-2 bg-lab-surface px-10 py-4 font-mono text-2xl font-bold tracking-[0.35em] md:text-3xl",
            s.text,
            s.border,
          )}
          animate={{ boxShadow: s.pulse }}
          transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
        >
          {level}
        </motion.div>
      </motion.div>
    </div>
  );
}
