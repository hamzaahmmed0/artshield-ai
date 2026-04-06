"use client";

import type { RiskLevel } from "@/types/analysis";
import { cn, formatDistance } from "@/lib/utils";

type Props = {
  targetDistance: number;
  low: number;
  high: number;
  riskLevel: RiskLevel;
};

function barColor(risk: RiskLevel): string {
  if (risk === "LOW") return "bg-risk-low";
  if (risk === "MEDIUM") return "bg-risk-medium";
  return "bg-risk-high";
}

export function DistanceMeter({ targetDistance, low, high, riskLevel }: Props) {
  const upper = Math.max(high * 1.2, targetDistance * 1.05, low * 1.5, 1);
  const pct = Math.min(100, Math.max(0, (targetDistance / upper) * 100));
  const lowPct = Math.min(100, (low / upper) * 100);
  const highPct = Math.min(100, (high / upper) * 100);

  return (
    <div className="space-y-3">
      <div className="flex items-end justify-between font-mono text-xs uppercase tracking-wider text-neutral-500">
        <span>Embedding distance</span>
        <span className="text-neutral-300">{formatDistance(targetDistance)}</span>
      </div>
      <div className="relative h-3 w-full border border-lab-border bg-lab-bg">
        <div className="pointer-events-none absolute left-0 top-0 h-full w-px bg-neutral-600/80" style={{ left: `${lowPct}%` }} />
        <div className="pointer-events-none absolute left-0 top-0 h-full w-px bg-neutral-600/80" style={{ left: `${highPct}%` }} />
        <div
          className={cn("h-full transition-all duration-500", barColor(riskLevel))}
          style={{ width: `${pct}%` }}
        />
        <div
          className="absolute top-1/2 h-4 w-0.5 -translate-y-1/2 bg-white shadow-[0_0_8px_rgba(255,255,255,0.6)]"
          style={{ left: `${pct}%`, marginLeft: "-1px" }}
        />
      </div>
      <div className="flex flex-wrap justify-between gap-2 font-mono text-[10px] text-neutral-500">
        <span>
          Low threshold <span className="text-neutral-400">{formatDistance(low)}</span>
        </span>
        <span>
          High threshold <span className="text-neutral-400">{formatDistance(high)}</span>
        </span>
      </div>
      <p className="font-mono text-[11px] leading-relaxed text-neutral-500">
        Bar fill shows distance to the claimed artist profile relative to calibrated low / high thresholds for that
        artist.
      </p>
    </div>
  );
}
