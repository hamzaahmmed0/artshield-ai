"use client";

import { formatArtistName, formatDistance } from "@/lib/utils";
import { cn } from "@/lib/utils";

type Props = {
  artist: string;
  distance: number;
  maxDistance: number;
  isClaimed: boolean;
};

export function ArtistMatchRow({ artist, distance, maxDistance, isClaimed }: Props) {
  const widthPct = maxDistance > 0 ? Math.min(100, (distance / maxDistance) * 100) : 0;

  return (
    <div
      className={cn(
        "flex flex-col gap-3 border border-transparent px-3 py-3 sm:grid sm:grid-cols-[minmax(0,1fr)_auto_minmax(100px,1fr)] sm:items-center sm:gap-4",
        isClaimed && "border-lab-accent/40 bg-lab-accent/[0.06]",
      )}
    >
      <div className="min-w-0">
        <span className="font-sans text-sm text-neutral-200">{formatArtistName(artist)}</span>
        {isClaimed ? (
          <span className="ml-2 font-mono text-[10px] uppercase tracking-widest text-lab-accent">claimed</span>
        ) : null}
      </div>
      <span className="font-mono text-sm tabular-nums text-neutral-400 sm:text-right">{formatDistance(distance)}</span>
      <div className="h-2 border border-lab-border bg-lab-bg sm:col-span-1">
        <div
          className={cn("h-full", isClaimed ? "bg-lab-accent/70" : "bg-neutral-600")}
          style={{ width: `${widthPct}%` }}
        />
      </div>
    </div>
  );
}
