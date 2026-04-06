import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatArtistName(kebab: string): string {
  return kebab
    .split("-")
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(" ");
}

export function formatDistance(value: number): string {
  return value.toFixed(2);
}

export function formatSimilarityPercent(score: number): string {
  const pct = Math.min(100, Math.max(0, score * 100));
  return `${pct.toFixed(1)}%`;
}
