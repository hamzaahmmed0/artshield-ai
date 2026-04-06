import type { AnalyzeResponse, ArtistsResponse, HealthResponse } from "@/types/analysis";

const DEFAULT_BASE = "http://localhost:8000";

export function getApiBaseUrl(): string {
  if (typeof process !== "undefined" && process.env.NEXT_PUBLIC_API_BASE_URL) {
    return process.env.NEXT_PUBLIC_API_BASE_URL.replace(/\/$/, "");
  }
  return DEFAULT_BASE;
}

function offlineMessage(): string {
  return "Backend offline. Please start the ArtShield API server.";
}

export async function fetchHealth(): Promise<HealthResponse> {
  const base = getApiBaseUrl();
  let response: Response;
  try {
    response = await fetch(`${base}/api/health`, { cache: "no-store" });
  } catch {
    throw new Error(offlineMessage());
  }
  if (!response.ok) {
    throw new Error(offlineMessage());
  }
  return response.json() as Promise<HealthResponse>;
}

export async function fetchArtists(): Promise<ArtistsResponse> {
  const base = getApiBaseUrl();
  let response: Response;
  try {
    response = await fetch(`${base}/api/artists`, { cache: "no-store" });
  } catch {
    throw new Error(offlineMessage());
  }
  if (!response.ok) {
    throw new Error(offlineMessage());
  }
  return response.json() as Promise<ArtistsResponse>;
}

export async function analyzeArtwork(image: File, claimedArtist: string): Promise<AnalyzeResponse> {
  const base = getApiBaseUrl();
  const body = new FormData();
  body.append("image", image);
  body.append("claimed_artist", claimedArtist);

  let response: Response;
  try {
    response = await fetch(`${base}/api/analyze`, {
      method: "POST",
      body,
    });
  } catch {
    throw new Error(offlineMessage());
  }

  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    if (response.status >= 500 || response.status === 0) {
      throw new Error(offlineMessage());
    }
    throw new Error(payload?.detail ?? "Analysis request failed.");
  }

  return response.json() as Promise<AnalyzeResponse>;
}

/** Reference thumbnails are not served by the default API; URL can be overridden via env. */
export function referenceImageUrl(filename: string): string {
  const base = getApiBaseUrl();
  const custom =
    typeof process !== "undefined" ? process.env.NEXT_PUBLIC_REFERENCE_IMAGE_BASE_URL : undefined;
  if (custom) {
    return `${custom.replace(/\/$/, "")}/${encodeURIComponent(filename)}`;
  }
  return `${base}/reference-assets/${encodeURIComponent(filename)}`;
}
