export type RiskLevel = "LOW" | "MEDIUM" | "HIGH";

export type ArtistMatch = {
  artist: string;
  distance: number;
};

export type NearestReferenceWork = {
  filename: string;
  similarity_score: number;
};

export type AnalyzeResponse = {
  claimed_artist: string;
  predicted_artist: string;
  risk_level: RiskLevel;
  target_distance: number;
  confidence_note: string;
  top_matches: ArtistMatch[];
  nearest_reference_works: NearestReferenceWork[];
  per_artist_thresholds: { low: number; high: number };
  recommendation: string;
  model_version: string;
  disclaimer: string;
};

export type ArtistsResponse = {
  supported_artists: string[];
  total: number;
};

export type HealthResponse = {
  status: string;
  model_loaded: boolean;
  model_version: string;
  artists_loaded: number;
  device: string;
};
