from pydantic import BaseModel, Field


class ArtworkMetadata(BaseModel):
    title: str
    claimed_artist: str
    year: str | None = None
    medium: str | None = None
    notes: str | None = None
    filename: str
    content_type: str


class SignalScore(BaseModel):
    label: str
    score: float = Field(ge=0, le=100)
    summary: str


class SuspiciousRegion(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    severity: float = Field(ge=0, le=100)


class AnalysisResponse(BaseModel):
    title: str
    claimed_artist: str
    verdict: str
    risk_score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=100)
    overview: str
    signals: list[SignalScore]
    suspicious_regions: list[SuspiciousRegion]
    next_step: str

