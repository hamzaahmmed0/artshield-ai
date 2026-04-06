from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.schemas.analysis import (
    AnalysisResponse,
    ArtworkMetadata,
    SignalScore,
    SuspiciousRegion,
)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return max(minimum, min(maximum, value))


def _build_suspicious_regions(gray_array: np.ndarray) -> list[SuspiciousRegion]:
    height, width = gray_array.shape
    grid_rows = 4
    grid_cols = 4
    patch_h = max(1, height // grid_rows)
    patch_w = max(1, width // grid_cols)

    regions: list[SuspiciousRegion] = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            y0 = row * patch_h
            x0 = col * patch_w
            patch = gray_array[y0 : min(height, y0 + patch_h), x0 : min(width, x0 + patch_w)]
            patch_std = float(np.std(patch))
            severity = _clamp((patch_std / 64.0) * 100.0)
            if severity >= 45:
                regions.append(
                    SuspiciousRegion(
                        x=x0,
                        y=y0,
                        width=patch.shape[1],
                        height=patch.shape[0],
                        severity=round(severity, 1),
                    )
                )

    return sorted(regions, key=lambda region: region.severity, reverse=True)[:4]


def analyze_artwork(metadata: ArtworkMetadata, image_bytes: bytes) -> AnalysisResponse:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file could not be parsed as an image.") from exc

    resized = image.resize((512, 512))
    array = np.asarray(resized, dtype=np.float32)
    gray = np.asarray(resized.convert("L"), dtype=np.float32)

    brightness = float(np.mean(gray) / 255.0)
    contrast = float(np.std(gray) / 64.0)

    grad_y, grad_x = np.gradient(gray)
    gradient_magnitude = np.sqrt((grad_x ** 2) + (grad_y ** 2))
    edge_density = float(np.mean(gradient_magnitude > 18.0))

    channel_std = np.std(array, axis=(0, 1))
    color_balance = float(np.mean(channel_std) / 64.0)

    texture_score = _clamp(contrast * 100.0)
    structure_score = _clamp((1.0 - abs(edge_density - 0.22) / 0.22) * 100.0)
    tonal_score = _clamp((1.0 - abs(brightness - 0.52) / 0.52) * 100.0)
    palette_score = _clamp(color_balance * 100.0)

    consistency_score = round(
        (0.35 * texture_score)
        + (0.25 * structure_score)
        + (0.20 * tonal_score)
        + (0.20 * palette_score),
        1,
    )
    risk_score = round(_clamp(100.0 - consistency_score), 1)
    confidence = round(_clamp(55.0 + abs(consistency_score - 50.0) * 0.8), 1)

    if risk_score < 35:
        verdict = "Low Risk"
        next_step = "Use this result as a screening signal, then compare against provenance and expert review."
    elif risk_score < 65:
        verdict = "Suspicious"
        next_step = "Review the highlighted regions and compare this piece against verified reference works."
    else:
        verdict = "Needs Expert Review"
        next_step = "Do not rely on this upload alone. Gather provenance and escalate to a professional reviewer."

    suspicious_regions = _build_suspicious_regions(gray)

    overview = (
        "This baseline report uses image texture, tonal balance, surface structure, and patch-level variance "
        "to estimate visual consistency. It is a starting point for product development, not a final authenticity decision."
    )

    signals = [
        SignalScore(
            label="Texture Consistency",
            score=round(texture_score, 1),
            summary="Measures local grayscale variation as a lightweight brushstroke and surface proxy.",
        ),
        SignalScore(
            label="Structural Balance",
            score=round(structure_score, 1),
            summary="Looks at edge density to estimate whether the image has an unusually flat or noisy structure.",
        ),
        SignalScore(
            label="Tonal Stability",
            score=round(tonal_score, 1),
            summary="Checks whether the overall brightness profile looks unusually skewed.",
        ),
        SignalScore(
            label="Palette Spread",
            score=round(palette_score, 1),
            summary="Estimates channel variation as a quick proxy for color complexity and consistency.",
        ),
    ]

    return AnalysisResponse(
        title=metadata.title,
        claimed_artist=metadata.claimed_artist,
        verdict=verdict,
        risk_score=risk_score,
        confidence=confidence,
        overview=overview,
        signals=signals,
        suspicious_regions=suspicious_regions,
        next_step=next_step,
    )

