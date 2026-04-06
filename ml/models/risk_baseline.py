from __future__ import annotations


def combine_scores(texture_score: float, structure_score: float, tonal_score: float, palette_score: float) -> float:
    return round(
        (0.35 * texture_score)
        + (0.25 * structure_score)
        + (0.20 * tonal_score)
        + (0.20 * palette_score),
        1,
    )

