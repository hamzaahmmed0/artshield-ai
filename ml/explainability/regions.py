from __future__ import annotations

import numpy as np


def top_variance_regions(gray_image: np.ndarray, grid_size: int = 4) -> list[dict[str, float]]:
    height, width = gray_image.shape
    patch_h = max(1, height // grid_size)
    patch_w = max(1, width // grid_size)
    regions: list[dict[str, float]] = []

    for row in range(grid_size):
        for col in range(grid_size):
            y0 = row * patch_h
            x0 = col * patch_w
            patch = gray_image[y0 : min(height, y0 + patch_h), x0 : min(width, x0 + patch_w)]
            regions.append(
                {
                    "x": float(x0),
                    "y": float(y0),
                    "width": float(patch.shape[1]),
                    "height": float(patch.shape[0]),
                    "variance": float(np.var(patch)),
                }
            )

    return sorted(regions, key=lambda item: item["variance"], reverse=True)[:4]

