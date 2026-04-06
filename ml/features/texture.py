from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def extract_lbp(image: np.ndarray, radius: int = 3, n_points: int = 24) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(grayscale, (224, 224))
    lbp = local_binary_pattern(resized, n_points, radius, method="uniform")
    histogram, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
    return histogram.astype(np.float32)


def extract_gabor(image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(grayscale, (224, 224))
    features: list[float] = []
    for theta in [0, 45, 90, 135]:
        kernel = cv2.getGaborKernel((21, 21), 5, np.deg2rad(theta), 10, 0.5)
        filtered = cv2.filter2D(resized, cv2.CV_64F, kernel)
        features.extend([float(filtered.mean()), float(filtered.std())])
    return np.asarray(features, dtype=np.float32)


def get_full_feature_vector(image: np.ndarray) -> np.ndarray:
    lbp = extract_lbp(image)
    gabor = extract_gabor(image)
    return np.concatenate([lbp, gabor], axis=0)

