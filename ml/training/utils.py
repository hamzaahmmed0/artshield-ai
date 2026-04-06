from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


ROOT_DIR = Path(__file__).resolve().parents[2]
ML_DIR = ROOT_DIR / "ml"
DATA_DIR = ML_DIR / "data" / "processed"
MODELS_DIR = ML_DIR / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
SPLITS_PATH = DATA_DIR / "splits.json"
PROFILES_PATH = MODELS_DIR / "artist_profiles.pkl"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_split_manifest(path: Path = SPLITS_PATH) -> dict[str, Any]:
    return load_json(path)


def build_artist_index(records: list[dict[str, Any]]) -> dict[str, int]:
    artists = sorted({record["artist"] for record in records})
    return {artist: index for index, artist in enumerate(artists)}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_dinov2_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class ArtworkDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], transform: transforms.Compose | None = None) -> None:
        self.records = records
        self.transform = transform or create_dinov2_transform()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        record = self.records[index]
        image_path = DATA_DIR / record["prepared_image_path"]
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, record


def collate_artworks(
    batch: list[tuple[torch.Tensor, dict[str, Any]]]
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    tensors, records = zip(*batch)
    return torch.stack(list(tensors), dim=0), list(records)


def mahalanobis_distance(embedding: np.ndarray, centroid: np.ndarray, precision: np.ndarray) -> float:
    delta = embedding - centroid
    return float(np.sqrt(np.maximum(delta @ precision @ delta.T, 0.0)))


def cosine_similarity_matrix(query: np.ndarray, references: np.ndarray) -> np.ndarray:
    query_norm = query / np.linalg.norm(query)
    reference_norms = references / np.linalg.norm(references, axis=1, keepdims=True)
    return reference_norms @ query_norm

