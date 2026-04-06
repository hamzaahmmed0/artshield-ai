from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms

from config import (
    FINETUNED_ARTIST_INDEX_PATH,
    FINETUNED_CHECKPOINT_PATH,
    FINETUNED_PROFILES_PATH,
    FINETUNED_TRAIN_EMBEDDINGS_PATH,
    FINETUNED_TRAIN_METADATA_PATH,
    FINETUNED_VAL_EMBEDDINGS_PATH,
    FINETUNED_VAL_LABELS_PATH,
    MODEL_VERSION,
    SUPPORTED_ARTISTS,
)


LOGGER = logging.getLogger("artshield.ml")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _cosine_similarity(query: np.ndarray, references: np.ndarray) -> np.ndarray:
    query_norm = query / max(np.linalg.norm(query), 1e-12)
    reference_norms = references / np.maximum(np.linalg.norm(references, axis=1, keepdims=True), 1e-12)
    return reference_norms @ query_norm


def _mahalanobis_distance(embedding: np.ndarray, centroid: np.ndarray, precision: np.ndarray) -> float:
    delta = embedding - centroid
    return float(np.sqrt(np.maximum(delta @ precision @ delta.T, 0.0)))


class FineTunedDinoClassifier(torch.nn.Module):
    def __init__(self, num_classes: int, image_size: int = IMAGE_SIZE) -> None:
        super().__init__()
        self.backbone_source = "timm"
        try:
            self.backbone = timm.create_model(
                "vit_base_patch14_dinov2.lvd142m",
                pretrained=True,
                img_size=image_size,
                num_classes=0,
            )
            self.embedding_dim = int(self.backbone.num_features)
        except Exception as exc:
            LOGGER.warning("timm pretrained load failed (%s). Falling back to cached torch.hub DINOv2 weights.", exc)
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            self.embedding_dim = int(self.backbone.embed_dim)
            self.backbone_source = "torchhub"
        self.head = torch.nn.Linear(self.embedding_dim, num_classes)

    def extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)
        if self.backbone_source == "timm":
            return self.backbone.forward_head(features, pre_logits=True)
        return features["x_norm_clstoken"]


def _build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _load_runtime() -> dict[str, Any]:
    checkpoint = torch.load(FINETUNED_CHECKPOINT_PATH, map_location="cpu")
    artist_to_index = {artist: int(index) for artist, index in checkpoint["artist_to_index"].items()}

    model = FineTunedDinoClassifier(num_classes=checkpoint["num_classes"], image_size=checkpoint["image_size"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.head = torch.nn.Identity()
    model.eval()
    model.to(DEVICE)

    profiles_payload = _load_pickle(FINETUNED_PROFILES_PATH)
    train_embeddings = np.load(FINETUNED_TRAIN_EMBEDDINGS_PATH).astype(np.float32)
    train_metadata = _load_json(FINETUNED_TRAIN_METADATA_PATH)
    val_embeddings = np.load(FINETUNED_VAL_EMBEDDINGS_PATH).astype(np.float32)
    val_labels = np.load(FINETUNED_VAL_LABELS_PATH).astype(np.int32)
    artist_index_artifact = {artist: int(index) for artist, index in _load_json(FINETUNED_ARTIST_INDEX_PATH).items()}

    inverse_artist_index = {index: artist for artist, index in artist_index_artifact.items()}

    thresholds: dict[str, dict[str, float]] = {}
    global_distances: list[float] = []
    for artist_name, label_id in artist_index_artifact.items():
        artist_mask = val_labels == int(label_id)
        artist_embeddings = val_embeddings[artist_mask]
        if len(artist_embeddings) == 0:
            continue
        profile = profiles_payload["profiles"][artist_name]
        distances = [
            _mahalanobis_distance(embedding, profile["centroid"], profile["precision"])
            for embedding in artist_embeddings
        ]
        global_distances.extend(distances)
        thresholds[artist_name] = {
            "low": round(float(np.quantile(distances, 0.85)), 4),
            "high": round(float(np.quantile(distances, 0.95)), 4),
        }

    fallback_thresholds = {
        "low": round(float(np.quantile(global_distances, 0.85)), 4),
        "high": round(float(np.quantile(global_distances, 0.95)), 4),
    }
    for artist_name in SUPPORTED_ARTISTS:
        thresholds.setdefault(artist_name, fallback_thresholds)

    LOGGER.info(
        "Loaded ML runtime on %s with %s artists and model version %s",
        DEVICE_NAME,
        len(artist_to_index),
        MODEL_VERSION,
    )
    return {
        "model": model,
        "artist_to_index": artist_to_index,
        "inverse_artist_index": inverse_artist_index,
        "profiles": profiles_payload["profiles"],
        "train_embeddings": train_embeddings,
        "train_metadata": train_metadata,
        "thresholds": thresholds,
        "transform": _build_transform(),
    }


def _risk_level(distance: float, thresholds: dict[str, float]) -> str:
    if distance <= thresholds["low"]:
        return "LOW"
    if distance <= thresholds["high"]:
        return "MEDIUM"
    return "HIGH"


def _confidence_note(distance: float, thresholds: dict[str, float], risk_level: str) -> str:
    if risk_level == "LOW":
        return (
            f"The artwork sits within the lower end of the claimed artist's validation distance band "
            f"({distance:.2f} vs LOW {thresholds['low']:.2f})."
        )
    if risk_level == "MEDIUM":
        return (
            f"The artwork falls between the claimed artist's low and high alert bands "
            f"({distance:.2f} vs {thresholds['low']:.2f}-{thresholds['high']:.2f})."
        )
    return (
        f"The artwork exceeds the claimed artist's high alert band "
        f"({distance:.2f} vs HIGH {thresholds['high']:.2f}), which raises style-consistency concern."
    )


def _recommendation(risk_level: str) -> str:
    if risk_level == "LOW":
        return "Use this as a positive screening signal, then verify provenance and expert context."
    if risk_level == "MEDIUM":
        return "Review provenance and compare against verified reference works before trusting the claim."
    return "Escalate this work to expert review before making any authenticity decision."


RUNTIME: dict[str, Any] | None = None
MODEL_LOAD_ERROR: str | None = None
MODEL_READY = False

try:
    RUNTIME = _load_runtime()
    MODEL_READY = True
except Exception as exc:
    MODEL_LOAD_ERROR = str(exc)
    LOGGER.exception("Failed to load ArtShield inference runtime")


def analyze_artwork(image: Image.Image, claimed_artist: str) -> dict[str, Any]:
    if not MODEL_READY or RUNTIME is None:
        raise RuntimeError(f"Model runtime is not available: {MODEL_LOAD_ERROR or 'unknown error'}")
    if claimed_artist not in SUPPORTED_ARTISTS:
        raise ValueError(f"Unsupported artist '{claimed_artist}'.")

    tensor = RUNTIME["transform"](image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            embedding_tensor = RUNTIME["model"].extract_embeddings(tensor)
    embedding = embedding_tensor.detach().float().cpu().numpy()[0].astype(np.float32)

    distances: list[tuple[str, float]] = []
    for artist_name, profile in RUNTIME["profiles"].items():
        distance = _mahalanobis_distance(embedding, profile["centroid"], profile["precision"])
        distances.append((artist_name, distance))
    distances.sort(key=lambda item: item[1])

    predicted_artist, _ = distances[0]
    target_distance = next(distance for artist_name, distance in distances if artist_name == claimed_artist)
    thresholds = RUNTIME["thresholds"][claimed_artist]
    level = _risk_level(target_distance, thresholds)

    similarities = _cosine_similarity(embedding, RUNTIME["train_embeddings"])
    best_indices = np.argsort(similarities)[::-1][:3]
    nearest_reference_works = [
        {
            "filename": Path(RUNTIME["train_metadata"][int(index)]["prepared_image_path"]).name,
            "similarity_score": round(float(similarities[int(index)]), 4),
        }
        for index in best_indices
    ]

    return {
        "claimed_artist": claimed_artist,
        "predicted_artist": predicted_artist,
        "risk_level": level,
        "target_distance": round(float(target_distance), 4),
        "confidence_note": _confidence_note(target_distance, thresholds, level),
        "top_matches": [
            {"artist": artist_name, "distance": round(float(distance), 4)}
            for artist_name, distance in distances[:3]
        ],
        "nearest_reference_works": nearest_reference_works,
        "per_artist_thresholds": {
            "low": thresholds["low"],
            "high": thresholds["high"],
        },
        "recommendation": _recommendation(level),
        "model_version": MODEL_VERSION,
        "disclaimer": "This output is a risk signal for expert review, not an authenticity verdict.",
    }

