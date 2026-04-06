from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.training.utils import (
    ARTIFACTS_DIR,
    PROFILES_PATH,
    cosine_similarity_matrix,
    create_dinov2_transform,
    get_device,
    load_json,
    load_pickle,
    mahalanobis_distance,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image inference with the ArtShield AI baseline model.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the artwork image.")
    parser.add_argument("--claimed-artist", type=str, default=None, help="Claimed artist for risk scoring.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest reference works to return.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size for DINOv2.")
    parser.add_argument("--model-name", default="dinov2_vitb14")
    parser.add_argument("--hub-repo", default="facebookresearch/dinov2")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the full inference report as JSON.",
    )
    return parser.parse_args()


def resolve_artist_name(requested_artist: str, available_artists: list[str]) -> str:
    direct_lookup = {artist.lower(): artist for artist in available_artists}
    normalized_lookup = {
        artist.lower().replace("_", "-").replace(" ", "-"): artist for artist in available_artists
    }
    lowered = requested_artist.lower().strip()
    normalized = lowered.replace("_", "-").replace(" ", "-")

    if lowered in direct_lookup:
        return direct_lookup[lowered]
    if normalized in normalized_lookup:
        return normalized_lookup[normalized]

    raise ValueError(
        f"Unknown artist '{requested_artist}'. Available artists: {', '.join(sorted(available_artists))}"
    )


def risk_level(distance: float, low_threshold: float, high_threshold: float) -> str:
    if distance <= low_threshold:
        return "LOW"
    if distance <= high_threshold:
        return "MEDIUM"
    return "HIGH"


def next_step_for_risk(level: str) -> str:
    if level == "LOW":
        return "Use this as an encouraging screening signal, then verify provenance and expert context."
    if level == "MEDIUM":
        return "Review the nearest reference works and inspect metadata before trusting the claim."
    return "Treat this as a high-risk claim and escalate to expert review before making any decision."


def load_thresholds() -> tuple[float, float]:
    report_path = ARTIFACTS_DIR / "evaluation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            "Missing evaluation_report.json. Run fit_artist_profiles.py and evaluate_anomaly_model.py first."
        )
    report = load_json(report_path)
    thresholds = report["summary"]["validation_thresholds"]
    return float(thresholds["low"]), float(thresholds["high"])


def load_model(hub_repo: str, model_name: str, device: torch.device) -> torch.nn.Module:
    model = torch.hub.load(hub_repo, model_name)
    model.eval()
    model.to(device)
    return model


def embed_image(image_path: Path, model: torch.nn.Module, image_size: int, device: torch.device) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = create_dinov2_transform(image_size=image_size)
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)

    tensor = tensor.to(device)
    autocast_enabled = device.type == "cuda"
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            embedding = model(tensor)
    return embedding.detach().float().cpu().numpy()[0]


def nearest_reference_works(
    query_embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_metadata: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    similarities = cosine_similarity_matrix(query_embedding.astype(np.float32), reference_embeddings.astype(np.float32))
    best_indices = np.argsort(similarities)[::-1][:top_k]
    neighbors: list[dict[str, Any]] = []

    for index in best_indices:
        item = reference_metadata[int(index)]
        neighbors.append(
            {
                "artist": item["artist"],
                "image_path": item["prepared_image_path"],
                "similarity": round(float(similarities[int(index)]), 4),
            }
        )

    return neighbors


def main() -> None:
    args = parse_args()
    device = get_device()
    profiles_payload = load_pickle(PROFILES_PATH)
    profiles = profiles_payload["profiles"]
    available_artists = sorted(profiles.keys())
    low_threshold, high_threshold = load_thresholds()

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading pretrained DINOv2 model...")
    model = load_model(args.hub_repo, args.model_name, device)
    query_embedding = embed_image(args.image, model, args.image_size, device)

    all_scores: list[tuple[str, float]] = []
    for artist_name, profile in profiles.items():
        distance = mahalanobis_distance(
            query_embedding.astype(np.float32),
            profile["centroid"],
            profile["precision"],
        )
        all_scores.append((artist_name, distance))
    all_scores.sort(key=lambda item: item[1])

    predicted_artist, predicted_distance = all_scores[0]
    target_artist = predicted_artist
    if args.claimed_artist:
        target_artist = resolve_artist_name(args.claimed_artist, available_artists)

    target_profile = profiles[target_artist]
    target_distance = mahalanobis_distance(
        query_embedding.astype(np.float32),
        target_profile["centroid"],
        target_profile["precision"],
    )
    target_risk = risk_level(target_distance, low_threshold, high_threshold)

    train_embeddings = np.load(ARTIFACTS_DIR / "embeddings_train.npy")
    train_labels = np.load(ARTIFACTS_DIR / "labels_train.npy")
    train_metadata = load_json(ARTIFACTS_DIR / "metadata_train.json")
    artist_to_index = {artist: int(index) for artist, index in load_json(ARTIFACTS_DIR / "artist_to_index.json").items()}
    target_label_id = artist_to_index[target_artist]
    artist_reference_indices = np.where(train_labels == target_label_id)[0]

    reference_neighbors = nearest_reference_works(
        query_embedding=query_embedding,
        reference_embeddings=train_embeddings[artist_reference_indices],
        reference_metadata=[train_metadata[int(index)] for index in artist_reference_indices],
        top_k=args.top_k,
    )

    report = {
        "image": str(args.image),
        "claimed_artist": target_artist if args.claimed_artist else None,
        "predicted_artist": predicted_artist,
        "predicted_artist_distance": round(float(predicted_distance), 4),
        "target_artist": target_artist,
        "target_distance_score": round(float(target_distance), 4),
        "risk_level": target_risk,
        "validation_thresholds": {
            "low": round(low_threshold, 4),
            "high": round(high_threshold, 4),
        },
        "top_artist_matches": [
            {"artist": artist_name, "distance": round(float(distance), 4)}
            for artist_name, distance in all_scores[:5]
        ],
        "nearest_reference_works": reference_neighbors,
        "next_step": next_step_for_risk(target_risk),
        "baseline_note": (
            "This is a baseline artist-conditioned anomaly model using frozen DINOv2 embeddings. "
            "It is a risk signal, not a final authenticity verdict."
        ),
    }

    print(json.dumps(report, indent=2))
    if args.output_json is not None:
        save_json(args.output_json, report)
        print(f"Saved inference report to {args.output_json}")


if __name__ == "__main__":
    main()
