from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.covariance import LedoitWolf

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.training.utils import ARTIFACTS_DIR, PROFILES_PATH, load_json, save_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit per-artist Gaussian profiles from DINO embeddings.")
    parser.add_argument("--embeddings-path", type=Path, default=ARTIFACTS_DIR / "embeddings_train.npy")
    parser.add_argument("--labels-path", type=Path, default=ARTIFACTS_DIR / "labels_train.npy")
    parser.add_argument("--artist-map", type=Path, default=ARTIFACTS_DIR / "artist_to_index.json")
    parser.add_argument("--output-path", type=Path, default=PROFILES_PATH)
    parser.add_argument("--umap-path", type=Path, default=ARTIFACTS_DIR / "umap_plot.png")
    return parser.parse_args()


def inverse_artist_index(artist_to_index: dict[str, int]) -> dict[int, str]:
    return {index: artist for artist, index in artist_to_index.items()}


def create_umap_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    inverse_index: dict[int, str],
    output_path: Path,
) -> None:
    reducer = umap.UMAP(random_state=42, metric="cosine")
    coords = reducer.fit_transform(embeddings)

    plt.figure(figsize=(14, 10))
    unique_labels = sorted(set(labels.tolist()))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))

    for plot_index, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.75,
            color=cmap(plot_index),
            label=inverse_index[label],
        )

    plt.title("ArtShield AI Embedding Space (UMAP)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    embeddings = np.load(args.embeddings_path)
    labels = np.load(args.labels_path)
    artist_to_index = load_json(args.artist_map)
    inverse_index = inverse_artist_index({artist: int(index) for artist, index in artist_to_index.items()})

    profiles: dict[str, dict[str, np.ndarray | int]] = {}
    for artist, label_id in artist_to_index.items():
        label_id = int(label_id)
        artist_embeddings = embeddings[labels == label_id]
        if len(artist_embeddings) < 2:
            continue

        estimator = LedoitWolf()
        estimator.fit(artist_embeddings)
        profiles[artist] = {
            "label_id": label_id,
            "count": int(len(artist_embeddings)),
            "centroid": estimator.location_.astype(np.float32),
            "covariance": estimator.covariance_.astype(np.float32),
            "precision": estimator.precision_.astype(np.float32),
        }

    payload = {
        "embedding_dim": int(embeddings.shape[1]),
        "profiles": profiles,
    }
    save_pickle(args.output_path, payload)
    print(f"Saved {len(profiles)} artist profiles to {args.output_path}")

    create_umap_plot(embeddings=embeddings, labels=labels, inverse_index=inverse_index, output_path=args.umap_path)
    print(f"Saved UMAP visualization to {args.umap_path}")


if __name__ == "__main__":
    main()
