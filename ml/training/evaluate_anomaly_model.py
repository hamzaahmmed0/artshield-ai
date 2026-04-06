from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.training.utils import (
    ARTIFACTS_DIR,
    PROFILES_PATH,
    cosine_similarity_matrix,
    load_json,
    load_pickle,
    mahalanobis_distance,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate artist anomaly profiles on true and impostor claims.")
    parser.add_argument("--profiles-path", type=Path, default=PROFILES_PATH)
    parser.add_argument("--artist-map", type=Path, default=ARTIFACTS_DIR / "artist_to_index.json")
    parser.add_argument("--output-report", type=Path, default=ARTIFACTS_DIR / "evaluation_report_v2.json")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest reference works to include per claim.")
    parser.add_argument("--train-embeddings-path", type=Path, default=ARTIFACTS_DIR / "embeddings_train.npy")
    parser.add_argument("--train-labels-path", type=Path, default=ARTIFACTS_DIR / "labels_train.npy")
    parser.add_argument("--train-metadata-path", type=Path, default=ARTIFACTS_DIR / "metadata_train.json")
    parser.add_argument("--val-embeddings-path", type=Path, default=ARTIFACTS_DIR / "embeddings_val.npy")
    parser.add_argument("--val-labels-path", type=Path, default=ARTIFACTS_DIR / "labels_val.npy")
    parser.add_argument("--val-metadata-path", type=Path, default=ARTIFACTS_DIR / "metadata_val.json")
    parser.add_argument("--test-embeddings-path", type=Path, default=ARTIFACTS_DIR / "embeddings_test.npy")
    parser.add_argument("--test-labels-path", type=Path, default=ARTIFACTS_DIR / "labels_test.npy")
    parser.add_argument("--test-metadata-path", type=Path, default=ARTIFACTS_DIR / "metadata_test.json")
    return parser.parse_args()


def load_artifacts(
    embeddings_path: Path,
    labels_path: Path,
    metadata_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    metadata = load_json(metadata_path)
    return embeddings, labels, metadata


def normalize_path(value: str | None) -> str | None:
    if value is None:
        return None
    return value.replace("\\", "/").lower()


def metadata_match_keys(record: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in ("id", "prepared_image_path", "image_path"):
        value = record.get(field)
        normalized = normalize_path(str(value)) if value is not None else None
        if normalized:
            keys.add(normalized)
            keys.add(normalize_path(Path(normalized).name) or "")
    keys.discard("")
    return keys


def build_artist_reference_index(
    labels: np.ndarray,
    metadata: list[dict[str, Any]],
    inverse_index: dict[int, str],
) -> dict[str, list[int]]:
    indices_by_artist: dict[str, list[int]] = defaultdict(list)
    for index, label_id in enumerate(labels.tolist()):
        artist_name = inverse_index[int(label_id)]
        indices_by_artist[artist_name].append(index)
    return dict(indices_by_artist)


def filter_reference_indices(
    reference_indices: list[int],
    reference_metadata: list[dict[str, Any]],
    query_metadata: dict[str, Any],
) -> tuple[list[int], int]:
    query_keys = metadata_match_keys(query_metadata)
    kept_indices: list[int] = []
    removed = 0

    for index in reference_indices:
        candidate_keys = metadata_match_keys(reference_metadata[index])
        if query_keys & candidate_keys:
            removed += 1
            continue
        kept_indices.append(index)

    return kept_indices, removed


def get_profile_for_claim(
    *,
    claimed_artist: str,
    query_metadata: dict[str, Any],
    base_profiles: dict[str, dict[str, Any]],
    train_embeddings: np.ndarray,
    train_metadata: list[dict[str, Any]],
    reference_indices_by_artist: dict[str, list[int]],
) -> tuple[np.ndarray, np.ndarray, int]:
    base_profile = base_profiles[claimed_artist]
    artist_indices = reference_indices_by_artist[claimed_artist]
    filtered_indices, removed = filter_reference_indices(artist_indices, train_metadata, query_metadata)

    if removed == 0 or len(filtered_indices) < 2:
        return base_profile["centroid"], base_profile["precision"], removed

    filtered_embeddings = train_embeddings[np.asarray(filtered_indices, dtype=np.int32)]
    estimator = LedoitWolf()
    estimator.fit(filtered_embeddings)
    return (
        estimator.location_.astype(np.float32),
        estimator.precision_.astype(np.float32),
        removed,
    )


def calibrate_thresholds(
    *,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    val_metadata: list[dict[str, Any]],
    inverse_index: dict[int, str],
    profiles: dict[str, dict[str, Any]],
    train_embeddings: np.ndarray,
    train_metadata: list[dict[str, Any]],
    reference_indices_by_artist: dict[str, list[int]],
) -> tuple[float, float]:
    distances: list[float] = []
    for embedding, label_id, metadata in zip(val_embeddings, val_labels, val_metadata, strict=True):
        artist = inverse_index[int(label_id)]
        centroid, precision, _ = get_profile_for_claim(
            claimed_artist=artist,
            query_metadata=metadata,
            base_profiles=profiles,
            train_embeddings=train_embeddings,
            train_metadata=train_metadata,
            reference_indices_by_artist=reference_indices_by_artist,
        )
        distances.append(mahalanobis_distance(embedding.astype(np.float32), centroid, precision))

    low_threshold = float(np.quantile(distances, 0.85))
    high_threshold = float(np.quantile(distances, 0.95))
    return low_threshold, high_threshold


def risk_level(distance: float, low_threshold: float, high_threshold: float) -> str:
    if distance <= low_threshold:
        return "LOW"
    if distance <= high_threshold:
        return "MEDIUM"
    return "HIGH"


def nearest_reference_works(
    *,
    query_embedding: np.ndarray,
    query_metadata: dict[str, Any],
    claimed_artist: str,
    train_embeddings: np.ndarray,
    train_metadata: list[dict[str, Any]],
    reference_indices_by_artist: dict[str, list[int]],
    top_k: int,
) -> tuple[list[dict[str, Any]], int]:
    artist_indices = reference_indices_by_artist[claimed_artist]
    filtered_indices, removed = filter_reference_indices(artist_indices, train_metadata, query_metadata)
    if not filtered_indices:
        return [], removed

    filtered_embeddings = train_embeddings[np.asarray(filtered_indices, dtype=np.int32)]
    filtered_metadata = [train_metadata[index] for index in filtered_indices]
    similarities = cosine_similarity_matrix(query_embedding.astype(np.float32), filtered_embeddings.astype(np.float32))
    best_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for local_index in best_indices:
        item = filtered_metadata[int(local_index)]
        results.append(
            {
                "artist": item["artist"],
                "image_path": item["prepared_image_path"],
                "similarity": round(float(similarities[int(local_index)]), 4),
            }
        )
    return results, removed


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def format_row(row: list[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    divider = "-+-".join("-" * width for width in widths)
    lines = [format_row(headers), divider]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def compute_per_artist_summary(
    *,
    available_artists: list[str],
    claim_evaluations: list[dict[str, Any]],
    retrieval_hits_by_artist: dict[str, int],
    retrieval_counts_by_artist: dict[str, int],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}

    for artist in available_artists:
        true_claims = [item for item in claim_evaluations if item["true_artist"] == artist and item["claimed_artist"] == artist]
        wrong_claims = [item for item in claim_evaluations if item["claimed_artist"] == artist and item["true_artist"] != artist]

        true_low = sum(item["risk_level"] == "LOW" for item in true_claims)
        wrong_low = sum(item["risk_level"] == "LOW" for item in wrong_claims)
        wrong_non_low = sum(item["risk_level"] in {"MEDIUM", "HIGH"} for item in wrong_claims)

        summary[artist] = {
            "true_claims": len(true_claims),
            "wrong_claims": len(wrong_claims),
            "tpr": round(safe_divide(true_low, len(true_claims)), 4),
            "fpr": round(safe_divide(wrong_low, len(wrong_claims)), 4),
            "tnr": round(safe_divide(wrong_non_low, len(wrong_claims)), 4),
            "top1_artist_retrieval_accuracy": round(
                safe_divide(retrieval_hits_by_artist.get(artist, 0), retrieval_counts_by_artist.get(artist, 0)),
                4,
            ),
            "average_true_distance": round(
                float(np.mean([item["distance_score"] for item in true_claims])) if true_claims else 0.0,
                4,
            ),
            "average_wrong_distance": round(
                float(np.mean([item["distance_score"] for item in wrong_claims])) if wrong_claims else 0.0,
                4,
            ),
        }

    return summary


def evaluate_artifact_set(
    *,
    profiles_path: Path,
    artist_map_path: Path,
    train_embeddings_path: Path,
    train_labels_path: Path,
    train_metadata_path: Path,
    val_embeddings_path: Path,
    val_labels_path: Path,
    val_metadata_path: Path,
    test_embeddings_path: Path,
    test_labels_path: Path,
    test_metadata_path: Path,
    top_k: int,
) -> dict[str, Any]:
    payload = load_pickle(profiles_path)
    profiles = payload["profiles"]
    artist_to_index = {artist: int(index) for artist, index in load_json(artist_map_path).items()}
    inverse_index = {index: artist for artist, index in artist_to_index.items()}
    available_artists = sorted(profiles.keys())

    train_embeddings, train_labels, train_metadata = load_artifacts(train_embeddings_path, train_labels_path, train_metadata_path)
    val_embeddings, val_labels, val_metadata = load_artifacts(val_embeddings_path, val_labels_path, val_metadata_path)
    test_embeddings, test_labels, test_metadata = load_artifacts(test_embeddings_path, test_labels_path, test_metadata_path)
    reference_indices_by_artist = build_artist_reference_index(train_labels, train_metadata, inverse_index)

    low_threshold, high_threshold = calibrate_thresholds(
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        val_metadata=val_metadata,
        inverse_index=inverse_index,
        profiles=profiles,
        train_embeddings=train_embeddings,
        train_metadata=train_metadata,
        reference_indices_by_artist=reference_indices_by_artist,
    )

    claim_evaluations: list[dict[str, Any]] = []
    retrieval_hits = 0
    retrieval_hits_by_artist: dict[str, int] = defaultdict(int)
    retrieval_counts_by_artist: dict[str, int] = defaultdict(int)
    self_match_removed_total = 0

    for embedding, label_id, metadata in zip(test_embeddings, test_labels, test_metadata, strict=True):
        true_artist = inverse_index[int(label_id)]
        retrieval_counts_by_artist[true_artist] += 1

        all_artist_scores: list[tuple[str, float]] = []
        artist_profiles_for_query: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}

        for artist_name in available_artists:
            centroid, precision, removed = get_profile_for_claim(
                claimed_artist=artist_name,
                query_metadata=metadata,
                base_profiles=profiles,
                train_embeddings=train_embeddings,
                train_metadata=train_metadata,
                reference_indices_by_artist=reference_indices_by_artist,
            )
            distance = mahalanobis_distance(embedding.astype(np.float32), centroid, precision)
            all_artist_scores.append((artist_name, distance))
            artist_profiles_for_query[artist_name] = (centroid, precision, removed)

        all_artist_scores.sort(key=lambda item: item[1])
        predicted_artist = all_artist_scores[0][0]
        if predicted_artist == true_artist:
            retrieval_hits += 1
            retrieval_hits_by_artist[true_artist] += 1

        for claimed_artist in available_artists:
            _, _, removed_for_profile = artist_profiles_for_query[claimed_artist]
            distance_score = next(distance for artist_name, distance in all_artist_scores if artist_name == claimed_artist)
            risk = risk_level(distance_score, low_threshold, high_threshold)
            nearest_neighbors, removed_for_neighbors = nearest_reference_works(
                query_embedding=embedding,
                query_metadata=metadata,
                claimed_artist=claimed_artist,
                train_embeddings=train_embeddings,
                train_metadata=train_metadata,
                reference_indices_by_artist=reference_indices_by_artist,
                top_k=top_k,
            )
            self_match_removed_total += removed_for_profile + removed_for_neighbors

            claim_evaluations.append(
                {
                    "id": metadata["id"],
                    "image_path": metadata["prepared_image_path"],
                    "true_artist": true_artist,
                    "claimed_artist": claimed_artist,
                    "claim_type": "true" if claimed_artist == true_artist else "impostor",
                    "predicted_artist": predicted_artist,
                    "distance_score": round(float(distance_score), 4),
                    "risk_level": risk,
                    "is_low_risk": risk == "LOW",
                    "is_top1_correct": predicted_artist == true_artist,
                    "top_artist_matches": [
                        {"artist": artist_name, "distance": round(float(distance), 4)}
                        for artist_name, distance in all_artist_scores[:5]
                    ],
                    "nearest_reference_works": nearest_neighbors,
                    "self_matches_removed": removed_for_profile + removed_for_neighbors,
                }
            )

    true_claims = [item for item in claim_evaluations if item["claim_type"] == "true"]
    wrong_claims = [item for item in claim_evaluations if item["claim_type"] == "impostor"]

    true_low = sum(item["risk_level"] == "LOW" for item in true_claims)
    wrong_low = sum(item["risk_level"] == "LOW" for item in wrong_claims)
    wrong_non_low = sum(item["risk_level"] in {"MEDIUM", "HIGH"} for item in wrong_claims)

    overall_summary = {
        "validation_thresholds": {
            "low": round(low_threshold, 4),
            "high": round(high_threshold, 4),
        },
        "test_images": len(test_metadata),
        "claim_evaluations": len(claim_evaluations),
        "true_claims": len(true_claims),
        "wrong_claims": len(wrong_claims),
        "true_positive_rate": round(safe_divide(true_low, len(true_claims)), 4),
        "false_positive_rate": round(safe_divide(wrong_low, len(wrong_claims)), 4),
        "true_negative_rate": round(safe_divide(wrong_non_low, len(wrong_claims)), 4),
        "top1_artist_retrieval_accuracy": round(safe_divide(retrieval_hits, len(test_metadata)), 4),
        "true_claim_risk_breakdown": {
            "LOW": sum(item["risk_level"] == "LOW" for item in true_claims),
            "MEDIUM": sum(item["risk_level"] == "MEDIUM" for item in true_claims),
            "HIGH": sum(item["risk_level"] == "HIGH" for item in true_claims),
        },
        "wrong_claim_risk_breakdown": {
            "LOW": sum(item["risk_level"] == "LOW" for item in wrong_claims),
            "MEDIUM": sum(item["risk_level"] == "MEDIUM" for item in wrong_claims),
            "HIGH": sum(item["risk_level"] == "HIGH" for item in wrong_claims),
        },
        "self_matches_removed_total": self_match_removed_total,
    }

    per_artist_summary = compute_per_artist_summary(
        available_artists=available_artists,
        claim_evaluations=claim_evaluations,
        retrieval_hits_by_artist=retrieval_hits_by_artist,
        retrieval_counts_by_artist=retrieval_counts_by_artist,
    )

    return {
        "summary": overall_summary,
        "per_artist": per_artist_summary,
        "claim_evaluations": claim_evaluations,
    }


def print_report_tables(report: dict[str, Any]) -> None:
    overall_summary = report["summary"]
    per_artist_summary = report["per_artist"]

    summary_rows = [
        ["TPR", percentage(overall_summary["true_positive_rate"])],
        ["FPR", percentage(overall_summary["false_positive_rate"])],
        ["TNR", percentage(overall_summary["true_negative_rate"])],
        ["Top-1 Retrieval", percentage(overall_summary["top1_artist_retrieval_accuracy"])],
        ["True Claims", str(overall_summary["true_claims"])],
        ["Wrong Claims", str(overall_summary["wrong_claims"])],
        ["Self-Matches Removed", str(overall_summary["self_matches_removed_total"])],
    ]
    print("Overall Summary")
    print(render_table(["Metric", "Value"], summary_rows))

    artist_rows = []
    for artist_name, artist_summary in per_artist_summary.items():
        artist_rows.append(
            [
                artist_name,
                percentage(artist_summary["tpr"]),
                percentage(artist_summary["fpr"]),
                percentage(artist_summary["tnr"]),
                percentage(artist_summary["top1_artist_retrieval_accuracy"]),
            ]
        )
    print("\nPer-Artist Summary")
    print(render_table(["Artist", "TPR", "FPR", "TNR", "Top-1"], artist_rows))


def main() -> None:
    args = parse_args()
    report = evaluate_artifact_set(
        profiles_path=args.profiles_path,
        artist_map_path=args.artist_map,
        train_embeddings_path=args.train_embeddings_path,
        train_labels_path=args.train_labels_path,
        train_metadata_path=args.train_metadata_path,
        val_embeddings_path=args.val_embeddings_path,
        val_labels_path=args.val_labels_path,
        val_metadata_path=args.val_metadata_path,
        test_embeddings_path=args.test_embeddings_path,
        test_labels_path=args.test_labels_path,
        test_metadata_path=args.test_metadata_path,
        top_k=args.top_k,
    )
    save_json(args.output_report, report)
    print_report_tables(report)
    print(f"\nDetailed report saved to {args.output_report}")


if __name__ == "__main__":
    main()
