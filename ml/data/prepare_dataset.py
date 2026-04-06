from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from math import floor
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "ml" / "data" / "processed"
METADATA_PATH = PROCESSED_DIR / "metadata.jsonl"
SPLITS_PATH = PROCESSED_DIR / "splits.json"
RESIZED_DIR = PROCESSED_DIR / "images_224"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare WikiArt images and create train/val/test splits.")
    parser.add_argument("--input-metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--output-splits", type=Path, default=SPLITS_PATH)
    parser.add_argument("--output-images", type=Path, default=RESIZED_DIR)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resize_and_pad(image: Image.Image, image_size: int) -> Image.Image:
    rgb = image.convert("RGB")
    return ImageOps.pad(rgb, (image_size, image_size), method=Image.Resampling.BICUBIC, color=(255, 255, 255))


def split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_count = floor(total * train_ratio)
    val_count = floor(total * val_ratio)
    test_count = total - train_count - val_count

    if total >= 3:
        train_count = max(train_count, 1)
        val_count = max(val_count, 1)
        test_count = total - train_count - val_count
        if test_count <= 0:
            test_count = 1
            if train_count > val_count:
                train_count -= 1
            else:
                val_count -= 1
    return train_count, val_count, test_count


def main() -> None:
    args = parse_args()
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.0.")

    records = load_jsonl(args.input_metadata)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["artist"]].append(record)

    args.output_images.mkdir(parents=True, exist_ok=True)
    prepared_records: list[dict[str, Any]] = []

    print("Resizing artwork images to a consistent 224x224 canvas...")
    for record in tqdm(records, desc="Preparing images"):
        source_path = PROCESSED_DIR / record["image_path"]
        output_path = args.output_images / record["artist_slug"] / Path(record["image_path"]).name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(source_path) as image:
            prepared = resize_and_pad(image, args.image_size)
            prepared.save(output_path, format="JPEG", quality=95)

        prepared_record = dict(record)
        prepared_record["prepared_image_path"] = str(output_path.relative_to(PROCESSED_DIR)).replace("\\", "/")
        prepared_records.append(prepared_record)

    prepared_by_id = {record["id"]: record for record in prepared_records}
    rng = random.Random(args.seed)
    splits = {"train": [], "val": [], "test": []}

    print("Creating artist-balanced splits...")
    for artist_name, artist_records in grouped.items():
        ids = [record["id"] for record in artist_records]
        rng.shuffle(ids)
        train_count, val_count, test_count = split_counts(len(ids), args.train_ratio, args.val_ratio)

        train_ids = ids[:train_count]
        val_ids = ids[train_count : train_count + val_count]
        test_ids = ids[train_count + val_count : train_count + val_count + test_count]

        splits["train"].extend(prepared_by_id[item_id] for item_id in train_ids)
        splits["val"].extend(prepared_by_id[item_id] for item_id in val_ids)
        splits["test"].extend(prepared_by_id[item_id] for item_id in test_ids)

    payload = {
        "image_size": args.image_size,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "splits": splits,
    }
    save_json(args.output_splits, payload)
    print(f"Saved split manifest to {args.output_splits}")
    print(
        "Split sizes:",
        {name: len(items) for name, items in splits.items()},
    )


if __name__ == "__main__":
    main()

