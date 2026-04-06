from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import ClassLabel, load_dataset
from PIL import Image
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "ml" / "data" / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"
METADATA_PATH = PROCESSED_DIR / "metadata.jsonl"
ARTISTS_PATH = PROCESSED_DIR / "artists.json"
DEFAULT_ARTISTS = [
    "claude-monet",
    "vincent-van-gogh",
    "pablo-picasso",
    "paul-cezanne",
    "pierre-auguste-renoir",
    "rembrandt",
    "salvador-dali",
    "albrecht-durer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and filter WikiArt for ArtShield AI.")
    parser.add_argument("--dataset-name", default="huggan/wikiart", help="Hugging Face dataset name.")
    parser.add_argument("--split", default="train", help="Dataset split to download.")
    parser.add_argument(
        "--artists",
        nargs="+",
        default=DEFAULT_ARTISTS,
        help="Artist slugs to keep. Defaults to a curated starter set.",
    )
    parser.add_argument(
        "--max-images-per-artist",
        type=int,
        default=80,
        help="Maximum number of works to save for each selected artist.",
    )
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use streaming mode to avoid downloading the full dataset first.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR, help="Processed output directory.")
    return parser.parse_args()


def decode_feature_value(feature: Any, value: Any) -> str:
    if isinstance(feature, ClassLabel):
        return feature.int2str(int(value))
    return str(value)


def slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value.lower()).strip("-")


def convert_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clean_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                for nested in child.rglob("*"):
                    if nested.is_file():
                        nested.unlink()
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_dir():
                        nested.rmdir()
                child.rmdir()
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    clean_output_dir(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    artist_slugs = [slugify(artist) for artist in args.artists]
    selected_artists = set(artist_slugs)
    max_images_per_artist = args.max_images_per_artist

    print(
        f"Loading dataset {args.dataset_name}:{args.split} "
        f"with streaming={'on' if args.streaming else 'off'}"
    )
    dataset = load_dataset(args.dataset_name, split=args.split, streaming=args.streaming)

    artist_feature = dataset.features["artist"]
    style_feature = dataset.features.get("style")
    genre_feature = dataset.features.get("genre")

    saved_records: list[dict[str, Any]] = []
    saved_counts = {artist_slug: 0 for artist_slug in selected_artists}
    skipped = 0
    total_target = len(selected_artists) * max_images_per_artist

    print(
        f"Collecting up to {max_images_per_artist} works each for {len(selected_artists)} artists "
        f"({total_target} images max)."
    )
    for index, sample in enumerate(tqdm(dataset, desc="Saving filtered images", total=total_target)):
        artist_slug = slugify(decode_feature_value(artist_feature, sample["artist"]))
        if artist_slug not in selected_artists:
            continue
        if saved_counts[artist_slug] >= max_images_per_artist:
            if all(count >= max_images_per_artist for count in saved_counts.values()):
                break
            continue

        try:
            image = convert_to_rgb(sample["image"])
        except Exception:
            skipped += 1
            continue

        filename = f"{artist_slug}-{saved_counts[artist_slug]:04d}.jpg"
        relative_path = Path("images") / artist_slug / filename
        absolute_path = output_dir / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(absolute_path, format="JPEG", quality=95)

        artist_name = decode_feature_value(artist_feature, sample["artist"])
        style_name = decode_feature_value(style_feature, sample["style"]) if style_feature else None
        genre_name = decode_feature_value(genre_feature, sample["genre"]) if genre_feature else None

        saved_records.append(
            {
                "id": f"{artist_slug}-{saved_counts[artist_slug]:04d}",
                "source_index": index,
                "artist": artist_name,
                "artist_slug": artist_slug,
                "style": style_name,
                "genre": genre_name,
                "image_path": str(relative_path).replace("\\", "/"),
            }
        )
        saved_counts[artist_slug] += 1

        if all(count >= max_images_per_artist for count in saved_counts.values()):
            break

    selected_summary = [
        {
            "artist": artist_slug,
            "count": saved_counts[artist_slug],
        }
        for artist_slug in sorted(selected_artists)
    ]
    write_json(output_dir / "artists.json", selected_summary)

    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        for record in saved_records:
            handle.write(json.dumps(record) + "\n")

    print(f"Saved {len(saved_records)} images to {output_dir}")
    print(f"Skipped {skipped} records due to image conversion issues")
    print(f"Artist summary written to {ARTISTS_PATH}")
    print(f"Metadata written to {METADATA_PATH}")
    print("Saved counts:", saved_counts)


if __name__ == "__main__":
    main()
