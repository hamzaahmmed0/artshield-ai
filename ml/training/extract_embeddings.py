from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.training.utils import (
    ARTIFACTS_DIR,
    build_artist_index,
    collate_artworks,
    create_dinov2_transform,
    ensure_dir,
    get_device,
    load_split_manifest,
    save_json,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen DINOv2 embeddings for ArtShield AI.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--model-name", default="dinov2_vitb14")
    parser.add_argument("--hub-repo", default="facebookresearch/dinov2")
    return parser.parse_args()


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, records, transform):
        from ml.training.utils import ArtworkDataset

        self.dataset = ArtworkDataset(records=records, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def extract_split_embeddings(
    *,
    split_name: str,
    records: list[dict],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    image_size: int,
    artist_to_index: dict[str, int],
) -> None:
    transform = create_dinov2_transform(image_size=image_size)
    dataset = DatasetWrapper(records=records, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_artworks,
    )

    embeddings: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict] = []

    autocast_enabled = device.type == "cuda"
    for images, batch_records in tqdm(dataloader, desc=f"Extracting {split_name} embeddings"):
        images = images.to(device, non_blocking=True)
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
                batch_embeddings = model(images)
        batch_embeddings = batch_embeddings.detach().float().cpu().numpy()
        embeddings.append(batch_embeddings)
        labels.extend(artist_to_index[record["artist"]] for record in batch_records)
        metadata.extend(batch_records)

    split_embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
    split_labels = np.asarray(labels, dtype=np.int32)

    np.save(ARTIFACTS_DIR / f"embeddings_{split_name}.npy", split_embeddings)
    np.save(ARTIFACTS_DIR / f"labels_{split_name}.npy", split_labels)
    save_json(ARTIFACTS_DIR / f"metadata_{split_name}.json", metadata)
    print(f"Saved {split_name} embeddings with shape {split_embeddings.shape}")


def main() -> None:
    args = parse_args()
    ensure_dir(ARTIFACTS_DIR)
    manifest = load_split_manifest()
    all_records = manifest["splits"]["train"] + manifest["splits"]["val"] + manifest["splits"]["test"]
    artist_to_index = build_artist_index(all_records)

    device = get_device()
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading pretrained DINOv2 model from torch.hub...")
    model = torch.hub.load(args.hub_repo, args.model_name)
    model.eval()
    model.to(device)

    save_json(
        ARTIFACTS_DIR / "artist_to_index.json",
        artist_to_index,
    )
    save_json(
        ARTIFACTS_DIR / "embedding_manifest.json",
        {
            "model_name": args.model_name,
            "hub_repo": args.hub_repo,
            "image_size": args.image_size,
            "device": str(device),
            "splits": {split: len(records) for split, records in manifest["splits"].items()},
        },
    )

    for split_name, records in manifest["splits"].items():
        extract_split_embeddings(
            split_name=split_name,
            records=records,
            model=model,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            artist_to_index=artist_to_index,
        )


if __name__ == "__main__":
    main()
