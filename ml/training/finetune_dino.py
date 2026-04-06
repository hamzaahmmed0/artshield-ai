from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
from sklearn.covariance import LedoitWolf
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.training.evaluate_anomaly_model import evaluate_artifact_set, print_report_tables, render_table
from ml.training.utils import (
    ARTIFACTS_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODELS_DIR,
    build_artist_index,
    ensure_dir,
    get_device,
    load_json,
    load_split_manifest,
    save_json,
    save_pickle,
)


class LabeledArtworkDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        artist_to_index: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.records = records
        self.artist_to_index = artist_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, dict[str, Any]]:
        from ml.training.utils import DATA_DIR
        from PIL import Image

        record = self.records[index]
        image_path = DATA_DIR / record["prepared_image_path"]
        with image_path.open("rb") as handle:
            image = Image.open(handle).convert("RGB")
            tensor = self.transform(image)
        label = self.artist_to_index[record["artist"]]
        return tensor, label, record


def classification_collate(
    batch: list[tuple[torch.Tensor, int, dict[str, Any]]]
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    images, labels, records = zip(*batch)
    return torch.stack(list(images), dim=0), torch.tensor(labels, dtype=torch.long), list(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DINOv2 on artist classification for ArtShield AI.")
    parser.add_argument("--splits-path", type=Path, default=ROOT_DIR / "ml" / "data" / "processed" / "splits.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--checkpoint-path", type=Path, default=MODELS_DIR / "finetuned_dino.pth")
    parser.add_argument("--finetune-log-path", type=Path, default=ARTIFACTS_DIR / "finetune_log.json")
    parser.add_argument("--finetuned-profile-path", type=Path, default=MODELS_DIR / "artist_profiles_finetuned.pkl")
    parser.add_argument("--finetuned-eval-path", type=Path, default=ARTIFACTS_DIR / "evaluation_report_v3.json")
    parser.add_argument("--baseline-eval-path", type=Path, default=ARTIFACTS_DIR / "evaluation_report_v2.json")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=16),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def create_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class FineTunedDinoClassifier(nn.Module):
    def __init__(self, num_classes: int, image_size: int, model_name: str) -> None:
        super().__init__()
        self.backbone_source = "timm"
        try:
            self.backbone = timm.create_model(model_name, pretrained=True, img_size=image_size, num_classes=0)
            self.embedding_dim = int(self.backbone.num_features)
        except Exception as exc:
            print(f"timm pretrained load failed ({exc}). Falling back to cached torch.hub DINOv2 weights.")
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            self.embedding_dim = int(self.backbone.embed_dim)
            self.backbone_source = "torchhub"
        self.head = nn.Linear(self.embedding_dim, num_classes)
        self._freeze_for_finetuning()

    def _freeze_for_finetuning(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for block in self.backbone.blocks[-2:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

        for parameter in self.backbone.norm.parameters():
            parameter.requires_grad = True

    def extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)
        if self.backbone_source == "timm":
            return self.backbone.forward_head(features, pre_logits=True)
        return features["x_norm_clstoken"]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.extract_embeddings(images)
        return self.head(embeddings)


def create_dataloader(
    records: list[dict[str, Any]],
    artist_to_index: dict[str, int],
    transform: transforms.Compose,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    dataset = LabeledArtworkDataset(records=records, artist_to_index=artist_to_index, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=classification_collate,
    )


def create_dataloaders(
    manifest: dict[str, Any],
    artist_to_index: dict[str, int],
    batch_size: int,
    num_workers: int,
    image_size: int,
    device: torch.device,
) -> tuple[dict[str, DataLoader], dict[str, DataLoader]]:
    train_loader = create_dataloader(
        manifest["splits"]["train"],
        artist_to_index,
        create_train_transform(image_size),
        batch_size,
        True,
        num_workers,
        device,
    )
    eval_loaders = {
        split_name: create_dataloader(
            manifest["splits"][split_name],
            artist_to_index,
            create_eval_transform(),
            batch_size,
            False,
            num_workers,
            device,
        )
        for split_name in ("train", "val", "test")
    }
    return {"train": train_loader, "val": eval_loaders["val"]}, eval_loaders


def run_epoch(
    *,
    model: FineTunedDinoClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW | None,
    device: torch.device,
    training: bool,
) -> tuple[float, float]:
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    scaler = torch.amp.GradScaler("cuda", enabled=training and device.type == "cuda")

    progress = tqdm(dataloader, desc="train" if training else "val", leave=False)
    for images, labels, _ in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            if training and optimizer is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_examples, 1), total_correct / max(total_examples, 1)


def build_optimizer(model: FineTunedDinoClassifier, head_lr: float, backbone_lr: float, weight_decay: float) -> AdamW:
    head_params = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    backbone_params = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    return AdamW(
        [
            {"params": head_params, "lr": head_lr},
            {"params": backbone_params, "lr": backbone_lr},
        ],
        weight_decay=weight_decay,
    )


def save_checkpoint(
    checkpoint_path: Path,
    model: FineTunedDinoClassifier,
    artist_to_index: dict[str, int],
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "artist_to_index": artist_to_index,
        "best_val_loss": best_val_loss,
        "model_name": args.model_name,
        "backbone_source": model.backbone_source,
        "image_size": args.image_size,
        "num_classes": len(artist_to_index),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def extract_split_embeddings(
    *,
    split_name: str,
    model: FineTunedDinoClassifier,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    model.eval()
    embeddings: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []

    for images, batch_labels, batch_records in tqdm(dataloader, desc=f"Embedding {split_name}", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                batch_embeddings = model.extract_embeddings(images)
        embeddings.append(batch_embeddings.detach().float().cpu().numpy())
        labels.extend(int(label) for label in batch_labels.tolist())
        metadata.extend(batch_records)

    return (
        np.concatenate(embeddings, axis=0).astype(np.float32),
        np.asarray(labels, dtype=np.int32),
        metadata,
    )


def save_finetuned_embeddings(
    *,
    model: FineTunedDinoClassifier,
    eval_loaders: dict[str, DataLoader],
    device: torch.device,
    artist_to_index: dict[str, int],
    checkpoint_path: Path,
) -> dict[str, Path]:
    output_paths: dict[str, Path] = {}
    for split_name, dataloader in eval_loaders.items():
        embeddings, labels, metadata = extract_split_embeddings(
            split_name=split_name,
            model=model,
            dataloader=dataloader,
            device=device,
        )
        embeddings_path = ARTIFACTS_DIR / f"embeddings_{split_name}_finetuned.npy"
        labels_path = ARTIFACTS_DIR / f"labels_{split_name}_finetuned.npy"
        metadata_path = ARTIFACTS_DIR / f"metadata_{split_name}_finetuned.json"
        np.save(embeddings_path, embeddings)
        np.save(labels_path, labels)
        save_json(metadata_path, metadata)
        output_paths[f"{split_name}_embeddings"] = embeddings_path
        output_paths[f"{split_name}_labels"] = labels_path
        output_paths[f"{split_name}_metadata"] = metadata_path
        print(f"Saved fine-tuned {split_name} embeddings with shape {embeddings.shape}")

    artist_map_path = ARTIFACTS_DIR / "artist_to_index_finetuned.json"
    save_json(artist_map_path, artist_to_index)
    output_paths["artist_map"] = artist_map_path
    save_json(
        ARTIFACTS_DIR / "embedding_manifest_finetuned.json",
        {
            "model_name": "finetuned_dinov2_vitb14",
            "backbone_source": model.backbone_source,
            "source_checkpoint": str(checkpoint_path),
            "splits": {
                split_name: int(np.load(output_paths[f"{split_name}_labels"]).shape[0])
                for split_name in ("train", "val", "test")
            },
        },
    )
    return output_paths


def fit_artist_profiles(
    *,
    embeddings: np.ndarray,
    labels: np.ndarray,
    artist_to_index: dict[str, int],
    output_path: Path,
) -> None:
    profiles: dict[str, dict[str, Any]] = {}
    for artist, label_id in artist_to_index.items():
        artist_embeddings = embeddings[labels == int(label_id)]
        if len(artist_embeddings) < 2:
            continue
        estimator = LedoitWolf()
        estimator.fit(artist_embeddings)
        profiles[artist] = {
            "label_id": int(label_id),
            "count": int(len(artist_embeddings)),
            "centroid": estimator.location_.astype(np.float32),
            "covariance": estimator.covariance_.astype(np.float32),
            "precision": estimator.precision_.astype(np.float32),
        }

    payload = {
        "embedding_dim": int(embeddings.shape[1]),
        "profiles": profiles,
    }
    save_pickle(output_path, payload)
    print(f"Saved fine-tuned artist profiles to {output_path}")


def comparison_table(
    baseline_report: dict[str, Any],
    finetuned_report: dict[str, Any],
) -> str:
    baseline_summary = baseline_report["summary"]
    finetuned_summary = finetuned_report["summary"]
    rows = [
        ["TPR", f"{baseline_summary['true_positive_rate'] * 100:.2f}%", f"{finetuned_summary['true_positive_rate'] * 100:.2f}%"],
        ["FPR", f"{baseline_summary['false_positive_rate'] * 100:.2f}%", f"{finetuned_summary['false_positive_rate'] * 100:.2f}%"],
        ["TNR", f"{baseline_summary['true_negative_rate'] * 100:.2f}%", f"{finetuned_summary['true_negative_rate'] * 100:.2f}%"],
        ["Top-1 Retrieval", f"{baseline_summary['top1_artist_retrieval_accuracy'] * 100:.2f}%", f"{finetuned_summary['top1_artist_retrieval_accuracy'] * 100:.2f}%"],
    ]
    return render_table(["Metric", "Baseline", "Fine-Tuned"], rows)


def main() -> None:
    args = parse_args()
    ensure_dir(ARTIFACTS_DIR)
    set_seed(args.seed)

    manifest = load_split_manifest(args.splits_path)
    all_records = manifest["splits"]["train"] + manifest["splits"]["val"] + manifest["splits"]["test"]
    artist_to_index = build_artist_index(all_records)
    num_classes = len(artist_to_index)

    device = get_device()
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_val_loaders, eval_loaders = create_dataloaders(
        manifest=manifest,
        artist_to_index=artist_to_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        device=device,
    )

    print("Loading pretrained DINOv2 backbone from timm...")
    model = FineTunedDinoClassifier(num_classes=num_classes, image_size=args.image_size, model_name=args.model_name)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args.head_lr, args.backbone_lr, args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_state_dict: dict[str, Any] | None = None
    finetune_log: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            dataloader=train_val_loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=True,
        )
        val_loss, val_acc = run_epoch(
            model=model,
            dataloader=train_val_loaders["val"],
            criterion=criterion,
            optimizer=None,
            device=device,
            training=False,
        )
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "head_lr": round(float(optimizer.param_groups[0]["lr"]), 10),
            "backbone_lr": round(float(optimizer.param_groups[1]["lr"]), 10),
        }
        finetune_log.append(epoch_record)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            save_checkpoint(args.checkpoint_path, model, artist_to_index, best_val_loss, args)

    save_json(args.finetune_log_path, finetune_log)
    print(f"Saved training log to {args.finetune_log_path}")

    if best_state_dict is None:
        raise RuntimeError("Training completed without producing a best checkpoint.")

    model.load_state_dict(best_state_dict)
    model.eval()

    finetuned_artifacts = save_finetuned_embeddings(
        model=model,
        eval_loaders=eval_loaders,
        device=device,
        artist_to_index=artist_to_index,
        checkpoint_path=args.checkpoint_path,
    )

    train_embeddings = np.load(finetuned_artifacts["train_embeddings"])
    train_labels = np.load(finetuned_artifacts["train_labels"])
    fit_artist_profiles(
        embeddings=train_embeddings,
        labels=train_labels,
        artist_to_index=artist_to_index,
        output_path=args.finetuned_profile_path,
    )

    finetuned_report = evaluate_artifact_set(
        profiles_path=args.finetuned_profile_path,
        artist_map_path=finetuned_artifacts["artist_map"],
        train_embeddings_path=finetuned_artifacts["train_embeddings"],
        train_labels_path=finetuned_artifacts["train_labels"],
        train_metadata_path=finetuned_artifacts["train_metadata"],
        val_embeddings_path=finetuned_artifacts["val_embeddings"],
        val_labels_path=finetuned_artifacts["val_labels"],
        val_metadata_path=finetuned_artifacts["val_metadata"],
        test_embeddings_path=finetuned_artifacts["test_embeddings"],
        test_labels_path=finetuned_artifacts["test_labels"],
        test_metadata_path=finetuned_artifacts["test_metadata"],
        top_k=args.top_k,
    )
    save_json(args.finetuned_eval_path, finetuned_report)
    print_report_tables(finetuned_report)
    print(f"\nSaved fine-tuned evaluation report to {args.finetuned_eval_path}")

    if args.baseline_eval_path.exists():
        baseline_report = load_json(args.baseline_eval_path)
        print("\nBaseline vs Fine-Tuned")
        print(comparison_table(baseline_report, finetuned_report))
    else:
        print(f"\nBaseline report not found at {args.baseline_eval_path}; skipping comparison table.")

    print("\nSaved files")
    saved_files = [
        args.checkpoint_path,
        args.finetune_log_path,
        finetuned_artifacts["train_embeddings"],
        finetuned_artifacts["val_embeddings"],
        finetuned_artifacts["test_embeddings"],
        finetuned_artifacts["train_labels"],
        finetuned_artifacts["val_labels"],
        finetuned_artifacts["test_labels"],
        finetuned_artifacts["train_metadata"],
        finetuned_artifacts["val_metadata"],
        finetuned_artifacts["test_metadata"],
        finetuned_artifacts["artist_map"],
        args.finetuned_profile_path,
        args.finetuned_eval_path,
    ]
    for path in saved_files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
