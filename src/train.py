"""Simple, reproducible SwiftMask-inspired training pipeline."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from models.segmentation_cnn import SegmentationCNN
from src.dataset import DUTSDataset


class BCEDiceLoss(nn.Module):
    """Stable combination of pixel-wise BCE and soft Dice losses."""

    def __init__(self, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        if not 0.0 <= dice_weight <= 1.0:
            raise ValueError("dice_weight must be in [0, 1]")
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits).flatten(1)
        target = targets.flatten(1)
        intersection = (probs * target).sum(dim=1)
        denominator = probs.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return (1.0 - self.dice_weight) * bce + self.dice_weight * dice_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(data_root: Path, batch_size: int, workers: int, seed: int, limit: int | None):
    # Two dataset instances share exactly the same ordered samples but have
    # independent augmentation settings. This prevents augmenting validation.
    train_all = DUTSDataset(data_root, split="DUTS-TR", image_size=128, augment=True)
    valid_all = DUTSDataset(data_root, split="DUTS-TR", image_size=128, augment=False)

    if len(train_all) != len(valid_all):
        raise RuntimeError("Train/validation dataset sample lists do not match.")

    count = len(train_all) if limit is None else min(limit, len(train_all))
    if count < 2:
        raise ValueError("At least two DUTS-TR images are required.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(count, generator=generator).tolist()
    val_count = max(1, round(count * 0.10))
    train_indices = indices[val_count:]
    val_indices = indices[:val_count]
    if not train_indices:
        raise ValueError("Training split is empty; increase --limit.")

    train_set = Subset(train_all, train_indices)
    val_set = Subset(valid_all, val_indices)
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        common["prefetch_factor"] = 2

    train_loader = DataLoader(train_set, shuffle=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    return train_loader, val_loader


def segmentation_metrics(logits: torch.Tensor, masks: torch.Tensor) -> tuple[float, float, float]:
    predictions = torch.sigmoid(logits) >= 0.5
    targets = masks >= 0.5
    tp = (predictions & targets).sum().item()
    fp = (predictions & ~targets).sum().item()
    fn = (~predictions & targets).sum().item()
    tn = (~predictions & ~targets).sum().item()
    fg_iou = tp / max(tp + fp + fn, 1)
    bg_iou = tn / max(tn + fp + fn, 1)
    return (fg_iou + bg_iou) / 2.0, fg_iou, bg_iou


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False, phase="Train"):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    samples = 0
    total_tp = total_fp = total_fn = total_tn = 0

    progress = tqdm(loader, desc=phase, leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = criterion(logits, masks)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()

        with torch.no_grad():
            pred = torch.sigmoid(logits.float()) >= 0.5
            target = masks >= 0.5
            total_tp += (pred & target).sum().item()
            total_fp += (pred & ~target).sum().item()
            total_fn += (~pred & target).sum().item()
            total_tn += (~pred & ~target).sum().item()

        batch = images.size(0)
        total_loss += loss.item() * batch
        samples += batch
        progress.set_postfix(loss=f"{total_loss / max(samples, 1):.4f}")

    fg_iou = total_tp / max(total_tp + total_fp + total_fn, 1)
    bg_iou = total_tn / max(total_tn + total_fp + total_fn, 1)
    miou = (fg_iou + bg_iou) / 2.0
    return total_loss / max(samples, 1), miou, fg_iou, bg_iou


def save_history(history: list[dict], output_dir: Path) -> None:
    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    epochs = [r["epoch"] for r in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [r["train_loss"] for r in history], label="Train loss")
    ax.plot(epochs, [r["val_loss"] for r in history], label="Validation loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Training history")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [r["val_miou"] for r in history], label="Validation mIoU")
    ax.plot(epochs, [r["val_fg_iou"] for r in history], label="Foreground IoU")
    ax.set(xlabel="Epoch", ylabel="IoU", ylim=(0, 1), title="Validation segmentation quality")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "validation_iou.png", dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SwiftMask-inspired lightweight segmentation model.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/swiftmask_v2"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional subset for a smoke test.")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs/batch-size must be positive and workers non-negative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive and weight decay non-negative")
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be at least 2")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "model_best.pt"
    history_path = output_dir / "history.json"
    if history_path.exists() and not args.overwrite:
        raise FileExistsError(f"{history_path} already exists. Use a new --output-dir or --overwrite.")

    train_loader, val_loader = make_loaders(
        args.data, args.batch_size, args.workers, args.seed, args.limit
    )
    model = SegmentationCNN().to(device)
    criterion = BCEDiceLoss(dice_weight=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    config = {
        "architecture": "SwiftMask-inspired RSU encoder-decoder",
        "train_split": "DUTS-TR",
        "validation_split": "deterministic 10% subset of DUTS-TR",
        "test_split": "DUTS-TE (not used during training)",
        "input_size": 128,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "mixed_precision": amp,
        "loss": "0.5 BCEWithLogits + 0.5 SoftDice",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Train images: {len(train_loader.dataset)} | Validation images: {len(val_loader.dataset)}")
    print(f"Input: 128x128 | Batch: {args.batch_size} | AMP: {amp}")
    print(f"Outputs: {output_dir}")

    history = []
    best_miou = -1.0
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss, _, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, amp, "Train"
        )
        val_loss, val_miou, fg_iou, bg_iou = run_epoch(
            model, val_loader, criterion, device, None, scaler, amp, "Validation"
        )
        scheduler.step()
        duration = time.perf_counter() - started
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "val_fg_iou": fg_iou,
            "val_bg_iou": bg_iou,
            "learning_rate": lr,
            "epoch_seconds": duration,
        }
        history.append(row)
        save_history(history, output_dir)

        improved = val_miou > best_miou or (np.isclose(val_miou, best_miou) and val_loss < best_val_loss)
        if improved:
            best_miou = val_miou
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

        torch.save(model.state_dict(), output_dir / "model_latest.pt")
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"val_mIoU={val_miou:.4f} (FG={fg_iou:.4f}, BG={bg_iou:.4f}) | "
            f"lr={lr:.2e} time={duration:.1f}s"
            + (" | saved best" if improved else "")
        )

    summary = {
        "best_validation_miou": best_miou,
        "best_validation_loss": best_val_loss,
        "best_checkpoint": str(best_path),
        "epochs_completed": len(history),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Training complete. Best validation mIoU: {best_miou:.4f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
