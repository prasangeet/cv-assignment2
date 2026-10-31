"""Reproducible training for the SwiftMask-inspired DUTS segmentation model."""
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
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from models.segmentation_cnn import SegmentationCNN, count_parameters
from src.dataset import DUTSDataset


class BCEDiceLoss(nn.Module):
    """Equal-weight BCE-with-logits and per-image soft Dice loss."""

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
        probabilities = torch.sigmoid(logits).flatten(1)
        targets = targets.flatten(1)
        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )
        dice_loss = 1.0 - dice.mean()
        return (1.0 - self.dice_weight) * bce + self.dice_weight * dice_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(
    data_root: Path,
    batch_size: int,
    workers: int,
    seed: int,
    limit: int | None = None,
    image_size: int = 128,
):
    # Separate instances ensure validation never receives random augmentation.
    train_all = DUTSDataset(
        data_root, split="DUTS-TR", image_size=image_size, augment=True
    )
    valid_all = DUTSDataset(
        data_root, split="DUTS-TR", image_size=image_size, augment=False
    )
    if [p[0].name for p in train_all.samples] != [p[0].name for p in valid_all.samples]:
        raise RuntimeError("Training and validation sample ordering differs.")

    count = len(train_all) if limit is None else min(limit, len(train_all))
    if count < 2:
        raise ValueError("At least two DUTS-TR images are required.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(count, generator=generator).tolist()
    val_count = max(1, round(count * 0.10))
    val_indices, train_indices = indices[:val_count], indices[val_count:]
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
    return (
        DataLoader(train_set, shuffle=True, **common),
        DataLoader(val_set, shuffle=False, **common),
    )


def _iou_counts(logits: torch.Tensor, masks: torch.Tensor):
    pred = torch.sigmoid(logits.float()) >= 0.5
    target = masks >= 0.5
    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()
    tn = (~pred & ~target).sum().item()
    return tp, fp, fn, tn


def _metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    fg = tp / max(tp + fp + fn, 1)
    bg = tn / max(tn + fp + fn, 1)
    return {
        "miou": (fg + bg) / 2.0,
        "foreground_iou": fg,
        "background_iou": bg,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    scaler=None,
    scheduler=None,
    use_amp: bool = False,
    channels_last: bool = False,
    epoch: int = 0,
    total_epochs: int = 0,
    log_interval: int = 25,
    accumulation_steps: int = 1,
    return_metrics: bool = False,
):
    """Run one epoch. Returns loss by default; optionally returns (loss, metrics).""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    tp = fp = fn = tn = 0
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True) if training else None

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} [{'Train' if training else 'Valid'}]",
        leave=False,
        dynamic_ncols=True,
    )
    for step, (images, masks) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, masks)

            if training:
                scaled_loss = loss / accumulation_steps
                if scaler is not None and use_amp:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                if step % accumulation_steps == 0 or step == len(loader):
                    if scaler is not None and use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    if scaler is not None and use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    if scheduler is not None:
                        scheduler.step()

        batch = images.size(0)
        total_loss += loss.detach().float().item() * batch
        total_samples += batch
        btp, bfp, bfn, btn = _iou_counts(logits.detach(), masks)
        tp += int(btp)
        fp += int(bfp)
        fn += int(bfn)
        tn += int(btn)
        progress.set_postfix(
            loss=f"{total_loss / max(total_samples, 1):.4f}",
            miou=f"{_metrics(tp, fp, fn, tn)['miou']:.4f}",
        )

    progress.close()
    if not total_samples:
        raise RuntimeError("DataLoader produced no samples.")
    result = _metrics(tp, fp, fn, tn)
    epoch_loss = total_loss / total_samples
    return (epoch_loss, result) if return_metrics else epoch_loss


def save_history(history: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    if history:
        with (output_dir / "history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

        epochs = [row["epoch"] for row in history]
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
        ax.plot(epochs, [r["val_foreground_iou"] for r in history], label="Foreground IoU")
        ax.set(xlabel="Epoch", ylabel="IoU", ylim=(0, 1), title="Validation quality")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "validation_iou.png", dpi=150)
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the lightweight SwiftMask-inspired segmentation model."
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/swiftmask_v2"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers non-negative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive; weight decay non-negative")
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be at least 2")
    if args.image_size < 32:
        parser.error("--image-size must be at least 32")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    if history_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{history_path} exists. Choose a new --output-dir or pass --overwrite."
        )

    train_loader, val_loader = make_loaders(
        args.data, args.batch_size, args.workers, args.seed,
        args.limit, args.image_size,
    )
    model = SegmentationCNN().to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = {
        "architecture": "SwiftMask-inspired RSU encoder-decoder",
        "train_split": "DUTS-TR",
        "validation_split": "deterministic 10% of DUTS-TR",
        "test_split": "DUTS-TE; never used for training or model selection",
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "mixed_precision": use_amp,
        "loss": "0.5 BCEWithLogits + 0.5 SoftDice",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Train images: {len(train_loader.dataset)} | "
        f"Validation images: {len(val_loader.dataset)}"
    )
    print(f"Resolution: {args.image_size}x{args.image_size} | AMP: {use_amp}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Experiment output: {output_dir}")

    history = []
    best_miou = -1.0
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, scaler=scaler,
            use_amp=use_amp, channels_last=device.type == "cuda",
            epoch=epoch, total_epochs=args.epochs, return_metrics=True,
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion, device,
            use_amp=use_amp, channels_last=device.type == "cuda",
            epoch=epoch, total_epochs=args.epochs, return_metrics=True,
        )
        duration = time.perf_counter() - started
        scheduler_lr_before = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_miou": train_metrics["miou"],
            "val_loss": val_loss,
            "val_miou": val_metrics["miou"],
            "val_foreground_iou": val_metrics["foreground_iou"],
            "val_background_iou": val_metrics["background_iou"],
            "learning_rate": scheduler_lr_before,
            "epoch_seconds": duration,
        }
        history.append(row)
        save_history(history, output_dir)

        improved = (
            val_metrics["miou"] > best_miou
            or (
                np.isclose(val_metrics["miou"], best_miou)
                and val_loss < best_val_loss
            )
        )
        if improved:
            best_miou = val_metrics["miou"]
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "model_best.pt")
        torch.save(model.state_dict(), output_dir / "model_latest.pt")
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_mIoU={train_metrics['miou']:.4f} | "
            f"val_loss={val_loss:.4f} val_mIoU={val_metrics['miou']:.4f} "
            f"(FG={val_metrics['foreground_iou']:.4f}, "
            f"BG={val_metrics['background_iou']:.4f}) | "
            f"lr={scheduler_lr_before:.2e} time={duration:.1f}s"
            + (" | saved best" if improved else "")
        )

    summary = {
        "best_validation_miou": best_miou,
        "best_validation_loss": best_val_loss,
        "best_checkpoint": str(output_dir / "model_best.pt"),
        "epochs_completed": len(history),
        "trainable_parameters": count_parameters(model),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Training complete. Best validation mIoU: {best_miou:.4f}")
    print(f"Best checkpoint: {output_dir / 'model_best.pt'}")


if __name__ == "__main__":
    main()
