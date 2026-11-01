"""Train the wider hybrid residual U-Net on DUTS using a saliency structure loss.

The model architecture is intentionally not modified here. This script imports
SegmentationCNN from models.segmentation_cnn and expects one binary-logit map
per input image. DUTS-TR is split deterministically into train/validation sets;
DUTS-TE is never used for training or checkpoint selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

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

from models.segmentation_cnn_wide import SegmentationCNN, count_parameters
from src.dataset import DUTSDataset


class StructureLoss(nn.Module):
    """Boundary-weighted BCE + weighted IoU loss for salient-object masks.

    A local average-pooling map identifies pixels near object boundaries.
    Those pixels receive higher weight, while the weighted IoU term preserves
    object-level overlap. All reductions are computed in FP32 for AMP safety.
    """

    def __init__(self, kernel_size: int = 31, boundary_weight: float = 5.0) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if boundary_weight < 0:
            raise ValueError("boundary_weight must be non-negative.")
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.boundary_weight = boundary_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float().clamp_(0.0, 1.0)

        local_mean = F.avg_pool2d(
            targets,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            count_include_pad=False,
        )
        weight = 1.0 + self.boundary_weight * torch.abs(local_mean - targets)

        bce_map = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        weighted_bce = (weight * bce_map).sum(dim=(2, 3)) / weight.sum(
            dim=(2, 3)
        ).clamp_min(1e-6)

        probabilities = torch.sigmoid(logits)
        intersection = (weight * probabilities * targets).sum(dim=(2, 3))
        union = (weight * (probabilities + targets)).sum(dim=(2, 3))
        weighted_iou_loss = 1.0 - (intersection + 1.0) / (
            union - intersection + 1.0
        )

        return (weighted_bce + weighted_iou_loss).mean()


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = torch.cuda.is_available() and not deterministic


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loaders(
    data_root: Path,
    batch_size: int,
    workers: int,
    seed: int,
    image_size: int,
    val_fraction: float,
    limit: int | None = None,
) -> tuple[DataLoader, DataLoader, list[int], list[int]]:
    """Create deterministic train/validation subsets from DUTS-TR only."""
    train_all = DUTSDataset(
        data_root, split="DUTS-TR", image_size=image_size, augment=True
    )
    valid_all = DUTSDataset(
        data_root, split="DUTS-TR", image_size=image_size, augment=False
    )

    if len(train_all) != len(valid_all):
        raise RuntimeError("Training and validation dataset lengths differ.")
    if [p[0].name for p in train_all.samples] != [p[0].name for p in valid_all.samples]:
        raise RuntimeError("Training and validation sample ordering differs.")
    if len(train_all) < 2:
        raise ValueError("DUTS-TR must contain at least two images.")

    split_generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_all), generator=split_generator).tolist()
    if limit is not None:
        indices = indices[: min(limit, len(indices))]
    if len(indices) < 2:
        raise ValueError("At least two selected images are required.")

    val_count = max(1, round(len(indices) * val_fraction))
    val_count = min(val_count, len(indices) - 1)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    train_set = Subset(train_all, train_indices)
    val_set = Subset(valid_all, val_indices)
    loader_generator = torch.Generator().manual_seed(seed + 1)

    common: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        common["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=loader_generator,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, train_indices, val_indices


def confusion_counts(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    predictions = torch.sigmoid(logits.float()) >= threshold
    targets = masks >= 0.5
    return torch.stack(
        (
            (predictions & targets).sum(),
            (predictions & ~targets).sum(),
            (~predictions & targets).sum(),
            (~predictions & ~targets).sum(),
        )
    ).to(dtype=torch.int64)


def calculate_metrics(counts: list[int] | tuple[int, int, int, int]) -> dict[str, float]:
    tp, fp, fn, tn = (int(value) for value in counts)
    foreground_iou = tp / max(tp + fp + fn, 1)
    background_iou = tn / max(tn + fp + fn, 1)
    return {
        "miou": (foreground_iou + background_iou) / 2.0,
        "foreground_iou": foreground_iou,
        "background_iou": background_iou,
        "foreground_precision": tp / max(tp + fp, 1),
        "foreground_recall": tp / max(tp + fn, 1),
    }


def make_optimizer(model: nn.Module, learning_rate: float, weight_decay: float):
    """Do not apply weight decay to biases or normalization parameters."""
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    groups = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.999))


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    updates_per_epoch: int,
    epochs: int,
    warmup_epochs: int,
    min_lr: float,
    base_lr: float,
):
    """Linear warm-up followed by cosine decay, stepped after each update."""
    total_steps = max(1, updates_per_epoch * epochs)
    warmup_steps = max(0, updates_per_epoch * warmup_epochs)
    min_ratio = min_lr / base_lr

    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1.0 / warmup_steps)
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    channels_last: bool = False,
    accumulation_steps: int = 1,
    threshold: float = 0.5,
    epoch: int = 0,
    total_epochs: int = 0,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    counts = torch.zeros(4, dtype=torch.int64, device=device)

    if training:
        optimizer.zero_grad(set_to_none=True)

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:02d}/{total_epochs:02d} [{'Train' if training else 'Valid'}]",
        leave=False,
        dynamic_ncols=True,
    )

    for step, (images, masks) in enumerate(progress, start=1):
        batch_size = images.size(0)
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda").float()
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with torch.set_grad_enabled(training):
            with autocast_context:
                logits = model(images)
                if logits.shape != masks.shape:
                    raise RuntimeError(
                        f"Model output {tuple(logits.shape)} does not match "
                        f"mask shape {tuple(masks.shape)}."
                    )
                loss = criterion(logits, masks)

            if training:
                # Normalize accumulated gradients by the actual number of
                # samples in this group, including a short final batch.
                group_start = ((step - 1) // accumulation_steps) * accumulation_steps
                group_batch_count = min(accumulation_steps, len(loader) - group_start)
                group_sample_start = group_start * loader.batch_size
                group_sample_count = min(
                    group_sample_start + group_batch_count * loader.batch_size,
                    len(loader.dataset),
                ) - group_sample_start
                scaled_loss = loss * (batch_size / max(group_sample_count, 1))

                if use_amp:
                    if scaler is None:
                        raise RuntimeError("AMP is enabled but GradScaler is missing.")
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                update_step = step % accumulation_steps == 0 or step == len(loader)
                if update_step:
                    if use_amp:
                        assert scaler is not None
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    did_update = True
                    if use_amp:
                        assert scaler is not None
                        old_scale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        did_update = scaler.get_scale() >= old_scale
                    else:
                        optimizer.step()

                    optimizer.zero_grad(set_to_none=True)
                    if did_update and scheduler is not None:
                        scheduler.step()

        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
        counts += confusion_counts(logits.detach(), masks, threshold)

        if step % 20 == 0 or step == len(loader):
            progress.set_postfix(
                loss=f"{total_loss / max(total_samples, 1):.4f}",
                miou=f"{calculate_metrics(counts.cpu().tolist())['miou']:.4f}",
            )

    progress.close()
    if total_samples == 0:
        raise RuntimeError("DataLoader produced no samples.")
    return total_loss / total_samples, calculate_metrics(counts.cpu().tolist())


def save_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    if not history:
        return

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["train_loss"] for row in history], label="Train loss")
    ax.plot(epochs, [row["val_loss"] for row in history], label="Validation loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Structure-loss training")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["val_miou"] for row in history], label="Validation mIoU")
    ax.plot(epochs, [row["val_foreground_iou"] for row in history], label="Foreground IoU")
    ax.plot(epochs, [row["val_precision"] for row in history], label="Foreground precision")
    ax.plot(epochs, [row["val_recall"] for row in history], label="Foreground recall")
    ax.set(xlabel="Epoch", ylabel="Score", ylim=(0, 1), title="Validation metrics")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "validation_metrics.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the wider hybrid residual U-Net with a saliency structure loss."
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/hybrid_wide_256_v1"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=5e-4)
    parser.add_argument("--boundary-kernel", type=int, default=31)
    parser.add_argument("--boundary-weight", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers non-negative")
    if args.accumulation_steps < 1 or args.torch_threads < 1:
        parser.error("accumulation steps and torch threads must be positive")
    if args.learning_rate <= 0 or args.min_learning_rate < 0:
        parser.error("learning rate must be positive and minimum learning rate non-negative")
    if args.min_learning_rate > args.learning_rate or args.weight_decay < 0:
        parser.error("minimum learning rate or weight decay is invalid")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        parser.error("warmup epochs must be non-negative and smaller than total epochs")
    if args.patience < 1 or args.min_delta < 0:
        parser.error("patience must be positive and min-delta non-negative")
    if args.boundary_kernel < 1 or args.boundary_kernel % 2 == 0:
        parser.error("boundary-kernel must be a positive odd integer")
    if args.boundary_weight < 0 or not 0.0 < args.val_fraction < 1.0:
        parser.error("boundary-weight or validation fraction is invalid")
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be at least 2")
    if args.image_size < 32 or not 0.0 < args.threshold < 1.0:
        parser.error("image-size must be >= 32 and threshold must be between 0 and 1")
    if args.resume is not None and args.overwrite:
        parser.error("--resume and --overwrite cannot be used together")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    cv2.setNumThreads(1)
    torch.set_num_threads(args.torch_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    channels_last = device.type == "cuda"

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    state_path = output_dir / "training_state.pt"
    if history_path.exists() and args.resume is None and not args.overwrite:
        raise FileExistsError(
            f"{history_path} already exists. Choose another --output-dir, "
            "use --resume, or pass --overwrite for a fresh run."
        )

    train_loader, val_loader, train_indices, val_indices = make_loaders(
        args.data,
        args.batch_size,
        args.workers,
        args.seed,
        args.image_size,
        args.val_fraction,
        args.limit,
    )

    model = SegmentationCNN().to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = StructureLoss(
        kernel_size=args.boundary_kernel,
        boundary_weight=args.boundary_weight,
    ).to(device)
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
    scheduler = make_scheduler(
        optimizer,
        updates_per_epoch,
        args.epochs,
        args.warmup_epochs,
        args.min_learning_rate,
        args.learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = {
        "architecture": "wider hybrid residual U-Net (16, 24, 40, 56, 72 channels)",
        "training_split": "DUTS-TR",
        "validation_split": "deterministic held-out subset of DUTS-TR",
        "test_split": "DUTS-TE; never used for training or checkpoint selection",
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "effective_batch_size_approximately": args.batch_size * args.accumulation_steps,
        "learning_rate": args.learning_rate,
        "minimum_learning_rate": args.min_learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": "linear warm-up followed by cosine decay per optimizer update",
        "loss": "boundary-weighted BCE + weighted IoU structure loss",
        "boundary_kernel": args.boundary_kernel,
        "boundary_weight": args.boundary_weight,
        "optimizer": "AdamW; no decay on biases and 1D normalization parameters",
        "validation_fraction": args.val_fraction,
        "threshold": args.threshold,
        "seed": args.seed,
        "device": str(device),
        "mixed_precision": use_amp,
        "checkpoint_selection": "validation mIoU at the configured threshold; loss breaks ties",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    split_path = output_dir / "split_indices.json"
    (output_dir / "split_indices.json").write_text(
        json.dumps({"train_indices": train_indices, "validation_indices": val_indices}, indent=2),
        encoding="utf-8",
    )

    history: list[dict[str, Any]] = []
    best_miou = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1

    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        required = {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "epoch"}
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(f"Resume checkpoint is missing keys: {sorted(missing)}")
        saved_config = checkpoint.get("config", {})
        if saved_config and saved_config.get("image_size") != args.image_size:
            raise ValueError("Cannot resume with a different image size.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if use_amp and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history = checkpoint.get("history", [])
        best_miou = float(checkpoint.get("best_miou", -1.0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        start_epoch = int(checkpoint["epoch"]) + 1
        if checkpoint.get("train_indices") != train_indices or checkpoint.get("val_indices") != val_indices:
            raise ValueError("Saved training/validation split differs from the current split.")
        if checkpoint.get("loader_generator_state") is not None:
            train_loader.generator.set_state(checkpoint["loader_generator_state"])
        if start_epoch > args.epochs:
            raise ValueError("Checkpoint has already reached --epochs; increase --epochs to continue.")
        print(f"Resuming at epoch {start_epoch} from {args.resume}")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"AMP: {use_amp} | channels_last: {channels_last}")
    print(f"Train images: {len(train_loader.dataset)} | Validation images: {len(val_loader.dataset)}")
    print(f"Resolution: {args.image_size}x{args.image_size}")
    print(f"Batch size: {args.batch_size} x {args.accumulation_steps} accumulation steps")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Loss: boundary-weighted BCE + weighted IoU (kernel={args.boundary_kernel}, weight={args.boundary_weight})")
    print(f"Output: {output_dir}")

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            use_amp=use_amp,
            channels_last=channels_last,
            accumulation_steps=args.accumulation_steps,
            threshold=args.threshold,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            use_amp=use_amp,
            channels_last=channels_last,
            threshold=args.threshold,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        duration = time.perf_counter() - started

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_miou": train_metrics["miou"],
            "val_loss": val_loss,
            "val_miou": val_metrics["miou"],
            "val_foreground_iou": val_metrics["foreground_iou"],
            "val_background_iou": val_metrics["background_iou"],
            "val_precision": val_metrics["foreground_precision"],
            "val_recall": val_metrics["foreground_recall"],
            "learning_rate": current_lr,
            "epoch_seconds": duration,
        }
        history.append(row)
        save_history(history, output_dir)

        improved = (
            val_metrics["miou"] > best_miou + args.min_delta
            or (
                abs(val_metrics["miou"] - best_miou) <= args.min_delta
                and val_loss < best_val_loss
            )
        )
        if improved:
            best_miou = val_metrics["miou"]
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / "model_best.pt")
            (output_dir / "best_metrics.json").write_text(
                json.dumps(
                    {
                        "best_epoch": best_epoch,
                        "validation_miou": best_miou,
                        "validation_loss": best_val_loss,
                        "validation_foreground_iou": val_metrics["foreground_iou"],
                        "validation_precision": val_metrics["foreground_precision"],
                        "validation_recall": val_metrics["foreground_recall"],
                        "threshold": args.threshold,
                        "loss": "boundary-weighted BCE + weighted IoU",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            epochs_without_improvement += 1

        torch.save(model.state_dict(), output_dir / "model_latest.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "best_miou": best_miou,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "loader_generator_state": train_loader.generator.get_state(),
                "config": config,
            },
            state_path,
        )

        next_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_mIoU={train_metrics['miou']:.4f} | "
            f"val_loss={val_loss:.4f} val_mIoU={val_metrics['miou']:.4f} "
            f"(FG={val_metrics['foreground_iou']:.4f}, "
            f"BG={val_metrics['background_iou']:.4f}, "
            f"P={val_metrics['foreground_precision']:.4f}, "
            f"R={val_metrics['foreground_recall']:.4f}) | "
            f"lr={current_lr:.2e}->{next_lr:.2e} time={duration:.1f}s"
            + (" | saved best" if improved else "")
        )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping: validation mIoU did not improve by "
                f"{args.min_delta:g} for {args.patience} consecutive epochs."
            )
            break

    summary = {
        "best_validation_miou": best_miou,
        "best_validation_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_completed": history[-1]["epoch"] if history else 0,
        "trainable_parameters": count_parameters(model),
        "best_checkpoint": str(output_dir / "model_best.pt"),
        "resume_checkpoint": str(state_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nTraining complete.")
    print(f"Best validation mIoU: {best_miou:.4f} (epoch {best_epoch})")
    print(f"Best model: {output_dir / 'model_best.pt'}")
    print(f"Resume state: {state_path}")


if __name__ == "__main__":
    main()
