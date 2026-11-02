"""Train the residual multi-scale U-Net at 128x128 on DUTS.

Training upgrades over the plain baseline:
- Same deterministic, foreground-area-stratified DUTS-TR split.
- Deep supervision at quarter- and half-resolution decoder stages.
- BCE + soft Dice for every supervised output.
- AdamW with no weight decay on bias/normalization parameters.
- Linear warm-up followed by cosine decay.
- AMP, channels-last, correct sample-weighted gradient accumulation, clipping.
- Selects checkpoints using validation quality Q = 0.7*mIoU + 0.3*Boundary-F1.
- DUTS-TE is never touched by this script.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

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
from src.metrics import boundary_f1


class BCEDiceLoss(nn.Module):
    """BCE-with-logits plus per-image soft Dice."""

    def __init__(self, dice_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        if not 0.0 <= dice_weight <= 1.0 or smooth <= 0:
            raise ValueError("Invalid Dice loss configuration.")
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        bce = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits).flatten(1)
        targets_flat = targets.flatten(1)
        intersection = (probabilities * targets_flat).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return (1.0 - self.dice_weight) * bce + self.dice_weight * dice_loss


class DeepSupervisionLoss(nn.Module):
    """Main output plus auxiliary decoder supervision.

    Output order: main, quarter-resolution decoder head, half-resolution head.
    Weights sum to one so the overall loss scale remains comparable to BCE+Dice.
    """

    def __init__(self) -> None:
        super().__init__()
        self.base = BCEDiceLoss(dice_weight=0.5)
        self.main_weight = 0.60
        self.quarter_weight = 0.15
        self.half_weight = 0.25

    def forward(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        main, aux_quarter, aux_half = outputs
        return (
            self.main_weight * self.base(main, targets)
            + self.quarter_weight * self.base(aux_quarter, targets)
            + self.half_weight * self.base(aux_half, targets)
        )


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = torch.cuda.is_available() and not deterministic


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def foreground_ratios(samples: list[tuple[Path, Path]]) -> np.ndarray:
    ratios = np.empty(len(samples), dtype=np.float64)
    for index, (_, mask_path) in enumerate(samples):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask while creating split: {mask_path}")
        ratios[index] = float(np.mean(mask >= 128))
    return ratios


def stratified_split(
    samples: list[tuple[Path, Path]],
    val_fraction: float,
    seed: int,
    bins: int,
) -> tuple[list[int], list[int], np.ndarray]:
    """Deterministically stratify by quantiles of foreground area."""
    ratios = foreground_ratios(samples)
    n = len(samples)
    if n < 2:
        raise ValueError("DUTS-TR must contain at least two images.")
    val_total = max(1, min(n - 1, round(n * val_fraction)))
    rng = np.random.default_rng(seed)
    strata = np.array_split(np.argsort(ratios, kind="stable"), min(bins, n))
    raw = np.array([len(group) * val_fraction for group in strata], dtype=float)
    allocations = np.floor(raw).astype(int)
    remaining = val_total - int(allocations.sum())
    for stratum_id in np.argsort(-(raw - allocations)):
        if remaining <= 0:
            break
        if allocations[stratum_id] < len(strata[stratum_id]):
            allocations[stratum_id] += 1
            remaining -= 1

    train_indices: list[int] = []
    val_indices: list[int] = []
    for group, count in zip(strata, allocations):
        shuffled = group.copy()
        rng.shuffle(shuffled)
        val_indices.extend(int(i) for i in shuffled[:count])
        train_indices.extend(int(i) for i in shuffled[count:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    if set(train_indices) & set(val_indices):
        raise RuntimeError("Train/validation leakage detected.")
    if len(train_indices) + len(val_indices) != n or len(val_indices) != val_total:
        raise RuntimeError("Unexpected train/validation split size.")
    return train_indices, val_indices, ratios


def make_loaders(args: argparse.Namespace):
    train_all = DUTSDataset(
        args.data, split="DUTS-TR", image_size=128, augment=True
    )
    valid_all = DUTSDataset(
        args.data, split="DUTS-TR", image_size=128, augment=False
    )
    if len(train_all) != len(valid_all):
        raise RuntimeError("Train and validation dataset lengths differ.")
    if [p[0].name for p in train_all.samples] != [p[0].name for p in valid_all.samples]:
        raise RuntimeError("Train and validation sample ordering differs.")

    if args.limit is not None:
        rng = np.random.default_rng(args.seed)
        selected = sorted(int(i) for i in rng.permutation(len(train_all))[:args.limit])
        selected_samples = [train_all.samples[i] for i in selected]
        local_train, local_val, selected_ratios = stratified_split(
            selected_samples, args.val_fraction, args.seed + 1, args.split_bins
        )
        train_indices = [selected[i] for i in local_train]
        val_indices = [selected[i] for i in local_val]
        ratios = np.zeros(len(train_all), dtype=np.float64)
        for original, ratio in zip(selected, selected_ratios):
            ratios[original] = ratio
    else:
        train_indices, val_indices, ratios = stratified_split(
            train_all.samples, args.val_fraction, args.seed, args.split_bins
        )

    if set(train_indices) & set(val_indices):
        raise RuntimeError("Train and validation indices overlap.")
    train_set = Subset(train_all, train_indices)
    val_set = Subset(valid_all, val_indices)
    generator = torch.Generator().manual_seed(args.seed + 17)
    common: dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
        "worker_init_fn": seed_worker,
    }
    if args.workers > 0:
        common["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_set, shuffle=True, generator=generator, drop_last=False, **common
    )
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, train_indices, val_indices, ratios, train_all.samples


def build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.999),
    )


def learning_rate_for_epoch(
    epoch: int, epochs: int, base_lr: float, min_lr: float, warmup_epochs: int
) -> float:
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    decay_length = max(epochs - warmup_epochs, 1)
    progress = min(max((epoch - warmup_epochs) / decay_length, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def counts_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float):
    pred = torch.sigmoid(logits.float()) >= threshold
    truth = targets >= 0.5
    return torch.stack(
        (
            (pred & truth).sum(),
            (pred & ~truth).sum(),
            (~pred & truth).sum(),
            (~pred & ~truth).sum(),
        )
    ).to(torch.int64)


def metrics_from_counts(counts: torch.Tensor) -> dict[str, float]:
    tp, fp, fn, tn = (int(v) for v in counts.detach().cpu().tolist())
    fg = tp / max(tp + fp + fn, 1)
    bg = tn / max(tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "miou": (fg + bg) / 2.0,
        "foreground_iou": fg,
        "background_iou": bg,
        "precision": precision,
        "recall": recall,
    }


def run_epoch(
    model: SegmentationCNN,
    loader: DataLoader,
    criterion: DeepSupervisionLoss,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    channels_last: bool,
    accumulation_steps: int,
    threshold: float,
    epoch: int,
    epochs: int,
    calculate_boundary: bool,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    counts = torch.zeros(4, dtype=torch.int64, device=device)
    boundary_scores: list[float] = []
    if training:
        optimizer.zero_grad(set_to_none=True)

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:02d}/{epochs:02d} [{'Train' if training else 'Valid'}]",
        leave=False,
        dynamic_ncols=True,
    )
    for step, (images, masks) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda").float()
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(images, return_aux=True)
                if not isinstance(outputs, tuple) or len(outputs) != 3:
                    raise RuntimeError("Model must return (main, aux_quarter, aux_half) with return_aux=True.")
                main_logits, aux_quarter, aux_half = outputs
                if main_logits.shape != masks.shape:
                    raise RuntimeError(
                        f"Logits shape {tuple(main_logits.shape)} does not match masks {tuple(masks.shape)}."
                    )
                loss = criterion(outputs, masks)

            if training:
                group_start = ((step - 1) // accumulation_steps) * accumulation_steps
                group_end = min(group_start + accumulation_steps, len(loader))
                first_sample = group_start * loader.batch_size
                group_samples = min(
                    (group_end - group_start) * loader.batch_size,
                    len(loader.dataset) - first_sample,
                )
                scaled_loss = loss * (images.size(0) / max(group_samples, 1))
                if use_amp:
                    if scaler is None:
                        raise RuntimeError("AMP enabled without a GradScaler.")
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                update_step = step % accumulation_steps == 0 or step == len(loader)
                if update_step:
                    if use_amp:
                        assert scaler is not None
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if use_amp:
                        assert scaler is not None
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        batch_size = images.size(0)
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
        counts += counts_from_logits(main_logits.detach(), masks, threshold)

        if calculate_boundary:
            probabilities = torch.sigmoid(main_logits.detach().float()) >= threshold
            predictions_np = probabilities[:, 0].cpu().numpy()
            targets_np = (masks[:, 0] >= 0.5).cpu().numpy()
            for pred, target in zip(predictions_np, targets_np):
                boundary_scores.append(float(boundary_f1(pred, target, tolerance=2)))

        if step % 20 == 0 or step == len(loader):
            current_metrics = metrics_from_counts(counts)
            progress.set_postfix(
                loss=f"{total_loss / max(total_samples, 1):.4f}",
                miou=f"{current_metrics['miou']:.4f}",
            )

    progress.close()
    if total_samples == 0:
        raise RuntimeError("DataLoader produced no samples.")
    metrics = metrics_from_counts(counts)
    if calculate_boundary:
        metrics["boundary_f1"] = float(np.mean(boundary_scores)) if boundary_scores else 0.0
        metrics["quality"] = 0.70 * metrics["miou"] + 0.30 * metrics["boundary_f1"]
    return total_loss / total_samples, metrics


def save_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
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
    ax.set(xlabel="Epoch", ylabel="Loss", title="Residual multi-scale U-Net training")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in (
        ("val_miou", "Validation mIoU"),
        ("val_foreground_iou", "Foreground IoU"),
        ("val_boundary_f1", "Boundary F1"),
        ("val_quality", "Validation Q"),
    ):
        ax.plot(epochs, [row[key] for row in history], label=label)
    ax.set(xlabel="Epoch", ylabel="Score", ylim=(0, 1), title="Validation quality")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "validation_metrics.png", dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train residual multi-scale U-Net at 128x128.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/unet_128_multiscale_v1"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--split-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-size", type=int, choices=[128], default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    if args.epochs < 2 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs must be >= 2, batch size positive, workers non-negative")
    if args.accumulation_steps < 1 or args.torch_threads < 1:
        parser.error("accumulation steps and torch threads must be positive")
    if args.learning_rate <= 0 or not 0 <= args.min_learning_rate <= args.learning_rate:
        parser.error("invalid learning-rate values")
    if args.weight_decay < 0 or not 0 <= args.warmup_epochs < args.epochs:
        parser.error("invalid weight-decay or warm-up values")
    if args.patience < 1 or not 0 < args.val_fraction < 1 or args.split_bins < 2:
        parser.error("invalid patience, validation fraction, or split bins")
    if args.limit is not None and args.limit < 2:
        parser.error("--limit must be at least 2")
    if not 0 < args.threshold < 1:
        parser.error("threshold must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed, args.deterministic)
    cv2.setNumThreads(1)
    torch.set_num_threads(args.torch_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    channels_last = device.type == "cuda"
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "history.json").exists():
        raise FileExistsError(f"{out_dir / 'history.json'} exists; choose a fresh --output-dir.")

    train_loader, val_loader, train_indices, val_indices, ratios, samples = make_loaders(args)
    split = {
        "seed": args.seed,
        "image_size": 128,
        "train_indices": train_indices,
        "validation_indices": val_indices,
        "train_filenames": [samples[i][0].name for i in train_indices],
        "validation_filenames": [samples[i][0].name for i in val_indices],
        "train_foreground_ratio_mean": float(np.mean([ratios[i] for i in train_indices])),
        "validation_foreground_ratio_mean": float(np.mean([ratios[i] for i in val_indices])),
    }
    (out_dir / "split_indices.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    model = SegmentationCNN().to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    criterion = DeepSupervisionLoss().to(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = {
        "architecture": "three-level residual U-Net with dilated multi-scale bottleneck and deep supervision",
        "input_resolution": 128,
        "train_split": "DUTS-TR",
        "validation_split": "deterministic foreground-area-stratified 10% subset of DUTS-TR",
        "test_split": "DUTS-TE; not used for training or checkpoint selection",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "effective_batch_size_approximately": args.batch_size * args.accumulation_steps,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "loss": "0.60 main BCE-Dice + 0.15 quarter aux BCE-Dice + 0.25 half aux BCE-Dice",
        "checkpoint_metric": "validation Q = 0.70*mIoU + 0.30*BoundaryF1",
        "seed": args.seed,
        "device": str(device),
        "mixed_precision": use_amp,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("=" * 68)
    print("128x128 RESIDUAL MULTI-SCALE U-NET")
    print("=" * 68)
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"AMP: {use_amp} | channels_last: {channels_last}")
    print(f"Train images: {len(train_loader.dataset)} | Validation images: {len(val_loader.dataset)}")
    print(f"Resolution: 128x128 | Parameters: {count_parameters(model):,}")
    print(f"Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"Loss: {config['loss']}")
    print("Checkpoint selection: validation Q (0.7 mIoU + 0.3 Boundary F1)")
    print(f"Output: {out_dir}")

    history: list[dict[str, Any]] = []
    best_quality = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        lr = learning_rate_for_epoch(
            epoch, args.epochs, args.learning_rate, args.min_learning_rate, args.warmup_epochs
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        started = time.perf_counter()

        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, scaler=scaler, use_amp=use_amp,
            channels_last=channels_last, accumulation_steps=args.accumulation_steps,
            threshold=args.threshold, epoch=epoch, epochs=args.epochs,
            calculate_boundary=False,
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion, device,
            optimizer=None, scaler=None, use_amp=use_amp,
            channels_last=channels_last, accumulation_steps=1,
            threshold=args.threshold, epoch=epoch, epochs=args.epochs,
            calculate_boundary=True,
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
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_boundary_f1": val_metrics["boundary_f1"],
            "val_quality": val_metrics["quality"],
            "learning_rate": lr,
            "epoch_seconds": duration,
        }
        history.append(row)
        save_history(history, out_dir)

        improved = (
            val_metrics["quality"] > best_quality + 1e-6
            or (abs(val_metrics["quality"] - best_quality) <= 1e-6 and val_loss < best_val_loss)
        )
        if improved:
            best_quality = val_metrics["quality"]
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), out_dir / "model_best.pt")
            (out_dir / "best_metrics.json").write_text(
                json.dumps({"best_epoch": best_epoch, "validation_quality": best_quality,
                            "validation_loss": best_val_loss, **val_metrics}, indent=2),
                encoding="utf-8",
            )
        else:
            stale_epochs += 1

        torch.save(model.state_dict(), out_dir / "model_latest.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "best_quality": best_quality,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "history": history,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "config": config,
            },
            out_dir / "training_state.pt",
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} "
            f"train_mIoU={train_metrics['miou']:.4f} | val_loss={val_loss:.4f} "
            f"val_mIoU={val_metrics['miou']:.4f} "
            f"(FG={val_metrics['foreground_iou']:.4f}, BG={val_metrics['background_iou']:.4f}, "
            f"P={val_metrics['precision']:.4f}, R={val_metrics['recall']:.4f}, "
            f"BF1={val_metrics['boundary_f1']:.4f}, Q={val_metrics['quality']:.4f}) | "
            f"lr={lr:.2e} time={duration:.1f}s"
            + (" | saved best" if improved else "")
        )

        if stale_epochs >= args.patience:
            print(f"Early stopping: validation Q did not improve for {args.patience} epochs.")
            break

    summary = {
        "best_validation_quality": best_quality,
        "best_validation_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_completed": history[-1]["epoch"] if history else 0,
        "trainable_parameters": count_parameters(model),
        "best_checkpoint": str(out_dir / "model_best.pt"),
        "resume_checkpoint": str(out_dir / "training_state.pt"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nTraining complete.")
    print(f"Best validation Q: {best_quality:.4f} (epoch {best_epoch})")
    print(f"Best checkpoint: {out_dir / 'model_best.pt'}")


if __name__ == "__main__":
    main()
