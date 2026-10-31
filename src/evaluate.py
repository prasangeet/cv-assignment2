"""Evaluate the trained model on DUTS-TE and save metrics/visualizations."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.segmentation_cnn import SegmentationCNN
from src.dataset import DUTSDataset
from src.metrics import boundary_f1 as boundary_f1_score


def binary_metrics(confusion: np.ndarray) -> dict[str, float]:
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    fg_union = tp + fp + fn
    bg_union = tn + fp + fn
    fg = tp / fg_union if fg_union else 0.0
    bg = tn / bg_union if bg_union else 0.0
    return {
        "foreground_iou": float(fg),
        "background_iou": float(bg),
        "miou": float((fg + bg) / 2.0),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Evaluate a SwiftMask checkpoint.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["DUTS-TE"], default="DUTS-TE")
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/swiftmask_v2/model_best.pt"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model-size", type=int, default=128)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch size must be positive and workers non-negative")

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DUTSDataset(
        args.data, split=args.split, image_size=256, augment=False
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = SegmentationCNN().to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    confusion = np.zeros((2, 2), dtype=np.int64)
    boundary_scores: list[float] = []
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    total_inference_seconds = 0.0
    num_images = 0

    for images, masks in loader:
        images_device = images.to(device, non_blocking=device.type == "cuda")
        model_images = F.interpolate(
            images_device,
            size=(args.model_size, args.model_size),
            mode="bilinear",
            align_corners=False,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        logits = model(model_images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_inference_seconds += time.perf_counter() - started

        probabilities = torch.sigmoid(logits)
        prediction = probabilities >= args.threshold
        prediction = F.interpolate(
            prediction.float(), size=(256, 256), mode="nearest"
        )[:, 0].cpu().numpy().astype(bool)
        target = masks[:, 0].numpy() >= 0.5

        for idx in range(len(prediction)):
            pred, gt = prediction[idx], target[idx]
            confusion[0, 0] += np.count_nonzero(~gt & ~pred)
            confusion[0, 1] += np.count_nonzero(~gt & pred)
            confusion[1, 0] += np.count_nonzero(gt & ~pred)
            confusion[1, 1] += np.count_nonzero(gt & pred)
            boundary_scores.append(boundary_f1_score(pred, gt, tolerance=2))
            if len(examples) < args.examples:
                rgb = (
                    images[idx].permute(1, 2, 0).numpy() * 255.0
                ).clip(0, 255).astype(np.uint8)
                examples.append((rgb, gt.astype(np.uint8), pred.astype(np.uint8)))
            num_images += 1

    metrics = binary_metrics(confusion)
    metrics.update({
        "boundary_f1": float(np.mean(boundary_scores)),
        "num_images": num_images,
        "confusion_matrix": confusion.tolist(),
        "model_inference_ms_per_image": (
            total_inference_seconds / max(num_images, 1) * 1000.0
        ),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "model_input_resolution": args.model_size,
        "threshold": args.threshold,
    })

    output_dir = args.output_dir or args.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    if examples:
        fig, axes = plt.subplots(
            len(examples), 4,
            figsize=(12, 3.0 * len(examples)),
            squeeze=False,
        )
        for row, (rgb, gt, pred) in enumerate(examples):
            axes[row, 0].imshow(rgb)
            axes[row, 0].set_title("Input")
            axes[row, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
            axes[row, 1].set_title("Ground truth")
            axes[row, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
            axes[row, 2].set_title("Prediction")
            overlay = rgb.copy()
            selected = pred.astype(bool)
            overlay[selected] = (
                0.55 * overlay[selected] + 0.45 * np.array([255, 0, 0])
            ).astype(np.uint8)
            axes[row, 3].imshow(overlay)
            axes[row, 3].set_title("Overlay")
            for axis in axes[row]:
                axis.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / "segmentation_examples.png", dpi=150)
        plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"Evaluation artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
