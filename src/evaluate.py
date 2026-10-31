"""Final DUTS-TE evaluation for the SwiftMask-inspired model."""
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


def binary_metrics(confusion: np.ndarray) -> dict:
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    fg_iou = tp / max(tp + fp + fn, 1)
    bg_iou = tn / max(tn + fp + fn, 1)
    return {
        "foreground_iou": float(fg_iou),
        "background_iou": float(bg_iou),
        "miou": float((fg_iou + bg_iou) / 2.0),
    }


def boundary_f1(gt: np.ndarray, pred: np.ndarray, tolerance: int = 2) -> float:
    gt_u8 = gt.astype(np.uint8)
    pred_u8 = pred.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    gt_edge = cv2.morphologyEx(gt_u8, cv2.MORPH_GRADIENT, kernel)
    pred_edge = cv2.morphologyEx(pred_u8, cv2.MORPH_GRADIENT, kernel)
    if gt_edge.sum() == 0 and pred_edge.sum() == 0:
        return 1.0
    if gt_edge.sum() == 0 or pred_edge.sum() == 0:
        return 0.0
    match_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1)
    )
    gt_dilated = cv2.dilate(gt_edge, match_kernel)
    pred_dilated = cv2.dilate(pred_edge, match_kernel)
    precision = float((pred_edge & gt_dilated).sum()) / max(int(pred_edge.sum()), 1)
    recall = float((gt_edge & pred_dilated).sum()) / max(int(gt_edge.sum()), 1)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--split", default="DUTS-TE")
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/swiftmask_v2/model_best.pt"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)

    # Dataset stays at 256 for scoring; model sees 128, exactly as configured
    # in SwiftMask's reference evaluation protocol.
    dataset = DUTSDataset(args.data, split=args.split, image_size=256, augment=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0)
    )

    model = SegmentationCNN().to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    confusion = np.zeros((2, 2), dtype=np.int64)
    boundary_scores: list[float] = []
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    total_time = 0.0
    total_images = 0

    for images, masks in loader:
        images_device = images.to(device, non_blocking=True)
        model_images = F.interpolate(
            images_device, size=(128, 128), mode="bilinear", align_corners=False
        )

        # Exclude host-side data loading from model inference timing.
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        logits = model(model_images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.perf_counter() - started

        low_res_mask = torch.sigmoid(logits) >= 0.5
        pred = F.interpolate(
            low_res_mask.float(), size=(256, 256), mode="nearest"
        )[:, 0].cpu().numpy().astype(bool)
        target = (masks[:, 0].numpy() >= 0.5)

        for i in range(pred.shape[0]):
            p = pred[i]
            g = target[i]
            confusion[0, 0] += np.count_nonzero(~g & ~p)
            confusion[0, 1] += np.count_nonzero(~g & p)
            confusion[1, 0] += np.count_nonzero(g & ~p)
            confusion[1, 1] += np.count_nonzero(g & p)
            boundary_scores.append(boundary_f1(g, p))
            if len(examples) < args.examples:
                examples.append((
                    (images[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8),
                    g.astype(np.uint8),
                    p.astype(np.uint8),
                ))
            total_images += 1

    metrics = binary_metrics(confusion)
    metrics["boundary_f1"] = float(np.mean(boundary_scores)) if boundary_scores else 0.0
    metrics["num_images"] = total_images
    metrics["confusion_matrix"] = confusion.tolist()
    metrics["model_inference_ms_per_image"] = (total_time / max(total_images, 1)) * 1000.0
    metrics["checkpoint_bytes"] = args.checkpoint.stat().st_size
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["split"] = args.split
    metrics["input_resolution"] = 128

    output_dir = args.output_dir or args.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if examples:
        cols = 4
        rows = len(examples)
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), squeeze=False)
        for idx, (image, gt, pred) in enumerate(examples):
            axes[idx, 0].imshow(image)
            axes[idx, 0].set_title("Input")
            axes[idx, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
            axes[idx, 1].set_title("Ground truth")
            axes[idx, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
            axes[idx, 2].set_title("Prediction")
            overlay = image.copy()
            overlay[pred.astype(bool)] = (
                0.55 * overlay[pred.astype(bool)] + 0.45 * np.array([255, 0, 0])
            ).astype(np.uint8)
            axes[idx, 3].imshow(overlay)
            axes[idx, 3].set_title("Prediction overlay")
            for ax in axes[idx]:
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / "segmentation_examples.png", dpi=150)
        plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation to: {output_dir}")


if __name__ == "__main__":
    main()
