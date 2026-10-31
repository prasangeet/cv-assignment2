"""Final CPU benchmark of the SwiftMask-inspired model on DUTS-TE."""
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

from models.segmentation_cnn import SegmentationCNN
from src.dataset import DUTSDataset
from src.metrics import boundary_f1, miou_from_confusion

IMAGE_SIZE = 256
MAX_CPU_TIME_MS = 50.0
MAX_MODEL_SIZE_BYTES = 4 * 1024 * 1024
BOUNDARY_TOLERANCE = 2


def calculate_ces(miou: float, boundary: float, cpu_ms: float, size_bytes: int):
    """Assignment CES: quality weighted by CPU latency and model footprint."""
    quality = 0.70 * miou + 0.30 * boundary
    if cpu_ms > MAX_CPU_TIME_MS:
        return quality, 0.0
    time_penalty = 0.20 * cpu_ms / MAX_CPU_TIME_MS
    size_penalty = 0.10 * min(1.0, size_bytes / MAX_MODEL_SIZE_BYTES)
    return quality, quality * (1.0 - time_penalty - size_penalty)


@torch.inference_mode()
def predict_one(
    image_bgr: np.ndarray,
    model: SegmentationCNN,
    model_size: int,
    threshold: float,
) -> np.ndarray:
    """Full CPU inference path: BGR -> resize/normalize -> model -> binary mask."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(
        image_rgb,
        (model_size, model_size),
        interpolation=cv2.INTER_AREA,
    )
    image_rgb = image_rgb.astype(np.float32) / 255.0
    chw = np.ascontiguousarray(image_rgb.transpose(2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0)

    logits = model(tensor)
    prediction = torch.sigmoid(logits) >= threshold
    prediction = F.interpolate(
        prediction.float(),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="nearest",
    )
    return prediction[0, 0].cpu().numpy().astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation quality and CES on DUTS-TE."
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["DUTS-TE"], default="DUTS-TE")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/swiftmask_v2/model_best.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()
    if args.model_size < 32:
        parser.error("--model-size must be at least 32")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.examples < 0:
        parser.error("--examples cannot be negative")
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # CES is explicitly a CPU benchmark. Keep timing conditions reproducible.
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    dataset = DUTSDataset(
        args.data,
        split=args.split,
        image_size=IMAGE_SIZE,
        augment=False,
    )
    if args.split == "DUTS-TE" and len(dataset) != 5019:
        raise RuntimeError(
            f"Expected 5019 DUTS-TE image-mask pairs, found {len(dataset)}."
        )

    model = SegmentationCNN().to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Warmup is excluded from benchmark timing.
    first_image, _ = dataset[0]
    warmup_rgb = np.rint(
        first_image.permute(1, 2, 0).numpy() * 255.0
    ).clip(0, 255).astype(np.uint8)
    warmup_bgr = cv2.cvtColor(warmup_rgb, cv2.COLOR_RGB2BGR)
    _ = predict_one(warmup_bgr, model, args.model_size, args.threshold)

    confusion = np.zeros((2, 2), dtype=np.int64)
    boundary_scores: list[float] = []
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    total_cpu_seconds = 0.0

    print("=" * 68)
    print("SWIFTMASK — DUTS-TE CPU EVALUATION")
    print("=" * 68)
    print(f"Images: {len(dataset)} | Device: CPU | Threads: 1")
    print(f"Model input: {args.model_size}x{args.model_size}")
    print(f"Checkpoint: {args.checkpoint}")

    for index in range(len(dataset)):
        image_tensor, mask_tensor = dataset[index]

        # The dataset returns RGB floats at 256x256. Reconstruct the processed
        # uint8 image so that the measured path includes BGR->RGB conversion.
        image_rgb = np.rint(
            image_tensor.permute(1, 2, 0).numpy() * 255.0
        ).clip(0, 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        target = mask_tensor[0].numpy() >= 0.5

        started = time.process_time()
        prediction = predict_one(
            image_bgr, model, args.model_size, args.threshold
        )
        total_cpu_seconds += time.process_time() - started

        pred = prediction.astype(bool)
        confusion[0, 0] += np.count_nonzero(~target & ~pred)
        confusion[0, 1] += np.count_nonzero(~target & pred)
        confusion[1, 0] += np.count_nonzero(target & ~pred)
        confusion[1, 1] += np.count_nonzero(target & pred)
        boundary_scores.append(
            boundary_f1(pred, target, tolerance=BOUNDARY_TOLERANCE)
        )

        if len(examples) < args.examples:
            examples.append((image_rgb, target.astype(np.uint8), prediction))

        completed = index + 1
        if completed % 100 == 0 or completed == len(dataset):
            print(f"Evaluated {completed}/{len(dataset)}")

    iou = miou_from_confusion(confusion)
    # Calculate the two class IoUs explicitly for reporting.
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    foreground_iou = tp / max(tp + fp + fn, 1)
    background_iou = tn / max(tn + fp + fn, 1)
    mean_boundary = float(np.mean(boundary_scores))
    average_cpu_ms = total_cpu_seconds / max(len(dataset), 1) * 1000.0
    model_size_bytes = args.checkpoint.stat().st_size
    quality, ces = calculate_ces(
        iou, mean_boundary, average_cpu_ms, model_size_bytes
    )

    results = {
        "model": args.checkpoint.name,
        "split": args.split,
        "num_images": len(dataset),
        "device": "CPU",
        "cpu_threads": 1,
        "model_input_resolution": args.model_size,
        "output_resolution": IMAGE_SIZE,
        "threshold": args.threshold,
        "total_cpu_seconds": total_cpu_seconds,
        "average_cpu_ms": average_cpu_ms,
        "model_size_bytes": model_size_bytes,
        "foreground_iou": float(foreground_iou),
        "background_iou": float(background_iou),
        "miou": float(iou),
        "boundary_f1": mean_boundary,
        "quality_score": float(quality),
        "ces_score": float(ces),
        "confusion_matrix": confusion.tolist(),
        "checkpoint": str(args.checkpoint),
    }

    output_dir = args.output_dir or args.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
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

    print("\n" + "=" * 68)
    print("FINAL RESULTS")
    print("=" * 68)
    print(f"Foreground IoU: {foreground_iou:.6f}")
    print(f"Background IoU: {background_iou:.6f}")
    print(f"mIoU:           {iou:.6f}")
    print(f"Boundary F1:    {mean_boundary:.6f}")
    print(f"CPU ms/image:   {average_cpu_ms:.4f}")
    print(f"Model size:     {model_size_bytes} bytes")
    print(f"Quality Q:      {quality:.6f}")
    print(f"CES:            {ces:.6f}")
    print(f"Saved results and visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
