"""Evaluate the current SegmentationCNN on DUTS-TE."""
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

from models.segmentation_cnn import SegmentationCNN
from src.dataset import DUTSDataset
from src.metrics import boundary_f1

MAX_CPU_TIME_MS = 50.0
MAX_MODEL_SIZE_BYTES = 4 * 1024 * 1024
BOUNDARY_TOLERANCE = 2


def calculate_ces(
    miou: float,
    boundary: float,
    cpu_ms: float,
    size_bytes: int,
) -> tuple[float, float]:
    """Calculate the assignment's quality and compute-efficiency scores."""
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
    image_size: int,
    threshold: float,
) -> np.ndarray:
    """Predict one binary mask using the same resize/normalization as DUTSDataset."""
    if image_bgr is None:
        raise ValueError("Input image is None.")

    image = cv2.resize(
        image_bgr,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
    tensor = torch.from_numpy(image).unsqueeze(0)

    logits = model(tensor)
    if logits.shape[-2:] != (image_size, image_size):
        raise RuntimeError(
            f"Model returned spatial size {tuple(logits.shape[-2:])}; "
            f"expected {(image_size, image_size)}."
        )
    return (torch.sigmoid(logits)[0, 0] >= threshold).cpu().numpy().astype(np.uint8)


def metrics_from_confusion(
    confusion: np.ndarray,
    boundary_scores: list[float],
) -> dict[str, float]:
    """Compute dataset-level foreground/background IoU and mean Boundary F1."""
    tn, fp = (int(x) for x in confusion[0])
    fn, tp = (int(x) for x in confusion[1])

    foreground_iou = tp / max(tp + fp + fn, 1)
    background_iou = tn / max(tn + fp + fn, 1)
    return {
        "foreground_iou": float(foreground_iou),
        "background_iou": float(background_iou),
        "miou": float((foreground_iou + background_iou) / 2.0),
        "boundary_f1": float(np.mean(boundary_scores)) if boundary_scores else 0.0,
    }


def save_visualizations(
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    if not examples:
        return

    fig, axes = plt.subplots(
        len(examples), 4,
        figsize=(12, 3.0 * len(examples)),
        squeeze=False,
    )
    for row, (image, target, prediction) in enumerate(examples):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title("Input")
        axes[row, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("Prediction")

        overlay = image.copy()
        selected = prediction.astype(bool)
        overlay[selected] = (
            0.55 * overlay[selected] + 0.45 * np.array([255, 0, 0])
        ).astype(np.uint8)
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title("Prediction overlay")
        for axis in axes[row]:
            axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "segmentation_examples.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the current SegmentationCNN on DUTS-TE."
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["DUTS-TE"], default="DUTS-TE")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    if args.image_size < 8:
        parser.error("--image-size must be at least 8")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.examples < 0:
        parser.error("--examples cannot be negative")
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    cv2.setNumThreads(1)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")  # Required for comparable CPU/CES benchmarking.

    dataset = DUTSDataset(
        root=args.data,
        split=args.split,
        image_size=args.image_size,
        augment=False,
    )
    if len(dataset) != 5019:
        raise RuntimeError(f"Expected 5019 DUTS-TE samples, found {len(dataset)}.")

    model = SegmentationCNN().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    # Training saves a plain state_dict so it can be loaded directly here.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    # Warm up CPU kernels before measuring inference time.
    first_image_path, _ = dataset.samples[0]
    warmup_image = cv2.imread(str(first_image_path), cv2.IMREAD_COLOR)
    if warmup_image is None:
        raise RuntimeError(f"Could not read warmup image: {first_image_path}")
    predict_one(warmup_image, model, args.image_size, args.threshold)

    confusion = np.zeros((2, 2), dtype=np.int64)  # [[TN, FP], [FN, TP]]
    boundary_scores: list[float] = []
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    total_cpu_seconds = 0.0

    print("=" * 68)
    print("SEGMENTATION MODEL — DUTS-TE EVALUATION")
    print("=" * 68)
    print(f"Images: {len(dataset)}")
    print(f"Device: {device} | CPU threads: {torch.get_num_threads()}")
    print(f"Input / prediction / metric resolution: {args.image_size}x{args.image_size}")
    print(f"Threshold: {args.threshold}")
    print(f"Checkpoint: {args.checkpoint}")

    for index, (image_path, mask_path) in enumerate(dataset.samples, start=1):
        # File I/O is outside the inference timer.
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        target = cv2.resize(
            mask,
            (args.image_size, args.image_size),
            interpolation=cv2.INTER_NEAREST,
        )
        target = (target >= 128).astype(np.uint8)

        started = time.process_time()
        prediction = predict_one(
            image_bgr, model, args.image_size, args.threshold
        )
        total_cpu_seconds += time.process_time() - started

        if prediction.shape != target.shape:
            raise RuntimeError(
                f"Prediction {prediction.shape} and target {target.shape} differ."
            )

        pred_bool = prediction.astype(bool)
        target_bool = target.astype(bool)
        confusion[0, 0] += np.count_nonzero(~target_bool & ~pred_bool)
        confusion[0, 1] += np.count_nonzero(~target_bool & pred_bool)
        confusion[1, 0] += np.count_nonzero(target_bool & ~pred_bool)
        confusion[1, 1] += np.count_nonzero(target_bool & pred_bool)

        boundary_scores.append(
            float(boundary_f1(
                pred_bool,
                target_bool,
                tolerance=BOUNDARY_TOLERANCE,
            ))
        )

        if len(examples) < args.examples:
            image_rgb = cv2.cvtColor(
                cv2.resize(
                    image_bgr,
                    (args.image_size, args.image_size),
                    interpolation=cv2.INTER_AREA,
                ),
                cv2.COLOR_BGR2RGB,
            )
            examples.append((image_rgb, target, prediction))

        if index % 100 == 0 or index == len(dataset):
            print(f"Evaluated {index}/{len(dataset)}")

    metrics = metrics_from_confusion(confusion, boundary_scores)
    average_cpu_ms = total_cpu_seconds * 1000.0 / len(dataset)
    model_size_bytes = args.checkpoint.stat().st_size
    quality, ces = calculate_ces(
        metrics["miou"], metrics["boundary_f1"], average_cpu_ms, model_size_bytes
    )

    output_dir = args.output_dir or args.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "model": args.checkpoint.name,
        "split": args.split,
        "num_images": len(dataset),
        "device": "CPU",
        "cpu_threads": args.threads,
        "input_resolution": args.image_size,
        "ground_truth_resolution": args.image_size,
        "prediction_resolution": args.image_size,
        "threshold": args.threshold,
        "total_cpu_seconds": total_cpu_seconds,
        "average_cpu_ms": average_cpu_ms,
        "model_size_bytes": model_size_bytes,
        **metrics,
        "quality_score": quality,
        "ces_score": ces,
        "confusion_matrix": confusion.tolist(),
        "checkpoint": str(args.checkpoint),
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    save_visualizations(examples, output_dir)

    print("\n" + "=" * 68)
    print(f"FINAL RESULTS — {args.image_size}x{args.image_size} EVALUATION")
    print("=" * 68)
    print(f"Foreground IoU: {metrics['foreground_iou']:.6f}")
    print(f"Background IoU: {metrics['background_iou']:.6f}")
    print(f"mIoU:           {metrics['miou']:.6f}")
    print(f"Boundary F1:    {metrics['boundary_f1']:.6f}")
    print("-" * 68)
    print(f"CPU ms/image:   {average_cpu_ms:.4f}")
    print(f"Model size:     {model_size_bytes} bytes")
    print("-" * 68)
    print(f"Quality Q:      {quality:.6f}")
    print(f"CES:            {ces:.6f}")
    print("-" * 68)
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion)
    print("-" * 68)
    print(f"Results saved to: {output_dir}")
    print("=" * 68)


if __name__ == "__main__":
    main()
