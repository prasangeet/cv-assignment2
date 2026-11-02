"""Official DUTS evaluation: 128x128 CNN inference, 256x256 metrics.

The timed pipeline starts with a 256x256 BGR image already in memory and ends
with a 256x256 uint8 {0,1} prediction. File I/O and original-image resizing
are deliberately outside the timed region, matching CV Assignment 2.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable

# The assignment requires these to be set before importing NumPy/OpenCV/PyTorch.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models.segmentation_cnn import SegmentationCNN
from src.dataset import DUTSDataset

MODEL_SIZE = 128
OFFICIAL_SIZE = 256
CPU_BUDGET_MS = 50.0
SIZE_BUDGET_BYTES = 4 * 1024 * 1024
BOUNDARY_TOLERANCE = 2


def boundary_map(mask: np.ndarray) -> np.ndarray:
    """4-connected foreground boundary; image exterior counts as background."""
    binary = np.ascontiguousarray(mask.astype(np.uint8))
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float32)
    neighbour_count = cv2.filter2D(
        binary, cv2.CV_32F, cross, borderType=cv2.BORDER_CONSTANT
    )
    return (binary.astype(bool) & (neighbour_count < 5.0))


def boundary_f1_5x5(prediction: np.ndarray, target: np.ndarray) -> float:
    """Per-image Boundary F1 with a 5x5 matching window (radius 2)."""
    pred_boundary = boundary_map(prediction)
    true_boundary = boundary_map(target)
    pred_count = int(pred_boundary.sum())
    true_count = int(true_boundary.sum())

    if pred_count == 0 and true_count == 0:
        return 1.0
    if pred_count == 0 or true_count == 0:
        return 0.0

    window = np.ones((2 * BOUNDARY_TOLERANCE + 1,) * 2, dtype=np.float32)
    true_nearby = cv2.filter2D(
        true_boundary.astype(np.uint8), cv2.CV_32F, window,
        borderType=cv2.BORDER_CONSTANT,
    ) > 0
    pred_nearby = cv2.filter2D(
        pred_boundary.astype(np.uint8), cv2.CV_32F, window,
        borderType=cv2.BORDER_CONSTANT,
    ) > 0

    precision = float(np.count_nonzero(pred_boundary & true_nearby)) / pred_count
    recall = float(np.count_nonzero(true_boundary & pred_nearby)) / true_count
    denominator = precision + recall
    return 0.0 if denominator == 0.0 else 2.0 * precision * recall / denominator


def calculate_ces(miou: float, boundary: float, cpu_ms: float, size_bytes: int) -> tuple[float, float]:
    quality = 0.70 * miou + 0.30 * boundary
    if cpu_ms > CPU_BUDGET_MS:
        return quality, 0.0
    time_penalty = 0.20 * cpu_ms / CPU_BUDGET_MS
    size_penalty = 0.10 * min(1.0, size_bytes / SIZE_BUDGET_BYTES)
    return quality, quality * (1.0 - time_penalty - size_penalty)


@torch.inference_mode()
def predict_from_256(image_bgr_256: np.ndarray, model: SegmentationCNN, threshold: float) -> np.ndarray:
    """256 BGR input -> 128 model input -> 256 binary uint8 mask."""
    if image_bgr_256 is None or image_bgr_256.shape[:2] != (OFFICIAL_SIZE, OFFICIAL_SIZE):
        raise ValueError(f"Expected a {OFFICIAL_SIZE}x{OFFICIAL_SIZE} BGR image in memory.")

    # All operations in this function count toward T_CPU.
    image_128 = cv2.resize(
        image_bgr_256, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA
    )
    image_rgb = cv2.cvtColor(image_128, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.ascontiguousarray(image_rgb.transpose(2, 0, 1), dtype=np.float32)
    tensor = torch.from_numpy(chw).unsqueeze(0)

    logits_128 = model(tensor)
    if not isinstance(logits_128, torch.Tensor):
        raise RuntimeError("SegmentationCNN(x) must return a logits tensor for inference.")
    if tuple(logits_128.shape[-2:]) != (MODEL_SIZE, MODEL_SIZE):
        raise RuntimeError(f"Expected 128x128 logits, received {tuple(logits_128.shape[-2:])}.")

    logits_256 = F.interpolate(
        logits_128.float(), size=(OFFICIAL_SIZE, OFFICIAL_SIZE),
        mode="bilinear", align_corners=False,
    )
    prediction = (torch.sigmoid(logits_256)[0, 0] >= threshold)
    return prediction.to(torch.uint8).cpu().numpy()


def choose_samples(args: argparse.Namespace, dataset: DUTSDataset) -> list[tuple[Path, Path]]:
    samples = list(dataset.samples)
    if args.split == "DUTS-TE":
        if len(samples) != 5019:
            raise RuntimeError(f"Expected 5,019 DUTS-TE samples, found {len(samples)}.")
        if args.subset != "all":
            raise ValueError("DUTS-TE supports only --subset all.")
        return samples

    if args.subset != "validation":
        raise ValueError("For DUTS-TR, evaluate only --subset validation using saved split indices.")
    split_path = args.split_indices or (args.checkpoint.parent / "split_indices.json")
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Validation split file not found: {split_path}. Pass --split-indices explicitly."
        )
    split_info = json.loads(split_path.read_text(encoding="utf-8"))
    validation_names = split_info.get("validation_filenames")
    if not isinstance(validation_names, list) or not validation_names:
        raise ValueError(f"{split_path} does not contain validation_filenames.")
    validation_set = set(validation_names)
    selected = [sample for sample in samples if sample[0].name in validation_set]
    selected_names = {sample[0].name for sample in selected}
    missing = validation_set - selected_names
    if missing:
        raise RuntimeError(f"{len(missing)} saved validation images are missing from DUTS-TR.")
    if len(selected) != len(validation_names):
        raise RuntimeError("Duplicate or inconsistent validation filenames in split file.")
    return selected


def save_visualizations(
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray]], output_dir: Path
) -> None:
    if not examples:
        return
    fig, axes = plt.subplots(len(examples), 4, figsize=(12, 3.0 * len(examples)), squeeze=False)
    for row, (image, target, prediction) in enumerate(examples):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title("Input 256x256")
        axes[row, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("Prediction")
        overlay = image.copy()
        selected = prediction.astype(bool)
        overlay[selected] = (
            0.55 * overlay[selected] + 0.45 * np.array([255, 0, 0], dtype=np.float32)
        ).astype(np.uint8)
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title("Prediction overlay")
        for axis in axes[row]:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "segmentation_examples_256.png", dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official 256x256 metrics with a 128x128 CNN inference resolution."
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("DUTS-TE", "DUTS-TR"), default="DUTS-TE")
    parser.add_argument("--subset", choices=("all", "validation"), default="all")
    parser.add_argument("--split-indices", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.examples < 0:
        parser.error("--examples cannot be negative")
    if args.split == "DUTS-TR" and args.subset != "validation":
        parser.error("For DUTS-TR, use --subset validation; training images must not be evaluated as validation.")
    if args.split == "DUTS-TE" and args.subset != "all":
        parser.error("DUTS-TE supports only --subset all.")
    return args


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    device = torch.device("cpu")

    # The dataset object is used for its canonical image/mask path pairs.
    # Actual official-resolution preprocessing is performed below.
    dataset = DUTSDataset(
        root=args.data, split=args.split, image_size=MODEL_SIZE, augment=False
    )
    samples = choose_samples(args, dataset)
    if not samples:
        raise RuntimeError("No evaluation samples were selected.")

    model = SegmentationCNN().to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=True)

    # Warm-up before timing, as required by the assignment.
    warm_image_path, _ = samples[0]
    warm_raw = cv2.imread(str(warm_image_path), cv2.IMREAD_COLOR)
    if warm_raw is None:
        raise RuntimeError(f"Could not read warm-up image: {warm_image_path}")
    warm_256 = cv2.resize(warm_raw, (OFFICIAL_SIZE, OFFICIAL_SIZE), interpolation=cv2.INTER_AREA)
    _ = predict_from_256(warm_256, model, args.threshold)

    confusion = np.zeros((2, 2), dtype=np.int64)  # [[TN, FP], [FN, TP]]
    boundary_scores: list[float] = []
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    total_cpu_seconds = 0.0

    print("=" * 68)
    print("OFFICIAL DUTS EVALUATION — 256x256 METRICS / 128x128 CNN")
    print("=" * 68)
    print(f"Split: {args.split} ({args.subset}) | Images: {len(samples)}")
    print(f"CNN input: {MODEL_SIZE}x{MODEL_SIZE} | Metric/output: {OFFICIAL_SIZE}x{OFFICIAL_SIZE}")
    print("Device: CPU | Threads: 1")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Threshold: {args.threshold}")

    for index, (image_path, mask_path) in enumerate(samples, start=1):
        # File I/O and original-photo resize are outside T_CPU.
        image_raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_raw is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        if mask_raw is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        image_256 = cv2.resize(
            image_raw, (OFFICIAL_SIZE, OFFICIAL_SIZE), interpolation=cv2.INTER_AREA
        )
        target_256 = cv2.resize(
            mask_raw, (OFFICIAL_SIZE, OFFICIAL_SIZE), interpolation=cv2.INTER_NEAREST
        )
        target = (target_256 >= 128).astype(np.uint8)

        # The timed region begins with image_256 already resident in memory and
        # ends when the 256x256 binary prediction has been materialized on CPU.
        started = time.process_time()
        prediction = predict_from_256(image_256, model, args.threshold)
        total_cpu_seconds += time.process_time() - started

        target_bool = target.astype(bool)
        pred_bool = prediction.astype(bool)
        confusion[0, 0] += np.count_nonzero(~target_bool & ~pred_bool)
        confusion[0, 1] += np.count_nonzero(~target_bool & pred_bool)
        confusion[1, 0] += np.count_nonzero(target_bool & ~pred_bool)
        confusion[1, 1] += np.count_nonzero(target_bool & pred_bool)
        boundary_scores.append(boundary_f1_5x5(pred_bool, target_bool))

        if len(examples) < args.examples:
            image_rgb = cv2.cvtColor(image_256, cv2.COLOR_BGR2RGB)
            examples.append((image_rgb, target, prediction))

        if index % 100 == 0 or index == len(samples):
            print(f"Evaluated {index}/{len(samples)}")

    tn, fp = (int(v) for v in confusion[0])
    fn, tp = (int(v) for v in confusion[1])
    fg_iou = tp / max(tp + fp + fn, 1)
    bg_iou = tn / max(tn + fp + fn, 1)
    miou = 0.5 * (fg_iou + bg_iou)
    mean_boundary = float(np.mean(boundary_scores))
    average_cpu_ms = 1000.0 * total_cpu_seconds / len(samples)
    model_size_bytes = args.checkpoint.stat().st_size
    quality, ces = calculate_ces(miou, mean_boundary, average_cpu_ms, model_size_bytes)

    output_dir = args.output_dir or args.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "model": args.checkpoint.name,
        "split": args.split,
        "subset": args.subset,
        "num_images": len(samples),
        "cnn_input_resolution": MODEL_SIZE,
        "official_metric_resolution": OFFICIAL_SIZE,
        "ground_truth_resolution": OFFICIAL_SIZE,
        "prediction_resolution": OFFICIAL_SIZE,
        "threshold": args.threshold,
        "device": "CPU",
        "cpu_threads": 1,
        "timing_protocol": "256x256 BGR in memory -> downsample/color/normalize/tensor/model/upsample/threshold -> 256x256 uint8 mask; file I/O excluded",
        "total_cpu_seconds": total_cpu_seconds,
        "average_cpu_ms": average_cpu_ms,
        "model_size_bytes": model_size_bytes,
        "foreground_iou": fg_iou,
        "background_iou": bg_iou,
        "miou": miou,
        "boundary_f1": mean_boundary,
        "quality_score": quality,
        "ces_score": ces,
        "confusion_matrix": confusion.tolist(),
        "checkpoint": str(args.checkpoint),
    }
    (output_dir / f"results_{args.split.lower()}_256.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    save_visualizations(examples, output_dir)

    print("\n" + "=" * 68)
    print("FINAL RESULTS — OFFICIAL 256x256 EVALUATION (128x128 MODEL)")
    print("=" * 68)
    print(f"Foreground IoU: {fg_iou:.6f}")
    print(f"Background IoU: {bg_iou:.6f}")
    print(f"mIoU:           {miou:.6f}")
    print(f"Boundary F1:    {mean_boundary:.6f}")
    print("-" * 68)
    print(f"CPU ms/image:   {average_cpu_ms:.4f}")
    print(f"Model size:     {model_size_bytes} bytes")
    print("-" * 68)
    print(f"Quality Q:      {quality:.6f}")
    print(f"CES:            {ces:.6f}")
    print("-" * 68)
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion)
    print(f"Results saved to: {output_dir}")
    print("=" * 68)


if __name__ == "__main__":
    main()
