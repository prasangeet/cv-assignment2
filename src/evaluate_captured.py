
"""Evaluate captured photos: segmentation quality, CPU latency, and model size.

All paired captured images are evaluated, including difficult scenes.
Results and visualizations overwrite the previous captured evaluation outputs.

Example:
    uv run python -m src.evaluate_captured \
        --checkpoint final/unet_128_multiscale_v1/model_best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Set thread limits before importing numerical libraries.
for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[name] = "1"

import cv2
import numpy as np
import torch

from models.segmentation_cnn import SegmentationCNN
from src.evaluate import (
    boundary_f1_5x5,
    predict_from_256,
    save_visualizations,
)

SIZE = 256
EXTENSIONS = (".png", ".jpg", ".jpeg")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate captured images with quality and latency metrics."
    )
    parser.add_argument(
        "--images", type=Path, default=Path("captured/images_256")
    )
    parser.add_argument(
        "--masks", type=Path, default=Path("captured/masks_256_binary")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("captured/evaluation")
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--repetitions", type=int, default=20)
    return parser.parse_args()


def find_file(directory: Path, stem: str):
    for extension in EXTENSIONS:
        path = directory / f"{stem}{extension}"
        if path.is_file():
            return path
    return None


def discover_samples(images_dir: Path, masks_dir: Path):
    samples = []

    for image_path in images_dir.iterdir():
        if image_path.suffix.lower() not in EXTENSIONS:
            continue

        mask_path = find_file(masks_dir, image_path.stem)

        if mask_path is not None:
            samples.append((image_path.stem, image_path, mask_path))

    return sorted(
        samples,
        key=lambda item: (
            0, int(item[0])
        ) if item[0].isdigit() else (1, item[0].casefold()),
    )


def calculate_scores(tn, fp, fn, tp):
    fg_iou = tp / max(tp + fp + fn, 1)
    bg_iou = tn / max(tn + fp + fn, 1)

    return {
        "foreground_iou": fg_iou,
        "background_iou": bg_iou,
        "miou": (fg_iou + bg_iou) / 2,
    }


def confusion_counts(prediction, target):
    pred = prediction.astype(bool)
    truth = target.astype(bool)

    return (
        int(np.count_nonzero(~truth & ~pred)),
        int(np.count_nonzero(~truth & pred)),
        int(np.count_nonzero(truth & ~pred)),
        int(np.count_nonzero(truth & pred)),
    )


def main():
    args = parse_args()

    if not 0 <= args.threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    if args.repetitions < 1:
        raise ValueError("Repetitions must be at least 1.")

    if not args.images.is_dir():
        raise FileNotFoundError(args.images)

    if not args.masks.is_dir():
        raise FileNotFoundError(args.masks)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    cv2.setNumThreads(1)
    torch.set_num_threads(1)

    samples = discover_samples(args.images, args.masks)

    if not samples:
        raise RuntimeError("No matching image/mask pairs found.")

    # Load and validate all images before timing.
    loaded = []

    for sample_id, image_path, mask_path in samples:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise RuntimeError(f"Could not read image/mask pair: {sample_id}")

        if image.shape[:2] != (SIZE, SIZE):
            raise ValueError(f"{image_path} must be 256x256.")

        if mask.shape != (SIZE, SIZE):
            raise ValueError(f"{mask_path} must be 256x256.")

        values = set(np.unique(mask).tolist())
        if not values.issubset({0, 255}):
            raise ValueError(
                f"{mask_path} is not binary. Found values: {sorted(values)}"
            )

        loaded.append({
            "id": sample_id,
            "image_path": image_path,
            "mask_path": mask_path,
            "image": image,
            "target": (mask >= 128).astype(np.uint8),
        })

    # Load the trained model on CPU.
    model = SegmentationCNN().cpu().eval()

    state = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)

    # One warm-up prediction, excluded from latency measurements.
    predict_from_256(loaded[0]["image"], model, args.threshold)

    output_dir = args.output_dir
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    total_confusion = np.zeros(4, dtype=np.int64)
    per_image = []
    visualization_examples = []
    all_times = []

    print("=" * 72)
    print("CAPTURED PHOTO EVALUATION")
    print("=" * 72)
    print(f"Images: {len(loaded)}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: CPU | Threads: 1")
    print(f"Resolution: 256x256")
    print(f"Timed repetitions per image: {args.repetitions}")
    print("All paired images included.")
    print("-" * 72)

    for sample in loaded:
        sample_id = sample["id"]
        image = sample["image"]
        target = sample["target"]

        prediction = None
        run_times = []

        # Timing starts with an in-memory 256x256 BGR image.
        # It ends with the 256x256 binary prediction.
        for _ in range(args.repetitions):
            start = time.process_time()

            prediction = predict_from_256(
                image, model, args.threshold
            )

            elapsed_ms = (time.process_time() - start) * 1000
            run_times.append(elapsed_ms)
            all_times.append(elapsed_ms)

        prediction = prediction.astype(np.uint8)

        tn, fp, fn, tp = confusion_counts(prediction, target)
        total_confusion += np.array([tn, fp, fn, tp])

        scores = calculate_scores(tn, fp, fn, tp)

        boundary = boundary_f1_5x5(
            prediction.astype(bool),
            target.astype(bool),
        )

        prediction_path = predictions_dir / f"{sample_id}.png"

        if not cv2.imwrite(
            str(prediction_path),
            prediction * np.uint8(255),
        ):
            raise RuntimeError(f"Failed to save {prediction_path}")

        latency_mean = float(np.mean(run_times))
        latency_std = float(np.std(run_times))

        per_image.append({
            "id": sample_id,
            "image": str(sample["image_path"]),
            "ground_truth_mask": str(sample["mask_path"]),
            "prediction": str(prediction_path),
            **scores,
            "boundary_f1": boundary,
            "foreground_pixels_percent": float(
                np.mean(target) * 100
            ),
            "cpu_ms_mean": latency_mean,
            "cpu_ms_std": latency_std,
            "latency_repetitions": args.repetitions,
        })

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        visualization_examples.append(
            (image_rgb, target, prediction)
        )

        print(
            f"{sample_id}: "
            f"mIoU={scores['miou']:.6f} | "
            f"Boundary F1={boundary:.6f} | "
            f"CPU={latency_mean:.3f} ms"
        )

    # Aggregate segmentation metrics.
    tn, fp, fn, tp = map(int, total_confusion)

    aggregate = calculate_scores(tn, fp, fn, tp)

    mean_boundary = float(np.mean([
        item["boundary_f1"] for item in per_image
    ]))

    mean_image_miou = float(np.mean([
        item["miou"] for item in per_image
    ]))

    quality = (
        0.70 * aggregate["miou"]
        + 0.30 * mean_boundary
    )

    # Aggregate repeated CPU measurements.
    cpu_mean = float(np.mean(all_times))
    cpu_median = float(np.median(all_times))
    cpu_std = float(np.std(all_times))
    cpu_min = float(np.min(all_times))
    cpu_max = float(np.max(all_times))

    # Exact checkpoint size, without compression.
    model_size = int(args.checkpoint.stat().st_size)

    results = {
        "evaluation": "captured_photos",
        "checkpoint": str(args.checkpoint),
        "num_images": len(loaded),
        "sample_ids": [item["id"] for item in loaded],
        "resolution": [SIZE, SIZE],
        "threshold": args.threshold,
        "device": "CPU",
        "cpu_threads": 1,
        "foreground_iou": aggregate["foreground_iou"],
        "background_iou": aggregate["background_iou"],
        "aggregate_miou": aggregate["miou"],
        "mean_per_image_miou": mean_image_miou,
        "mean_boundary_f1": mean_boundary,
        "quality_score": quality,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "latency": {
            "warmup_runs": 1,
            "repetitions_per_image": args.repetitions,
            "total_timed_predictions": len(all_times),
            "average_cpu_ms_per_prediction": cpu_mean,
            "median_cpu_ms_per_prediction": cpu_median,
            "std_cpu_ms_per_prediction": cpu_std,
            "min_cpu_ms_per_prediction": cpu_min,
            "max_cpu_ms_per_prediction": cpu_max,
        },
        "model_size_bytes": model_size,
        "model_size_mib": model_size / (1024 * 1024),
        "ces_score": None,
        "ces_note": (
            "Custom captured-photo evaluation. Official CES must use "
            "the full DUTS-TE benchmark timing protocol."
        ),
        "per_image": per_image,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results_captured.json"

    # Overwrite previous results.
    results_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    save_visualizations(
        visualization_examples,
        output_dir,
    )

    print("-" * 72)
    print("AGGREGATE RESULTS")
    print(f"Foreground IoU:       {aggregate['foreground_iou']:.6f}")
    print(f"Background IoU:       {aggregate['background_iou']:.6f}")
    print(f"Aggregate mIoU:       {aggregate['miou']:.6f}")
    print(f"Mean image mIoU:      {mean_image_miou:.6f}")
    print(f"Mean Boundary F1:     {mean_boundary:.6f}")
    print(f"Quality Q:            {quality:.6f}")
    print("-" * 72)
    print(f"T CPU mean ms:        {cpu_mean:.3f}")
    print(f"T CPU median ms:      {cpu_median:.3f}")
    print(f"T CPU std ms:         {cpu_std:.3f}")
    print(f"T CPU range ms:       {cpu_min:.3f} - {cpu_max:.3f}")
    print(f"S disk bytes:         {model_size}")
    print(f"Model size MiB:       {model_size / (1024 * 1024):.4f}")
    print("CES:                  Not calculated for custom photos")
    print(f"Results overwritten:  {results_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
