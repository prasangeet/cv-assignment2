"""Benchmark CPU inference latency for the current DUTS segmentation model.

Measures both:
1. Model-only latency (preprocessed batch-1 tensor -> logits).
2. Evaluator-style in-memory pipeline latency (BGR resize/conversion/normalization,
   tensor creation, model forward, sigmoid, threshold, and output conversion).

Image file I/O is excluded from timing, matching src.evaluate's benchmark protocol.
No training or checkpoint is required.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np
import torch

from models.segmentation_cnn import SegmentationCNN, count_parameters
from src.dataset import DUTSDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark batch-1 CPU latency before training."
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["DUTS-TE", "DUTS-TR"], default="DUTS-TR")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--samples", type=int, default=200,
                        help="Number of images sampled across the split; use 0 for all images.")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200,
                        help="Repeated iterations for model-only timing.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Optional trained state_dict. Not required for latency testing.")
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/latency_benchmark_256.json"))
    parser.add_argument("--max-cpu-ms", type=float, default=50.0)
    args = parser.parse_args()

    if args.image_size < 32:
        parser.error("--image-size must be at least 32")
    if args.samples < 0 or args.warmup < 1 or args.iterations < 1:
        parser.error("samples must be >= 0; warmup and iterations must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("threshold must be between 0 and 1")
    if args.max_cpu_ms <= 0:
        parser.error("max-cpu-ms must be positive")
    return args


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def preprocess(image_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    if image_bgr is None:
        raise ValueError("Encountered an unreadable image.")
    image = cv2.resize(
        image_bgr,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
    return torch.from_numpy(image).unsqueeze(0)


@torch.inference_mode()
def run_pipeline(
    image_bgr: np.ndarray,
    model: torch.nn.Module,
    image_size: int,
    threshold: float,
) -> np.ndarray:
    tensor = preprocess(image_bgr, image_size)
    logits = model(tensor)
    prediction = torch.sigmoid(logits) >= threshold
    return prediction[0, 0].numpy().astype(np.uint8)


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    # Mirror evaluator CPU settings.
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    dataset = DUTSDataset(
        root=args.data,
        split=args.split,
        image_size=args.image_size,
        augment=False,
    )
    if not dataset.samples:
        raise RuntimeError(f"No images found for {args.split} under {args.data}.")

    if args.samples == 0 or args.samples >= len(dataset.samples):
        selected_indices = np.arange(len(dataset.samples), dtype=np.int64)
    else:
        # Spread the sample over the whole split instead of timing only its first files.
        selected_indices = np.linspace(
            0, len(dataset.samples) - 1, num=args.samples, dtype=np.int64
        )

    # Load files before timing, so storage latency is not included.
    raw_images: list[np.ndarray] = []
    for index in selected_indices:
        image_path, _mask_path = dataset.samples[int(index)]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        raw_images.append(image_bgr)

    model = SegmentationCNN().cpu().eval()
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)

    first_tensor = preprocess(raw_images[0], args.image_size)

    # Warm up kernels and memory paths before collecting timings.
    for _ in range(args.warmup):
        _ = model(first_tensor)
        _ = run_pipeline(raw_images[0], model, args.image_size, args.threshold)

    # Model-only timing: excludes image preprocessing and thresholding.
    model_only_ms: list[float] = []
    for _ in range(args.iterations):
        started = time.process_time()
        _ = model(first_tensor)
        model_only_ms.append((time.process_time() - started) * 1000.0)

    # Evaluator-style timing: preprocessing + forward + sigmoid/threshold.
    pipeline_ms: list[float] = []
    for image_bgr in raw_images:
        started = time.process_time()
        _ = run_pipeline(image_bgr, model, args.image_size, args.threshold)
        pipeline_ms.append((time.process_time() - started) * 1000.0)

    # Estimate the serialized state_dict size. Actual trained checkpoint file
    # size may differ slightly because of serialization metadata.
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    estimated_size_bytes = buffer.tell()

    result = {
        "architecture": type(model).__name__,
        "split_used_for_timing": args.split,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "weights": "checkpoint" if args.checkpoint else "random initialization (architecture-only timing)",
        "device": "CPU",
        "cpu_threads": torch.get_num_threads(),
        "batch_size": 1,
        "input_resolution": args.image_size,
        "timed_images": len(raw_images),
        "warmup_iterations": args.warmup,
        "model_only_iterations": args.iterations,
        "trainable_parameters": count_parameters(model),
        "estimated_state_dict_size_bytes": estimated_size_bytes,
        "model_only_cpu_ms_mean": mean(model_only_ms),
        "model_only_cpu_ms_median": median(model_only_ms),
        "model_only_cpu_ms_p95": percentile(model_only_ms, 95),
        "pipeline_cpu_ms_mean": mean(pipeline_ms),
        "pipeline_cpu_ms_median": median(pipeline_ms),
        "pipeline_cpu_ms_p95": percentile(pipeline_ms, 95),
        "max_cpu_ms": args.max_cpu_ms,
        "within_latency_budget": mean(pipeline_ms) <= args.max_cpu_ms,
        "timing_note": "CPU process time; image file I/O excluded; pipeline includes preprocessing, model, sigmoid and threshold.",
        "ces_note": "CES cannot be calculated before trained-model segmentation quality metrics are available.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 68)
    print(f"CPU LATENCY BENCHMARK — {args.image_size}x{args.image_size}")
    print("=" * 68)
    print(f"Architecture:          {type(model).__name__}")
    print(f"Weights:               {result['weights']}")
    print(f"Images timed:          {len(raw_images)}")
    print(f"CPU threads:           {torch.get_num_threads()}")
    print(f"Trainable parameters:  {count_parameters(model):,}")
    print(f"Estimated model size:  {estimated_size_bytes:,} bytes")
    print("-" * 68)
    print(f"Model-only mean:       {mean(model_only_ms):.4f} ms/image")
    print(f"Model-only median:     {median(model_only_ms):.4f} ms/image")
    print(f"Model-only p95:        {percentile(model_only_ms, 95):.4f} ms/image")
    print("-" * 68)
    print(f"Pipeline mean:         {mean(pipeline_ms):.4f} ms/image")
    print(f"Pipeline median:       {median(pipeline_ms):.4f} ms/image")
    print(f"Pipeline p95:          {percentile(pipeline_ms, 95):.4f} ms/image")
    print(f"Latency budget:        {args.max_cpu_ms:.2f} ms/image")
    print(f"Within budget:         {'YES' if result['within_latency_budget'] else 'NO'}")
    print("CES:                   Not available before trained quality metrics")
    print("-" * 68)
    print(f"Benchmark saved to:   {args.output}")
    print("=" * 68)


if __name__ == "__main__":
    main()
