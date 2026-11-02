"""Assignment-compliant prediction CLI for the 128x128 CNN / 256x256 masks.

Usage:
    python predict.py --image photo.jpg --out mask.png

The original photo is resized to 256x256 before timing. Twenty timed runs
start with that 256x256 image in memory and end with a 256x256 binary mask.
Only the two required timing/size lines are printed on successful execution.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.segmentation_cnn import SegmentationCNN

MODEL_SIZE = 128
OUTPUT_SIZE = 256
DEFAULT_CHECKPOINT = Path("final/unet_128_multiscale_v1/model_best.pt")


@torch.inference_mode()
def predict_from_256(image_bgr_256: np.ndarray, model: SegmentationCNN, threshold: float) -> np.ndarray:
    if image_bgr_256 is None or image_bgr_256.shape[:2] != (OUTPUT_SIZE, OUTPUT_SIZE):
        raise ValueError("Expected a 256x256 BGR image.")

    image_128 = cv2.resize(
        image_bgr_256, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA
    )
    image_rgb = cv2.cvtColor(image_128, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.ascontiguousarray(image_rgb.transpose(2, 0, 1), dtype=np.float32)
    tensor = torch.from_numpy(chw).unsqueeze(0)

    logits_128 = model(tensor)
    logits_256 = F.interpolate(
        logits_128.float(), size=(OUTPUT_SIZE, OUTPUT_SIZE),
        mode="bilinear", align_corners=False,
    )
    prediction = torch.sigmoid(logits_256)[0, 0] >= threshold
    return prediction.to(torch.uint8).cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a 256x256 binary salient-object mask.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    model = SegmentationCNN().cpu().eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)

    # File I/O and original-image resize are outside the assignment's timer.
    image_raw = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_raw is None:
        raise RuntimeError(f"Could not read input image: {args.image}")
    image_256 = cv2.resize(
        image_raw, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA
    )

    # Required warm-up run is not included in the 20-run average.
    _ = predict_from_256(image_256, model, args.threshold)

    total_cpu_seconds = 0.0
    prediction = None
    repetitions = 20
    for _ in range(repetitions):
        started = time.process_time()
        prediction = predict_from_256(image_256, model, args.threshold)
        total_cpu_seconds += time.process_time() - started

    assert prediction is not None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), prediction * np.uint8(255)):
        raise RuntimeError(f"Could not write output mask: {args.out}")

    average_ms = 1000.0 * total_cpu_seconds / repetitions
    size_bytes = args.checkpoint.stat().st_size
    print(f"T CPU ms: {average_ms:.2f}")
    print(f"S disk bytes: {size_bytes}")


if __name__ == "__main__":
    main()
