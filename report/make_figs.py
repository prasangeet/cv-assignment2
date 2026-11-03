"""Generate the figures required by report/report.tex.

Run from any working directory:
    uv run python report/make_figs.py

The script reads only files already present in the repository and writes generated
figures to report/figs/. It does not modify source data or evaluation results.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = Path(__file__).resolve().parent
FIG_DIR = REPORT_DIR / "figs"
EXPERIMENT_DIR = ROOT / "final" / "unet_128_multiscale_v1"
CAPTURED_DIR = ROOT / "captured"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def find_by_stem(directory: Path, stem: str) -> Path:
    for extension in (".png", ".jpg", ".jpeg"):
        candidate = directory / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No image with stem {stem!r} in {directory}")


def overlay_rgb(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color_rgb: tuple[int, int, int],
    alpha: float = 0.42,
) -> np.ndarray:
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = image.astype(np.float32)
    foreground = mask > 0
    if np.any(foreground):
        color = np.asarray(color_rgb, dtype=np.float32)
        result[foreground] = (
            (1.0 - alpha) * result[foreground] + alpha * color
        )
    return np.clip(result, 0, 255).astype(np.uint8)


def make_training_curves() -> None:
    history_path = EXPERIMENT_DIR / "history.csv"
    history = read_csv(history_path)
    if not history:
        raise RuntimeError(f"No training history rows in {history_path}")

    epochs = np.asarray([int(row["epoch"]) for row in history])
    train_loss = np.asarray([float(row["train_loss"]) for row in history])
    val_loss = np.asarray([float(row["val_loss"]) for row in history])
    val_miou = np.asarray([float(row["val_miou"]) for row in history])
    val_bf1 = np.asarray([float(row["val_boundary_f1"]) for row in history])
    val_q = np.asarray([float(row["val_quality"]) for row in history])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    axes[0].plot(epochs, train_loss, label="Training loss", linewidth=1.7)
    axes[0].plot(epochs, val_loss, label="Validation loss", linewidth=1.7)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and validation loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, val_miou, label="Validation mIoU", linewidth=1.7)
    axes[1].plot(epochs, val_bf1, label="Boundary F1", linewidth=1.7)
    axes[1].plot(epochs, val_q, label="Quality Q", linewidth=1.7)
    axes[1].axvline(56, linestyle="--", linewidth=1.1, label="Selected epoch 56")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation metrics")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "curves.pdf", bbox_inches="tight")
    plt.close(fig)


def make_captured_figure() -> None:
    results_path = CAPTURED_DIR / "evaluation" / "results_captured.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    samples = results.get("per_image", [])
    if not samples:
        raise RuntimeError(f"No per-image results in {results_path}")

    columns = 4
    rows = math.ceil(len(samples) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12.0, 2.55 * rows),
        squeeze=False,
    )

    for axis in axes.flat:
        axis.axis("off")

    for index, item in enumerate(samples):
        sample_id = str(item["id"])
        image_path = find_by_stem(CAPTURED_DIR / "images_256", sample_id)
        mask_path = find_by_stem(CAPTURED_DIR / "masks_256_binary", sample_id)
        prediction_path = find_by_stem(
            CAPTURED_DIR / "evaluation" / "predictions", sample_id
        )

        image = read_bgr(image_path)
        target = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        prediction = cv2.imread(str(prediction_path), cv2.IMREAD_GRAYSCALE)
        if target is None or prediction is None:
            raise RuntimeError(f"Could not read masks for captured sample {sample_id}")
        if image.shape[:2] != target.shape or target.shape != prediction.shape:
            raise ValueError(f"Image/mask size mismatch for captured sample {sample_id}")

        photo = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ground_truth_overlay = overlay_rgb(image, target, (42, 157, 143))
        prediction_overlay = overlay_rgb(image, prediction, (231, 111, 81))
        triptych = np.concatenate(
            [photo, ground_truth_overlay, prediction_overlay], axis=1
        )

        axis = axes.flat[index]
        axis.imshow(triptych)
        axis.set_title(
            f"ID {sample_id}  |  mIoU {float(item['miou']):.3f}"
            f"  |  BF1 {float(item['boundary_f1']):.3f}",
            fontsize=8,
            pad=3,
        )
        axis.axis("off")

    fig.suptitle(
        "Captured photographs: input | ground-truth overlay | prediction overlay",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.9, w_pad=0.55)
    fig.savefig(FIG_DIR / "captured.pdf", bbox_inches="tight")
    plt.close(fig)


def make_duts_figure() -> None:
    source = EXPERIMENT_DIR / "segmentation_examples_256.png"
    image = read_bgr(source)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imsave(FIG_DIR / "duts.png", image_rgb)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_training_curves()
    make_captured_figure()
    make_duts_figure()
    print(f"Report figures written to: {FIG_DIR}")
    for name in ("curves.pdf", "captured.pdf", "duts.png"):
        path = FIG_DIR / name
        print(f"  {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
