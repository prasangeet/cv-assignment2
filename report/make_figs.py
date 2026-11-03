"""Regenerate report figures from the repository's committed artifacts.

Run from any working directory:
    python report/make_figs.py
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = Path(__file__).resolve().parent / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NAVY = "#1F3A5F"
TEAL = "#2A9D8F"
AMBER = "#E9A23B"
CORAL = "#E76F51"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


# 1. Training curves.
history_path = ROOT / "final/unet_128_multiscale_v1/history.csv"
history = pd.read_csv(history_path)

fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9), dpi=300)
axes[0].plot(history.epoch, history.train_loss, color=NAVY, lw=1.2, label="train")
axes[0].plot(history.epoch, history.val_loss, color=CORAL, lw=1.2, label="val")
axes[0].set_xlabel("epoch")
axes[0].set_title("Loss", fontsize=8)
axes[0].legend(frameon=False, fontsize=6.5)

axes[1].plot(history.epoch, history.val_miou, color=NAVY, lw=1.2, label="mIoU")
axes[1].plot(history.epoch, history.val_boundary_f1, color=TEAL, lw=1.2, label="BF1")
axes[1].plot(history.epoch, history.val_quality, color=AMBER, lw=1.2, label="Q")
axes[1].axvline(56, color="#999", lw=0.6, ls="--")
axes[1].set_xlabel("epoch")
axes[1].set_title("Validation", fontsize=8)
axes[1].legend(frameon=False, fontsize=6.5, loc="lower right")

for axis in axes:
    axis.grid(alpha=0.25, lw=0.4)
fig.tight_layout(pad=0.4)
fig.savefig(FIG_DIR / "curves.pdf")
plt.close(fig)


# 2. Captured overlays: input, ground-truth overlay, prediction overlay.
miou = [0.915, 0.579, 0.809, 0.938, 0.944, 0.950, 0.648]
names = ["Chair", "Suitcase", "Bicycles", "Backpack", "Clogs", "Controller", "Mouse"]


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = image.astype(float).copy()
    tint = np.array(color, dtype=float)
    output[mask > 0] = 0.55 * output[mask > 0] + 0.45 * tint
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    output = output.astype(np.uint8)
    cv2.drawContours(output, contours, -1, color, 1)
    return output


fig, axes = plt.subplots(3, 7, figsize=(7.2, 3.55), dpi=300)
for index in range(7):
    sample_id = index + 1
    image_path = ROOT / f"captured/images_256/{sample_id}.png"
    ground_truth_path = ROOT / f"captured/masks_256_binary/{sample_id}.png"
    prediction_path = ROOT / f"captured/evaluation/predictions/{sample_id}.png"

    image = cv2.cvtColor(read_image(image_path), cv2.COLOR_BGR2RGB)
    ground_truth = read_image(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    prediction = read_image(prediction_path, cv2.IMREAD_GRAYSCALE)

    panels = [
        image,
        overlay(image, ground_truth, (42, 157, 143)),
        overlay(image, prediction, (231, 111, 81)),
    ]
    for row, panel in enumerate(panels):
        axes[row, index].imshow(panel)
        axes[row, index].set_xticks([])
        axes[row, index].set_yticks([])
        for spine in axes[row, index].spines.values():
            spine.set_visible(False)

    axes[0, index].set_title(
        f"{sample_id}. {names[index]}\\nmIoU {miou[index]:.3f}",
        fontsize=6.5,
        pad=2,
    )

for row, label in enumerate(["Photo", "Ground truth", "Prediction"]):
    axes[row, 0].set_ylabel(label, fontsize=7)

fig.subplots_adjust(wspace=0.03, hspace=0.03, left=0.04, right=0.995, top=0.9, bottom=0.005)
fig.savefig(FIG_DIR / "captured.pdf")
plt.close(fig)


# 3. DUTS-TE qualitative examples: selected rows from the 8-row example sheet.
sheet_path = ROOT / "final/unet_128_multiscale_v1/segmentation_examples_256.png"
sheet = cv2.cvtColor(read_image(sheet_path), cv2.COLOR_BGR2RGB)
height = sheet.shape[0] // 8
rows = [2, 7, 3, 5]
strip = np.vstack([sheet[row * height:(row + 1) * height, :1350] for row in rows])
if not cv2.imwrite(str(FIG_DIR / "duts.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)):
    raise RuntimeError("Could not write report/figs/duts.png")

print(f"Report figures written to {FIG_DIR}")
