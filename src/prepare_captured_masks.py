"""Prepare binary masks and visual review assets for captured CV images.

Expected inputs (defaults):
    captured/images_256/<sample_id>.png
    captured/masks_256/<sample_id>.<png|jpg|jpeg>

Generated outputs:
    captured/masks_256_binary/<sample_id>.png
    captured/mask_review/overlays/<sample_id>.png
    captured/mask_review/contact_sheet_*.jpg
    captured/mask_review/review_sheet.jpg
    captured/mask_review/index.html
    captured/mask_review/summary.csv

The source annotations in captured/masks_256 are never modified.
"""
from __future__ import annotations

import argparse
import csv
import html
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_SIZE = 256


@dataclass
class Sample:
    sample_id: str
    image_path: Path
    mask_path: Path
    image: np.ndarray
    source_mask: np.ndarray
    binary_mask: np.ndarray
    overlay: np.ndarray
    binary_path: Path
    overlay_path: Path


def natural_key(path: Path) -> tuple[int, int | str]:
    """Sort numeric filenames numerically, then other names alphabetically."""
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem.casefold())


def index_files(directory: Path, extensions: set[str]) -> dict[str, Path]:
    """Index supported files by stem and reject ambiguous duplicate stems."""
    indexed: dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=natural_key):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        key = path.stem
        if key in indexed:
            raise RuntimeError(
                f"Duplicate sample ID '{key}' in {directory}: "
                f"{indexed[key].name} and {path.name}"
            )
        indexed[key] = path
    return indexed


def read_mask_grayscale(path: Path) -> np.ndarray:
    """Read grayscale/palette/RGBA annotations, treating transparent pixels as background."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"Could not read mask: {path}")

    if raw.ndim == 2:
        gray = raw
    elif raw.ndim == 3 and raw.shape[2] == 4:
        gray = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2GRAY)
        gray[raw[:, :, 3] == 0] = 0
    elif raw.ndim == 3 and raw.shape[2] == 3:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported mask shape {raw.shape}: {path}")

    if gray.dtype != np.uint8:
        # Convert higher-bit-depth annotations to the same 0..255 scale used
        # by the threshold. Constant-valued masks become all background.
        minimum = float(gray.min())
        maximum = float(gray.max())
        if maximum <= minimum:
            gray = np.zeros(gray.shape, dtype=np.uint8)
        else:
            gray = np.clip(
                (gray.astype(np.float32) - minimum) * (255.0 / (maximum - minimum)),
                0,
                255,
            ).astype(np.uint8)

    return gray


def make_overlay(
    image: np.ndarray,
    binary_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Tint foreground green and draw its boundary in red."""
    overlay = image.copy()
    foreground = binary_mask > 0

    if np.any(foreground):
        green = np.zeros_like(image)
        green[:, :] = (40, 220, 40)  # BGR green
        blended = cv2.addWeighted(image, 1.0 - alpha, green, alpha, 0)
        overlay[foreground] = blended[foreground]

        contours, _ = cv2.findContours(
            binary_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)

    return overlay


def add_label(tile: np.ndarray, label: str) -> np.ndarray:
    """Add a light label strip above a BGR tile."""
    height, width = tile.shape[:2]
    result = np.full((height + 30, width, 3), 248, dtype=np.uint8)
    result[30:, :] = tile
    cv2.putText(
        result,
        label,
        (7, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    return result


def make_contact_sheet(
    entries: list[tuple[str, np.ndarray]],
    output_path: Path,
    columns: int,
    tile_size: int,
) -> None:
    """Create a grid contact sheet from labeled images."""
    if not entries:
        raise ValueError(f"Cannot create an empty contact sheet: {output_path}")

    tiles: list[np.ndarray] = []
    for label, image in entries:
        resized = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        tiles.append(add_label(resized, label))

    cell_h, cell_w = tiles[0].shape[:2]
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.full((rows * cell_h, columns * cell_w, 3), 232, dtype=np.uint8)

    for index, tile in enumerate(tiles):
        row, col = divmod(index, columns)
        y, x = row * cell_h, col * cell_w
        sheet[y : y + cell_h, x : x + cell_w] = tile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Could not save contact sheet: {output_path}")


def make_review_sheet(
    samples: list[Sample],
    output_path: Path,
    panel_size: int,
) -> None:
    """Create one row per sample: image, annotation, binary mask, overlay."""
    if not samples:
        raise ValueError("Cannot create a review sheet without samples.")

    label_h = 28
    gutter = 12
    left_label_w = 74
    row_h = panel_size + label_h + 16
    panel_w = panel_size
    sheet_w = left_label_w + 4 * panel_w + 5 * gutter
    sheet_h = len(samples) * row_h + gutter
    sheet = np.full((sheet_h, sheet_w, 3), 238, dtype=np.uint8)

    column_labels = ["PHOTO", "ANNOTATION", "BINARY", "OVERLAY"]
    for col, label in enumerate(column_labels):
        x = left_label_w + gutter + col * (panel_w + gutter)
        cv2.putText(
            sheet,
            label,
            (x + 3, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    for row, sample in enumerate(samples):
        y = gutter + 28 + row * row_h
        cv2.putText(
            sheet,
            f"ID {sample.sample_id}",
            (8, y + panel_size // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )

        panels = [
            sample.image,
            cv2.cvtColor(sample.source_mask, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(sample.binary_mask, cv2.COLOR_GRAY2BGR),
            sample.overlay,
        ]
        for col, panel in enumerate(panels):
            x = left_label_w + gutter + col * (panel_w + gutter)
            resized = cv2.resize(
                panel, (panel_size, panel_size), interpolation=cv2.INTER_NEAREST
            )
            sheet[y : y + panel_size, x : x + panel_size] = resized

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Could not save review sheet: {output_path}")


def write_html_gallery(samples: list[Sample], output_path: Path) -> None:
    """Write a self-contained, local HTML gallery for detailed visual review."""
    cards: list[str] = []
    for sample in samples:
        fg_pixels = int(np.count_nonzero(sample.binary_mask))
        total_pixels = int(sample.binary_mask.size)
        fg_percent = 100.0 * fg_pixels / total_pixels
        relative = {
            "photo": Path("..") / "images_256" / sample.image_path.name,
            "annotation": Path("..") / "masks_256" / sample.mask_path.name,
            "binary": Path("..") / "masks_256_binary" / sample.binary_path.name,
            "overlay": Path("overlays") / sample.overlay_path.name,
        }
        images = "\n".join(
            f'<figure><img src="{html.escape(path.as_posix())}" '
            f'alt="{kind} {html.escape(sample.sample_id)}"><figcaption>{kind}</figcaption></figure>'
            for kind, path in relative.items()
        )
        cards.append(
            f'<section class="sample"><h2>Sample {html.escape(sample.sample_id)} '
            f'<small>Foreground: {fg_percent:.2f}%</small></h2><div class="panels">{images}</div></section>'
        )

    document = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Captured mask review</title>
<style>
:root { color-scheme: light; font-family: system-ui, sans-serif; }
body { margin: 24px; color: #202124; background: #f5f6f8; }
h1 { margin-bottom: 6px; }
.intro { color: #555; margin-top: 0; }
.sample { background: white; border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin: 18px 0; }
.sample h2 { margin: 0 0 12px; font-size: 1.1rem; }
.sample small { color: #666; font-weight: normal; margin-left: 12px; }
.panels { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
figure { margin: 0; min-width: 0; }
figure img { display: block; width: 100%; height: auto; image-rendering: auto; border: 1px solid #ddd; }
figcaption { color: #555; text-transform: capitalize; font-size: .9rem; margin-top: 5px; }
@media (max-width: 850px) { .panels { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 480px) { body { margin: 10px; } .panels { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<h1>Captured mask review</h1>
<p class="intro">Compare each photo, hand annotation, generated binary mask, and foreground overlay. Green indicates foreground; red traces the boundary.</p>
""" + "\n".join(cards) + "\n</body>\n</html>\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert captured annotations to binary masks and generate visual review assets."
    )
    parser.add_argument(
        "--images", type=Path, default=Path("captured/images_256"),
        help="Prepared input photos (default: captured/images_256)",
    )
    parser.add_argument(
        "--masks", type=Path, default=Path("captured/masks_256"),
        help="Manually annotated masks (default: captured/masks_256)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("captured"),
        help="Output root (default: captured)",
    )
    parser.add_argument(
        "--threshold", type=int, default=128,
        help="Grayscale threshold: values >= threshold become foreground (default: 128)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.42,
        help="Foreground tint opacity for overlays, between 0 and 1 (default: 0.42)",
    )
    parser.add_argument(
        "--columns", type=int, default=3,
        help="Number of columns in individual contact sheets (default: 3)",
    )
    parser.add_argument(
        "--size", type=int, default=DEFAULT_SIZE,
        help="Expected square image/mask size (default: 256)",
    )
    parser.add_argument(
        "--resize-masks", action="store_true",
        help="Resize mismatched annotation masks to the photo size using nearest-neighbor interpolation.",
    )
    parser.add_argument(
        "--review-panel-size", type=int, default=220,
        help="Panel size in the combined review sheet (default: 220)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.images.is_dir():
        raise FileNotFoundError(f"Image directory not found: {args.images}")
    if not args.masks.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {args.masks}")
    if not 0 <= args.threshold <= 255:
        raise ValueError("--threshold must be between 0 and 255")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")
    if args.columns < 1 or args.size < 1 or args.review_panel_size < 1:
        raise ValueError("--columns, --size, and --review-panel-size must be positive")

    image_files = index_files(args.images, IMAGE_EXTENSIONS)
    mask_files = index_files(args.masks, IMAGE_EXTENSIONS)
    if not image_files:
        raise RuntimeError(f"No supported images found in {args.images}")
    if not mask_files:
        raise RuntimeError(f"No supported annotations found in {args.masks}")

    missing_masks = sorted(set(image_files) - set(mask_files), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x.casefold()))
    extra_masks = sorted(set(mask_files) - set(image_files), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x.casefold()))
    if missing_masks or extra_masks:
        details = []
        if missing_masks:
            details.append("Missing masks for image IDs: " + ", ".join(missing_masks))
        if extra_masks:
            details.append("Masks with no matching image IDs: " + ", ".join(extra_masks))
        raise RuntimeError("Image/mask pairing check failed:\n- " + "\n- ".join(details))

    binary_dir = args.output / "masks_256_binary"
    review_dir = args.output / "mask_review"
    overlay_dir = review_dir / "overlays"
    binary_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    samples: list[Sample] = []
    for sample_id in sorted(image_files, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x.casefold())):
        image_path = image_files[sample_id]
        mask_path = mask_files[sample_id]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        if image.shape[:2] != (args.size, args.size):
            raise ValueError(
                f"Expected {args.size}x{args.size} photo, got "
                f"{image.shape[1]}x{image.shape[0]}: {image_path}"
            )

        source_mask = read_mask_grayscale(mask_path)
        if source_mask.shape != image.shape[:2]:
            if not args.resize_masks:
                raise ValueError(
                    f"Mask/photo size mismatch for ID {sample_id}: "
                    f"mask={source_mask.shape[1]}x{source_mask.shape[0]}, "
                    f"photo={image.shape[1]}x{image.shape[0]}. "
                    "Use --resize-masks to resize the annotation with nearest-neighbor."
                )
            source_mask = cv2.resize(
                source_mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        binary_mask = np.where(source_mask >= args.threshold, 255, 0).astype(np.uint8)
        binary_path = binary_dir / f"{sample_id}.png"
        if not cv2.imwrite(str(binary_path), binary_mask):
            raise RuntimeError(f"Could not save binary mask: {binary_path}")

        overlay = make_overlay(image, binary_mask, args.alpha)
        overlay_path = overlay_dir / f"{sample_id}.png"
        if not cv2.imwrite(str(overlay_path), overlay):
            raise RuntimeError(f"Could not save overlay: {overlay_path}")

        unique_values = np.unique(source_mask)
        foreground_pixels = int(np.count_nonzero(binary_mask))
        foreground_percent = 100.0 * foreground_pixels / binary_mask.size
        print(
            f"ID {sample_id}: binary={binary_path} | "
            f"source values={len(unique_values)} | "
            f"binary values={np.unique(binary_mask).tolist()} | "
            f"foreground={foreground_percent:.2f}%"
        )

        samples.append(
            Sample(
                sample_id=sample_id,
                image_path=image_path,
                mask_path=mask_path,
                image=image,
                source_mask=source_mask,
                binary_mask=binary_mask,
                overlay=overlay,
                binary_path=binary_path,
                overlay_path=overlay_path,
            )
        )

    # Four independent contact sheets make it easy to inspect each data type.
    make_contact_sheet(
        [(f"PHOTO {s.sample_id}", s.image) for s in samples],
        review_dir / "contact_sheet_photos.jpg",
        args.columns,
        args.size,
    )
    make_contact_sheet(
        [(f"ANNOTATION {s.sample_id}", cv2.cvtColor(s.source_mask, cv2.COLOR_GRAY2BGR)) for s in samples],
        review_dir / "contact_sheet_annotations.jpg",
        args.columns,
        args.size,
    )
    make_contact_sheet(
        [(f"BINARY {s.sample_id}", cv2.cvtColor(s.binary_mask, cv2.COLOR_GRAY2BGR)) for s in samples],
        review_dir / "contact_sheet_binary_masks.jpg",
        args.columns,
        args.size,
    )
    make_contact_sheet(
        [(f"OVERLAY {s.sample_id}", s.overlay) for s in samples],
        review_dir / "contact_sheet_overlays.jpg",
        args.columns,
        args.size,
    )

    make_review_sheet(
        samples,
        review_dir / "review_sheet.jpg",
        args.review_panel_size,
    )
    write_html_gallery(samples, review_dir / "index.html")

    summary_path = review_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "image_file",
                "annotation_file",
                "binary_mask_file",
                "width",
                "height",
                "threshold",
                "source_unique_values_count",
                "binary_unique_values",
                "foreground_pixels",
                "background_pixels",
                "foreground_percent",
            ]
        )
        for sample in samples:
            foreground = int(np.count_nonzero(sample.binary_mask))
            total = int(sample.binary_mask.size)
            writer.writerow(
                [
                    sample.sample_id,
                    sample.image_path.name,
                    sample.mask_path.name,
                    sample.binary_path.name,
                    sample.binary_mask.shape[1],
                    sample.binary_mask.shape[0],
                    args.threshold,
                    int(np.unique(sample.source_mask).size),
                    "0;255",
                    foreground,
                    total - foreground,
                    f"{100.0 * foreground / total:.4f}",
                ]
            )

    print("\nMask preparation complete.")
    print(f"Samples processed: {len(samples)}")
    print(f"Binary masks: {binary_dir}")
    print(f"Overlays: {overlay_dir}")
    print(f"Review sheet: {review_dir / 'review_sheet.jpg'}")
    print(f"HTML gallery: {review_dir / 'index.html'}")
    print(f"Summary CSV: {summary_path}")
    print(f"Contact sheets: {review_dir / 'contact_sheet_*.jpg'}")


if __name__ == "__main__":
    main()
