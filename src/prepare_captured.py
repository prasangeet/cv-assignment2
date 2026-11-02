
"""Prepare personal captured photos for CV Assignment 2."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


SIZE = 256
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def natural_key(path: Path):
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem.casefold())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("captures"))
    parser.add_argument("--output", type=Path, default=Path("captured"))
    args = parser.parse_args()

    if not args.source.is_dir():
        raise FileNotFoundError(f"Source folder not found: {args.source}")

    paths = sorted(
        [
            p for p in args.source.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=natural_key,
    )

    if not paths:
        raise RuntimeError("No supported images found.")

    original_dir = args.output / "original"
    images_dir = args.output / "images_256"
    masks_dir = args.output / "masks_256"

    for directory in (original_dir, images_dir, masks_dir):
        directory.mkdir(parents=True, exist_ok=True)

    contact_tiles = []

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Could not read {path}")

        # Preserve the original photograph unchanged.
        shutil.copy2(path, original_dir / path.name)

        # Assignment preprocessing: squash to 256x256.
        resized = cv2.resize(
            image,
            (SIZE, SIZE),
            interpolation=cv2.INTER_AREA,
        )

        output_path = images_dir / f"{path.stem}.png"

        if not cv2.imwrite(str(output_path), resized):
            raise RuntimeError(f"Could not save {output_path}")

        print(
            f"{path.name}: {image.shape[1]}x{image.shape[0]}"
            f" -> {SIZE}x{SIZE}"
        )

        # Prepare thumbnails for visual inspection.
        tile = np.full((290, 276, 3), 245, dtype=np.uint8)
        tile[0:256, 10:266] = resized
        cv2.putText(
            tile,
            path.name,
            (10, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        contact_tiles.append(tile)

    # Create a contact sheet of all processed photographs.
    columns = 3
    rows = (len(contact_tiles) + columns - 1) // columns
    sheet = np.full(
        (rows * 290, columns * 276, 3),
        230,
        dtype=np.uint8,
    )

    for index, tile in enumerate(contact_tiles):
        row, col = divmod(index, columns)
        y, x = row * 290, col * 276
        sheet[y:y + 290, x:x + 276] = tile

    contact_path = args.output / "contact_sheet.jpg"
    cv2.imwrite(str(contact_path), sheet)

    print("\nPreprocessing complete.")
    print(f"Images processed: {len(paths)}")
    print(f"Originals: {original_dir}")
    print(f"Resized images: {images_dir}")
    print(f"Empty mask directory: {masks_dir}")
    print(f"Contact sheet: {contact_path}")


if __name__ == "__main__":
    main()
