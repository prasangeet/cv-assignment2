import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_SIZE = 256


class DUTSDataset(Dataset):
    """DUTS dataset for binary salient-object segmentation."""

    def __init__(self, root: str | Path, split: str = "DUTS-TR"):
        self.root = Path(root)
        self.split = split

        if split not in {"DUTS-TR", "DUTS-TE"}:
            raise ValueError(f"Unsupported dataset split: {split}")

        self.image_dir = self.root / split / f"{split}-Image"
        self.mask_dir = self.root / split / f"{split}-Mask"

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        if not self.mask_dir.is_dir():
            raise FileNotFoundError(
                f"Mask directory not found: {self.mask_dir}"
            )

        image_paths = sorted(
            p for p in self.image_dir.iterdir() if p.is_file()
        )
        mask_paths = sorted(
            p for p in self.mask_dir.iterdir() if p.is_file()
        )

        masks_by_stem = {p.stem: p for p in mask_paths}

        self.samples = [
            (image_path, masks_by_stem[image_path.stem])
            for image_path in image_paths
            if image_path.stem in masks_by_stem
        ]

        if not self.samples:
            raise RuntimeError(
                f"No matching image-mask pairs found for {split}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        mask = cv2.resize(
            mask,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

        mask = (mask >= 128).astype(np.float32)

        image = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        ).float() / 255.0

        mask = torch.from_numpy(
            np.ascontiguousarray(mask[None, :, :])
        ).float()

        return image, mask
