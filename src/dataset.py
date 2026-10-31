"""DUTS dataset with deterministic image-mask pairing and train-only augmentation."""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_SIZE = 256
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class DUTSDataset(Dataset):
    """Loads one DUTS split and returns RGB images and binary masks."""

    def __init__(
        self,
        root: str | Path,
        split: str = "DUTS-TR",
        image_size: int = IMAGE_SIZE,
        augment: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = int(image_size)
        self.augment = bool(augment)

        if split not in {"DUTS-TR", "DUTS-TE"}:
            raise ValueError(f"Unsupported split: {split}")
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32")

        self.image_dir = self.root / split / f"{split}-Image"
        self.mask_dir = self.root / split / f"{split}-Mask"
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        images = sorted(
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )
        masks = {
            p.stem: p for p in self.mask_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".png"
        }
        missing = [p.name for p in images if p.stem not in masks]
        if missing:
            raise RuntimeError(
                f"{len(missing)} images have no matching masks; examples: {missing[:5]}"
            )
        self.samples = [(p, masks[p.stem]) for p in images]
        if not self.samples:
            raise RuntimeError(f"No image-mask pairs found in {split}")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Geometric transforms are always applied to image and mask together.
        if random.random() < 0.5:
            image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
        if random.random() < 0.10:
            image, mask = cv2.flip(image, 0), cv2.flip(mask, 0)
        if random.random() < 0.5:
            h, w = mask.shape
            scale = random.uniform(0.90, 1.10)
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
            image = cv2.warpAffine(
                image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            mask = cv2.warpAffine(
                mask, matrix, (w, h), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )

        # Photometric transforms affect only the image.
        image = image.astype(np.float32)
        image = image * random.uniform(0.90, 1.10) + random.uniform(-12.0, 12.0)
        gains = np.array(
            [random.uniform(0.95, 1.05) for _ in range(3)], dtype=np.float32
        ).reshape(1, 1, 3)
        image = np.clip(image * gains, 0, 255).astype(np.uint8)
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        image = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )
        mask = cv2.resize(
            mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
        )
        if self.augment:
            image, mask = self._augment(image, mask)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask = (mask >= 128).astype(np.float32)
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask[None, :, :]))
        return image_tensor, mask_tensor
