from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


@pytest.mark.parametrize(
    ("split", "expected_count"),
    [
        ("DUTS-TR", 10553),
        ("DUTS-TE", 5019),
    ],
)
def test_dataset_directories_exist(split, expected_count):
    """Each dataset split should contain image and mask directories."""
    split_root = DATA_ROOT / split

    image_dir = split_root / f"{split}-Image"
    mask_dir = split_root / f"{split}-Mask"

    if not split_root.exists():
        pytest.skip(f"{split} dataset is not available")

    assert image_dir.is_dir(), f"Missing image directory: {image_dir}"
    assert mask_dir.is_dir(), f"Missing mask directory: {mask_dir}"


@pytest.mark.parametrize(
    ("split", "expected_count"),
    [
        ("DUTS-TR", 10553),
        ("DUTS-TE", 5019),
    ],
)
def test_dataset_image_and_mask_counts(split, expected_count):
    """Image and mask counts should match the expected dataset size."""
    split_root = DATA_ROOT / split

    image_dir = split_root / f"{split}-Image"
    mask_dir = split_root / f"{split}-Mask"

    if not image_dir.is_dir() or not mask_dir.is_dir():
        pytest.skip(f"{split} dataset directories are not available")

    image_files = [
        path for path in image_dir.iterdir()
        if path.is_file()
    ]

    mask_files = [
        path for path in mask_dir.iterdir()
        if path.is_file()
    ]

    assert len(image_files) == expected_count
    assert len(mask_files) == expected_count


def test_training_and_test_directories_are_separate():
    """Training and test data must use separate directories."""
    train_dir = DATA_ROOT / "DUTS-TR"
    test_dir = DATA_ROOT / "DUTS-TE"

    if not train_dir.exists() or not test_dir.exists():
        pytest.skip("Both dataset splits must be available")

    assert train_dir.resolve() != test_dir.resolve()
