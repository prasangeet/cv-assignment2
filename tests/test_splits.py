import pytest
from torch.utils.data import TensorDataset

import torch

from src.splits import split_dataset


def make_dataset(size):
    return TensorDataset(torch.arange(size))


def test_split_has_expected_sizes():
    dataset = make_dataset(1000)

    train_set, val_set = split_dataset(
        dataset,
        validation_fraction=0.1,
        seed=42,
    )

    assert len(train_set) == 900
    assert len(val_set) == 100


def test_split_is_deterministic():
    dataset = make_dataset(100)

    train_a, val_a = split_dataset(dataset, seed=42)
    train_b, val_b = split_dataset(dataset, seed=42)

    assert train_a.indices == train_b.indices
    assert val_a.indices == val_b.indices


def test_split_contains_every_sample_once():
    dataset = make_dataset(100)

    train_set, val_set = split_dataset(dataset, seed=42)

    combined = list(train_set.indices) + list(val_set.indices)

    assert len(combined) == len(dataset)
    assert len(set(combined)) == len(dataset)
    assert set(combined) == set(range(len(dataset)))


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_invalid_validation_fraction_raises(fraction):
    dataset = make_dataset(100)

    with pytest.raises(ValueError):
        split_dataset(dataset, validation_fraction=fraction)


def test_too_small_dataset_raises():
    dataset = make_dataset(1)

    with pytest.raises(ValueError):
        split_dataset(dataset)
