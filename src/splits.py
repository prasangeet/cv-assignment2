from torch.utils.data import random_split


def split_dataset(dataset, validation_fraction=0.1, seed=42):
    """Split a dataset deterministically into training and validation sets."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    dataset_size = len(dataset)
    validation_size = max(1, round(dataset_size * validation_fraction))
    training_size = dataset_size - validation_size

    if training_size < 1:
        raise ValueError("Dataset must contain at least two samples")

    generator = __import__("torch").Generator().manual_seed(seed)

    return random_split(
        dataset,
        [training_size, validation_size],
        generator=generator,
    )
