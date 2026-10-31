import pytest
import torch

from models.segmentation_cnn import SegmentationCNN


@pytest.fixture(scope="session")
def model():
    """Create a model for the test session."""
    return SegmentationCNN()


@pytest.fixture
def sample_image():
    """Return a synthetic RGB image tensor."""
    return torch.rand(1, 3, 256, 256)


@pytest.fixture
def small_image_batch():
    """Return a small batch for shape and batching tests."""
    return torch.rand(2, 3, 256, 256)
