import numpy as np
import pytest

from src.metrics import (
    boundary_f1,
    confusion_matrix,
    mean_boundary_f1,
    mean_iou,
    miou_from_confusion,
)


def test_identical_masks_have_perfect_scores():
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 1

    assert mean_iou(mask, mask) == pytest.approx(1.0)
    assert boundary_f1(mask, mask) == pytest.approx(1.0)


def test_completely_wrong_mask_has_zero_iou():
    target = np.zeros((8, 8), dtype=np.uint8)
    prediction = np.ones((8, 8), dtype=np.uint8)

    # Neither background nor foreground has any intersection.
    assert mean_iou(prediction, target) == pytest.approx(0.0)


def test_confusion_matrix():
    target = np.array([
        [0, 0],
        [1, 1],
    ])

    prediction = np.array([
        [0, 1],
        [1, 0],
    ])

    expected = np.array([
        [1, 1],
        [1, 1],
    ])

    np.testing.assert_array_equal(
        confusion_matrix(prediction, target),
        expected,
    )


def test_miou_from_confusion():
    confusion = np.array([
        [8, 2],
        [1, 9],
    ])

    expected = ((8 / 11) + (9 / 12)) / 2

    assert miou_from_confusion(confusion) == pytest.approx(expected)


def test_empty_masks_have_perfect_boundary_f1():
    empty = np.zeros((16, 16), dtype=np.uint8)

    assert boundary_f1(empty, empty) == pytest.approx(1.0)


def test_one_empty_mask_has_zero_boundary_f1():
    empty = np.zeros((16, 16), dtype=np.uint8)
    full = np.ones((16, 16), dtype=np.uint8)

    assert boundary_f1(full, empty) == pytest.approx(0.0)


def test_boundary_tolerance():
    target = np.zeros((20, 20), dtype=np.uint8)
    prediction = np.zeros_like(target)

    target[5:15, 5:15] = 1
    prediction[5:15, 6:16] = 1

    assert boundary_f1(
        prediction,
        target,
        tolerance=2,
    ) > 0.5


def test_mean_boundary_f1_averages_images():
    empty = np.zeros((8, 8), dtype=np.uint8)
    foreground = np.zeros((8, 8), dtype=np.uint8)
    foreground[2:6, 2:6] = 1

    predictions = np.stack([empty, foreground])
    targets = np.stack([empty, foreground])

    assert mean_boundary_f1(
        predictions,
        targets,
    ) == pytest.approx(1.0)


def test_mismatched_shapes_raise():
    prediction = np.zeros((8, 8), dtype=np.uint8)
    target = np.zeros((7, 8), dtype=np.uint8)

    with pytest.raises(ValueError):
        mean_iou(prediction, target)

    with pytest.raises(ValueError):
        boundary_f1(prediction, target)


def test_empty_dataset_raises():
    predictions = np.empty((0, 8, 8), dtype=np.uint8)
    targets = np.empty((0, 8, 8), dtype=np.uint8)

    with pytest.raises(ValueError):
        mean_boundary_f1(predictions, targets)
