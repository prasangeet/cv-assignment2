import numpy as np


def _to_binary_mask(mask):
    """Convert a mask to a 2D Boolean array."""
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask, received shape {mask.shape}"
        )

    return mask > 0


def _boundary_map(mask):
    """
    Identify foreground pixels having at least one background
    neighbor in the four cardinal directions.

    Pixels outside the image are treated as background.
    """
    mask = _to_binary_mask(mask)

    padded = np.pad(
        mask,
        pad_width=1,
        mode="constant",
        constant_values=False,
    )

    center = padded[1:-1, 1:-1]

    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]

    return center & ~(up & down & left & right)


def _has_neighbor_within_tolerance(source, tolerance):
    """
    Mark pixels having at least one True source pixel within
    a square window of side 2*tolerance + 1.

    This uses array slicing rather than prohibited morphology
    functions from OpenCV or other libraries.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    height, width = source.shape
    padded = np.pad(
        source,
        pad_width=tolerance,
        mode="constant",
        constant_values=False,
    )

    result = np.zeros((height, width), dtype=bool)

    for dy in range(2 * tolerance + 1):
        for dx in range(2 * tolerance + 1):
            result |= padded[
                dy:dy + height,
                dx:dx + width,
            ]

    return result


def confusion_matrix(prediction, target):
    """
    Compute a 2x2 confusion matrix for binary segmentation.

    Rows represent ground-truth classes; columns represent
    predicted classes. Class 0 is background and class 1
    is foreground.
    """
    prediction = _to_binary_mask(prediction)
    target = _to_binary_mask(target)

    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")

    target_flat = target.ravel().astype(np.int64)
    prediction_flat = prediction.ravel().astype(np.int64)

    return np.bincount(
        2 * target_flat + prediction_flat,
        minlength=4,
    ).reshape(2, 2)


def miou_from_confusion(confusion):
    """Compute mean IoU across background and foreground."""
    confusion = np.asarray(confusion, dtype=np.int64)

    if confusion.shape != (2, 2):
        raise ValueError("Confusion matrix must have shape (2, 2)")

    intersection = np.diag(confusion)
    union = (
        confusion.sum(axis=0)
        + confusion.sum(axis=1)
        - intersection
    )

    iou = np.divide(
        intersection,
        union,
        out=np.zeros(2, dtype=np.float64),
        where=union > 0,
    )

    return float(iou.mean())


def mean_iou(predictions, targets):
    """
    Compute dataset-level mIoU using one accumulated confusion matrix.

    Accepts either one HxW mask or a batch of NxHxW masks.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have matching shapes")

    if predictions.ndim == 2:
        predictions = predictions[None, ...]
        targets = targets[None, ...]
    elif predictions.ndim != 3:
        raise ValueError("Expected HxW or NxHxW masks")

    total_confusion = np.zeros((2, 2), dtype=np.int64)

    for prediction, target in zip(predictions, targets):
        total_confusion += confusion_matrix(prediction, target)

    return miou_from_confusion(total_confusion)


def boundary_f1(prediction, target, tolerance=2):
    """
    Compute Boundary F1 for one image.

    A boundary pixel is a foreground pixel with at least one
    background neighbor in the up/down/left/right directions.

    A predicted boundary matches if a ground-truth boundary
    exists within a (2*tolerance + 1) square neighborhood,
    and vice versa.
    """
    prediction = _to_binary_mask(prediction)
    target = _to_binary_mask(target)

    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")

    pred_boundary = _boundary_map(prediction)
    target_boundary = _boundary_map(target)

    pred_count = int(pred_boundary.sum())
    target_count = int(target_boundary.sum())

    if pred_count == 0 and target_count == 0:
        return 1.0

    if pred_count == 0 or target_count == 0:
        return 0.0

    target_neighborhood = _has_neighbor_within_tolerance(
        target_boundary,
        tolerance,
    )
    pred_neighborhood = _has_neighbor_within_tolerance(
        pred_boundary,
        tolerance,
    )

    matched_prediction = int(
        (pred_boundary & target_neighborhood).sum()
    )
    matched_target = int(
        (target_boundary & pred_neighborhood).sum()
    )

    precision = matched_prediction / pred_count
    recall = matched_target / target_count

    if precision + recall == 0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


def mean_boundary_f1(predictions, targets, tolerance=2):
    """
    Average per-image Boundary F1.

    Accepts either one HxW mask or a batch of NxHxW masks.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have matching shapes")

    if predictions.ndim == 2:
        predictions = predictions[None, ...]
        targets = targets[None, ...]
    elif predictions.ndim != 3:
        raise ValueError("Expected HxW or NxHxW masks")

    if len(predictions) == 0:
        raise ValueError("Cannot evaluate an empty dataset")

    scores = [
        boundary_f1(prediction, target, tolerance)
        for prediction, target in zip(predictions, targets)
    ]

    return float(np.mean(scores))
