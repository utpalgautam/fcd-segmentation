"""Basic binary segmentation metrics."""

import numpy as np


def dice_score(prediction, target, *, empty_score=1.0):
    pred, truth = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError("Prediction and target must have the same shape")
    denom = int(pred.sum()) + int(truth.sum())
    if denom == 0:
        return float(empty_score)
    return float(2 * np.logical_and(pred, truth).sum() / denom)


def sensitivity(prediction, target):
    pred, truth = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError("Prediction and target must have the same shape")
    positives = int(truth.sum())
    if positives == 0:
        return 1.0
    return float(np.logical_and(pred, truth).sum() / positives)
