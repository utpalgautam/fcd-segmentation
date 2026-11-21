"""Binary segmentation metrics for FCD lesion evaluation.

Supports:
- Dice Similarity Coefficient (DSC)
- Sensitivity / Recall
- Specificity
- Precision / Positive Predictive Value (PPV)
- 95th Percentile Hausdorff Distance (HD95) with voxel spacing support
- Robust handling of edge cases (empty target, empty prediction, both empty)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_score(prediction, target, *, empty_score: float = 1.0) -> float:
    """Compute Dice Similarity Coefficient (DSC) between binary masks.

    If both prediction and target are completely empty (no lesion), returns empty_score (default: 1.0).
    If only one is empty while the other is non-empty, returns 0.0.
    """
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape} != target {truth.shape}")

    pred_sum = int(pred.sum())
    truth_sum = int(truth.sum())
    denom = pred_sum + truth_sum

    if denom == 0:
        return float(empty_score)
    if pred_sum == 0 or truth_sum == 0:
        return 0.0

    intersection = int(np.logical_and(pred, truth).sum())
    return float(2.0 * intersection / denom)


def sensitivity(prediction, target, *, empty_score: float = 1.0) -> float:
    """Compute Sensitivity / True Positive Rate / Recall.

    TP / (TP + FN).
    If target is empty (positives == 0):
        If prediction is also empty (no false alarms), returns empty_score (1.0).
        If prediction has false positives, returns 0.0 (or empty_score).
    """
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape} != target {truth.shape}")

    positives = int(truth.sum())
    if positives == 0:
        # No ground-truth lesion: if pred is clean, perfect recall; else false positive
        return float(empty_score) if int(pred.sum()) == 0 else 0.0

    intersection = int(np.logical_and(pred, truth).sum())
    return float(intersection / positives)


# Alias for recall
recall = sensitivity


def specificity(prediction, target, *, empty_score: float = 1.0) -> float:
    """Compute Specificity / True Negative Rate.

    TN / (TN + FP).
    """
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape} != target {truth.shape}")

    negatives = int(np.logical_not(truth).sum())
    if negatives == 0:
        return float(empty_score)

    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(truth)).sum())
    return float(tn / negatives)


def precision(prediction, target, *, empty_score: float = 1.0) -> float:
    """Compute Precision / Positive Predictive Value (PPV).

    TP / (TP + FP).
    If prediction is empty:
        If target is also empty, returns empty_score (1.0).
        If target had lesions (false negative only), returns 0.0.
    """
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape} != target {truth.shape}")

    pred_positives = int(pred.sum())
    if pred_positives == 0:
        return float(empty_score) if int(truth.sum()) == 0 else 0.0

    intersection = int(np.logical_and(pred, truth).sum())
    return float(intersection / pred_positives)


def hd95(
    prediction,
    target,
    voxel_spacing: tuple[float, ...] | list[float] | None = None,
    *,
    empty_score: float = 0.0,
    penalty: float | None = None,
) -> float:
    """Compute 95th Percentile Hausdorff Distance (HD95) between binary masks.

    Parameters
    ----------
    prediction : array-like
        Binary prediction mask.
    target : array-like
        Binary ground-truth mask.
    voxel_spacing : tuple of float, optional
        Voxel spacing in mm along each dimension (e.g. (1.0, 1.0, 1.0)).
    empty_score : float
        Score returned when both masks are empty (default: 0.0 mm).
    penalty : float or None
        Value returned when one mask is empty while the other is non-empty.
        If None, returns float('inf').

    Returns
    -------
    float: 95th percentile distance in physical units (mm) if spacing provided, else voxels.
    """
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape} != target {truth.shape}")

    pred_empty = not np.any(pred)
    truth_empty = not np.any(truth)

    if pred_empty and truth_empty:
        return float(empty_score)

    if pred_empty or truth_empty:
        if penalty is not None:
            return float(penalty)
        return float("inf")

    # Both non-empty: compute surface distance via distance transform
    sampling = voxel_spacing if voxel_spacing is not None else [1.0] * pred.ndim

    # Extract surface / boundary voxels using morphological erosion
    from scipy.ndimage import binary_erosion

    pred_border = np.logical_xor(pred, binary_erosion(pred))
    truth_border = np.logical_xor(truth, binary_erosion(truth))

    # If erosion consumed all (single voxel case), use the mask itself
    if not np.any(pred_border):
        pred_border = pred
    if not np.any(truth_border):
        truth_border = truth

    # Distance map from truth boundary
    dt_truth = distance_transform_edt(~truth_border, sampling=sampling)
    distances_pred_to_truth = dt_truth[pred_border]

    # Distance map from pred boundary
    dt_pred = distance_transform_edt(~pred_border, sampling=sampling)
    distances_truth_to_pred = dt_pred[truth_border]

    all_distances = np.concatenate([distances_pred_to_truth, distances_truth_to_pred])
    if len(all_distances) == 0:
        return 0.0

    return float(np.percentile(all_distances, 95))


def compute_all_metrics(
    prediction,
    target,
    voxel_spacing: tuple[float, ...] | list[float] | None = None,
) -> dict[str, float]:
    """Compute comprehensive metrics dictionary for a single case.

    Returns
    -------
    dict with keys:
        - dice: Dice Similarity Coefficient
        - sensitivity: Recall / TPR
        - specificity: TNR
        - precision: PPV
        - hd95: 95th percentile Hausdorff distance
        - tp: True positive voxel count
        - fp: False positive voxel count
        - fn: False negative voxel count
        - tn: True negative voxel count
        - target_volume_voxels: Ground truth lesion volume
        - pred_volume_voxels: Predicted lesion volume
    """
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)

    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, np.logical_not(truth)).sum())
    fn = int(np.logical_and(np.logical_not(pred), truth).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(truth)).sum())

    dsc = dice_score(pred, truth)
    sens = sensitivity(pred, truth)
    spec = specificity(pred, truth)
    prec = precision(pred, truth)
    h95 = hd95(pred, truth, voxel_spacing=voxel_spacing)

    return {
        "dice": dsc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "hd95": h95,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "target_volume_voxels": float(truth.sum()),
        "pred_volume_voxels": float(pred.sum()),
    }
