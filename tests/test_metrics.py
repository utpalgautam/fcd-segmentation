import numpy as np
import pytest

from fcd_segmentation.evaluation.metrics import (
    compute_all_metrics,
    dice_score,
    hd95,
    precision,
    recall,
    sensitivity,
    specificity,
)


def test_dice_identical():
    mask = np.array([0, 1, 1, 0])
    assert dice_score(mask, mask) == 1.0


def test_empty_dice_policy():
    mask = np.zeros(4)
    assert dice_score(mask, mask) == 1.0
    assert dice_score(mask, mask, empty_score=0.0) == 0.0


def test_dice_one_empty():
    pred = np.zeros(4)
    target = np.array([0, 1, 0, 0])
    assert dice_score(pred, target) == 0.0
    assert dice_score(target, pred) == 0.0


def test_shape_mismatch():
    with pytest.raises(ValueError):
        dice_score(np.zeros(2), np.zeros(3))


def test_sensitivity_and_recall():
    pred = np.array([0, 1, 1])
    target = np.array([1, 1, 0])
    assert sensitivity(pred, target) == 0.5
    assert recall(pred, target) == 0.5


def test_sensitivity_empty_target():
    # When target has no lesions: clean pred gives 1.0; false positives give 0.0
    assert sensitivity(np.zeros(5), np.zeros(5)) == 1.0
    assert sensitivity(np.array([1, 0, 0, 0, 0]), np.zeros(5)) == 0.0


def test_specificity():
    pred = np.array([0, 0, 1, 0])
    target = np.array([0, 0, 0, 1])
    # TN: index 0, 1 -> 2. Negatives in target: index 0, 1, 2 -> 3. Spec = 2/3
    assert abs(specificity(pred, target) - 2 / 3) < 1e-5


def test_precision():
    pred = np.array([1, 1, 0])
    target = np.array([1, 0, 0])
    # TP = 1, FP = 1 -> 1/2
    assert precision(pred, target) == 0.5


def test_precision_empty_prediction():
    # Empty pred with empty target -> 1.0; with non-empty target -> 0.0
    assert precision(np.zeros(4), np.zeros(4)) == 1.0
    assert precision(np.zeros(4), np.array([1, 0, 0, 0])) == 0.0


def test_hd95_identical_masks():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[3:7, 3:7, 3:7] = True
    assert hd95(mask, mask) == 0.0


def test_hd95_both_empty():
    empty = np.zeros((10, 10, 10), dtype=bool)
    assert hd95(empty, empty) == 0.0


def test_hd95_one_empty():
    empty = np.zeros((10, 10, 10), dtype=bool)
    non_empty = np.zeros((10, 10, 10), dtype=bool)
    non_empty[2:4, 2:4, 2:4] = True
    assert hd95(empty, non_empty) == float("inf")
    assert hd95(empty, non_empty, penalty=100.0) == 100.0


def test_hd95_with_voxel_spacing():
    mask1 = np.zeros((20, 20, 20), dtype=bool)
    mask2 = np.zeros((20, 20, 20), dtype=bool)
    mask1[5, 5, 5] = True
    mask2[5, 5, 8] = True  # Shifted by 3 voxels along z

    # Spacing 1mm vs 2mm
    dist_1mm = hd95(mask1, mask2, voxel_spacing=(1.0, 1.0, 1.0))
    dist_2mm = hd95(mask1, mask2, voxel_spacing=(1.0, 1.0, 2.0))

    assert abs(dist_1mm - 3.0) < 1e-4
    assert abs(dist_2mm - 6.0) < 1e-4


def test_compute_all_metrics():
    pred = np.zeros((10, 10, 10), dtype=bool)
    target = np.zeros((10, 10, 10), dtype=bool)
    pred[2:6, 2:6, 2:6] = True
    target[3:7, 2:6, 2:6] = True

    metrics = compute_all_metrics(pred, target, voxel_spacing=(1.0, 1.0, 1.0))
    expected_keys = {
        "dice",
        "sensitivity",
        "specificity",
        "precision",
        "hd95",
        "tp",
        "fp",
        "fn",
        "tn",
        "target_volume_voxels",
        "pred_volume_voxels",
    }
    assert expected_keys.issubset(set(metrics.keys()))
    assert 0.0 < metrics["dice"] < 1.0
    assert metrics["tp"] > 0
