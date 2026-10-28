import numpy as np
import pytest
from fcd_segmentation.evaluation.metrics import dice_score, sensitivity

def test_dice_identical():
    mask = np.array([0, 1, 1, 0])
    assert dice_score(mask, mask) == 1.0

def test_empty_dice_policy():
    mask = np.zeros(4)
    assert dice_score(mask, mask) == 1.0
    assert dice_score(mask, mask, empty_score=0.0) == 0.0

def test_shape_mismatch():
    with pytest.raises(ValueError):
        dice_score(np.zeros(2), np.zeros(3))

def test_sensitivity():
    assert sensitivity(np.array([0, 1, 1]), np.array([1, 1, 0])) == 0.5
