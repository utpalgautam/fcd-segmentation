import numpy as np
import pytest
from fcd_segmentation.preprocessing.normalization import normalize_intensity

def test_normalization_float32_zero_background():
    image = np.zeros((5, 5, 5))
    image[1:4, 1:4, 1:4] = np.arange(27).reshape(3, 3, 3) + 1
    result = normalize_intensity(image)
    assert result.dtype == np.float32
    assert np.all(result[image == 0] == 0)
    assert abs(float(result[image != 0].mean())) < 1e-5

def test_empty_foreground():
    assert not np.count_nonzero(normalize_intensity(np.zeros((3, 3, 3))))

def test_reject_non_3d():
    with pytest.raises(ValueError):
        normalize_intensity(np.zeros((3, 3)))

def test_mask_shape():
    with pytest.raises(ValueError):
        normalize_intensity(np.ones((3, 3, 3)), np.ones((2, 2, 2)))
