import numpy as np
import pytest

from fcd_segmentation.preprocessing.geometry import same_grid, validate_3d_image

nib = pytest.importorskip("nibabel")


def test_same_grid():
    a = nib.Nifti1Image(np.zeros((3, 4, 5)), np.eye(4))
    b = nib.Nifti1Image(np.ones((3, 4, 5)), np.eye(4))
    c = nib.Nifti1Image(np.ones((3, 4, 5)), np.diag([2, 1, 1, 1]))
    assert same_grid(a, b)
    assert not same_grid(a, c)


def test_validate_3d():
    validate_3d_image(nib.Nifti1Image(np.zeros((2, 3, 4)), np.eye(4)))
    with pytest.raises(ValueError):
        validate_3d_image(nib.Nifti1Image(np.zeros((2, 3)), np.eye(4)))
