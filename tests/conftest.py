"""Shared synthetic fixtures for the test suite; never loads private study data."""

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def affine():
    return np.diag([1.5, 2.0, 2.5, 1.0])


@pytest.fixture
def image_data():
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    data[0, 0, 0] = 0
    return data


@pytest.fixture
def image(image_data, affine):
    return nib.Nifti1Image(image_data.copy(), affine)


@pytest.fixture
def mask(image_data, affine):
    data = np.zeros(image_data.shape, dtype=np.uint8)
    data[1:3, 1:4, 1:5] = 1
    return nib.Nifti1Image(data, affine)


@pytest.fixture
def write_nifti(tmp_path):
    def _write(name, data, affine=None):
        path = tmp_path / name
        nifti = nib.Nifti1Image(
            np.asarray(data),
            np.eye(4) if affine is None else affine,
        )
        nib.save(nifti, path)
        return path

    return _write
