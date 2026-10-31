import nibabel as nib
import numpy as np
import pytest

from fcd_segmentation.preprocessing.io import (
    load_nifti,
    load_nifti_data,
    save_nifti,
)


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_save_and_load_round_trip(tmp_path, image, suffix):
    path = tmp_path / f"nested/scan{suffix}"
    returned = save_nifti(image, path)
    assert returned == path
    assert path.is_file()
    loaded = load_nifti(path)
    np.testing.assert_allclose(np.asarray(loaded.dataobj), np.asarray(image.dataobj))
    np.testing.assert_allclose(loaded.affine, image.affine)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_nifti(tmp_path / "missing.nii.gz")


def test_load_invalid_file_raises_value_error(tmp_path):
    path = tmp_path / "bad.nii"
    path.write_text("not a NIfTI file")
    with pytest.raises(ValueError, match="Could not load"):
        load_nifti(path)


def test_load_nifti_data_returns_requested_dtype(tmp_path, image_data, affine):
    path = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(image_data.astype(np.float64), affine), path)
    result = load_nifti_data(path, dtype=np.float32)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, image_data)


def test_load_nifti_data_returns_independent_array(tmp_path, image):
    path = tmp_path / "scan.nii.gz"
    nib.save(image, path)
    result = load_nifti_data(path)
    result[0, 0, 0] = 1234
    assert np.asarray(nib.load(path).dataobj)[0, 0, 0] != 1234


def test_save_refuses_overwrite(tmp_path, image):
    path = tmp_path / "scan.nii.gz"
    save_nifti(image, path)
    with pytest.raises(FileExistsError):
        save_nifti(image, path)


def test_save_allows_overwrite_when_requested(tmp_path, image):
    path = tmp_path / "scan.nii.gz"
    save_nifti(image, path)
    save_nifti(image, path, overwrite=True)
    assert path.is_file()


@pytest.mark.parametrize("name", ["scan.txt", "scan.gz", "scan.nii.gz.bak"])
def test_save_rejects_invalid_extension(tmp_path, image, name):
    with pytest.raises(ValueError, match="Output path|Compressed NIfTI"):
        save_nifti(image, tmp_path / name)


def test_save_converts_dtype(tmp_path, image):
    path = tmp_path / "integer.nii.gz"
    save_nifti(image, path, dtype=np.float64)
    loaded = nib.load(path)
    assert np.asarray(loaded.dataobj).dtype == np.float64


def test_save_rejects_nonfinite_data(tmp_path, affine):
    data = np.ones((3, 3, 3), dtype=np.float32)
    data[0, 0, 0] = np.nan
    image = nib.Nifti1Image(data, affine)
    with pytest.raises(ValueError, match="NaN or infinite"):
        save_nifti(image, tmp_path / "bad.nii.gz")


def test_load_rejects_nonfinite_data_by_default(tmp_path, affine):
    data = np.ones((3, 3, 3), dtype=np.float32)
    data[0, 0, 0] = np.inf
    path = tmp_path / "bad.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), path)
    with pytest.raises(ValueError, match="NaN or infinite"):
        load_nifti(path)


def test_load_can_skip_voxel_validation(tmp_path, affine):
    data = np.ones((3, 3, 3), dtype=np.float32)
    data[0, 0, 0] = np.nan
    path = tmp_path / "bad_voxel.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), path)
    assert load_nifti(path, validate=True, dtype=None).shape == data.shape


def test_save_rejects_non_nifti_object(tmp_path):
    with pytest.raises(ValueError, match="NIfTI"):
        save_nifti(np.zeros((2, 2, 2)), tmp_path / "x.nii.gz")
