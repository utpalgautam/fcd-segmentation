import nibabel as nib
import numpy as np
import pytest

from fcd_segmentation.preprocessing.normalization import (
    normalize_image,
    normalize_intensity,
    normalize_nifti,
)


def test_normalize_output_is_float32_and_same_shape():
    data = np.arange(1, 65, dtype=np.float64).reshape(4, 4, 4)
    result = normalize_intensity(data, clip_percentiles=None)
    assert result.dtype == np.float32
    assert result.shape == data.shape


def test_normalize_foreground_has_zero_mean_and_unit_std():
    data = np.arange(1, 65, dtype=np.float32).reshape(4, 4, 4)
    result = normalize_intensity(data, clip_percentiles=None)
    assert np.mean(result) == pytest.approx(0, abs=1e-6)
    assert np.std(result) == pytest.approx(1, abs=1e-6)


def test_background_stays_zero():
    data = np.zeros((4, 4, 4), dtype=np.float32)
    data[1:, 1:, 1:] = np.arange(1, 28).reshape(3, 3, 3)
    result = normalize_intensity(data, clip_percentiles=None)
    assert np.all(result[data == 0] == 0)


def test_explicit_mask_controls_foreground():
    data = np.arange(1, 65, dtype=np.float32).reshape(4, 4, 4)
    mask = np.zeros(data.shape, dtype=np.uint8)
    mask[:2] = 1
    result = normalize_intensity(data, mask=mask, clip_percentiles=None)
    assert np.all(result[mask == 0] == 0)
    assert np.mean(result[mask != 0]) == pytest.approx(0, abs=1e-6)


def test_input_is_not_modified():
    data = np.arange(1, 65, dtype=np.float32).reshape(4, 4, 4)
    original = data.copy()
    normalize_intensity(data, clip_percentiles=None)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("data", [np.zeros((3, 3, 3)), np.ones((3, 3, 3))])
def test_no_foreground_or_constant_foreground_raises(data):
    with pytest.raises(ValueError):
        normalize_intensity(data)


@pytest.mark.parametrize("data", [np.ones((3, 3)), np.ones((2, 2, 2, 2)), np.array([])])
def test_invalid_dimensions_or_empty_input_raises(data):
    with pytest.raises(ValueError):
        normalize_intensity(data)


def test_mask_shape_mismatch_raises():
    with pytest.raises(ValueError, match="identical shapes"):
        normalize_intensity(np.arange(27).reshape(3, 3, 3), mask=np.ones((2, 2, 2)))


@pytest.mark.parametrize("eps", [0, -1, np.nan, np.inf])
def test_invalid_eps_raises(eps):
    data = np.arange(1, 28, dtype=np.float32).reshape(3, 3, 3)
    with pytest.raises(ValueError, match="eps"):
        normalize_intensity(data, eps=eps, clip_percentiles=None)


@pytest.mark.parametrize("percentiles", [(99, 1), (-1, 99), (0, 101), (1,), (1, 2, 3)])
def test_invalid_percentiles_raise(percentiles):
    data = np.arange(1, 28, dtype=np.float32).reshape(3, 3, 3)
    with pytest.raises(ValueError):
        normalize_intensity(data, clip_percentiles=percentiles)


def test_nonfinite_voxels_are_excluded_from_foreground():
    data = np.arange(1, 28, dtype=np.float32).reshape(3, 3, 3)
    data[0, 0, 0] = np.nan
    result = normalize_intensity(data, clip_percentiles=None)
    assert np.isfinite(result).all()
    assert result[0, 0, 0] == 0


def test_normalize_image_preserves_affine(image):
    result = normalize_image(image, clip_percentiles=None)
    np.testing.assert_allclose(result.affine, image.affine)
    assert result.shape == image.shape
    assert np.asarray(result.dataobj).dtype == np.float32


def test_normalize_nifti_file_round_trip(tmp_path, image):
    source = tmp_path / "source.nii.gz"
    output = tmp_path / "nested/normalized.nii.gz"
    nib.save(image, source)
    result_path = normalize_nifti(source, output, clip_percentiles=None)
    assert result_path == output
    assert output.is_file()
    result = nib.load(output)
    assert result.shape == image.shape
    np.testing.assert_allclose(result.affine, image.affine)


def test_normalize_nifti_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        normalize_nifti(tmp_path / "missing.nii.gz", tmp_path / "out.nii.gz")


def test_normalize_nifti_mask_shape_mismatch_raises(tmp_path, image):
    source = tmp_path / "source.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(image, source)
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), mask_path)
    with pytest.raises(ValueError):
        normalize_nifti(source, tmp_path / "out.nii.gz", mask_path=mask_path)
