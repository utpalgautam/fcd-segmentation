import nibabel as nib
import numpy as np
import pytest

from fcd_segmentation.preprocessing.resampling import (
    resample_file_to_reference,
    resample_file_to_spacing,
    resample_to_reference,
    resample_to_spacing,
)


def test_resample_to_reference_matches_target_grid(image):
    affine = image.affine.copy()
    affine[:3, 3] += [0.5, 0, 0]
    reference = nib.Nifti1Image(np.zeros((5, 6, 7)), affine)
    result = resample_to_reference(image, reference)
    assert result.shape == reference.shape
    np.testing.assert_allclose(result.affine, reference.affine)


def test_mask_resampling_preserves_discrete_labels(mask):
    affine = mask.affine.copy()
    affine[0, 3] += 0.25
    reference = nib.Nifti1Image(np.zeros(mask.shape), affine)
    result = resample_to_reference(mask, reference, is_mask=True)
    assert set(np.unique(np.asarray(result.dataobj))).issubset({0, 1})
    assert np.asarray(result.dataobj).dtype == np.asarray(mask.dataobj).dtype


def test_image_resampling_is_finite(image):
    result = resample_to_reference(image, image)
    assert np.isfinite(np.asarray(result.dataobj)).all()


def test_resample_to_spacing_has_requested_voxel_sizes(image):
    result = resample_to_spacing(image, (1.0, 1.0, 1.0))
    np.testing.assert_allclose(nib.affines.voxel_sizes(result.affine), [1, 1, 1])
    assert result.ndim == 3


@pytest.mark.parametrize(
    "spacing",
    [(1, 1), (1, 1, 1, 1), (0, 1, 1), (-1, 1, 1), (np.nan, 1, 1), (np.inf, 1, 1)],
)
def test_invalid_target_spacing_raises(image, spacing):
    with pytest.raises(ValueError):
        resample_to_spacing(image, spacing)


def test_resample_to_spacing_mask_preserves_labels(mask):
    result = resample_to_spacing(mask, (1.0, 1.0, 1.0), is_mask=True)
    assert set(np.unique(np.asarray(result.dataobj))).issubset({0, 1})
    assert np.asarray(result.dataobj).dtype == np.asarray(mask.dataobj).dtype


def test_resampling_rejects_nonfinite_source(image):
    data = np.asarray(image.dataobj).copy()
    data[0, 0, 0] = np.nan
    bad = nib.Nifti1Image(data, image.affine)
    with pytest.raises(ValueError, match="NaN or infinite"):
        resample_to_spacing(bad, (1, 1, 1))


def test_file_resample_to_reference(tmp_path, image):
    source = tmp_path / "source.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    output = tmp_path / "nested/output.nii.gz"
    nib.save(image, source)
    nib.save(nib.Nifti1Image(np.zeros((5, 5, 5)), np.eye(4)), reference)
    assert resample_file_to_reference(source, reference, output) == output
    assert nib.load(output).shape == (5, 5, 5)


def test_file_resample_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resample_file_to_reference(
            tmp_path / "missing.nii.gz",
            tmp_path / "reference.nii.gz",
            tmp_path / "out.nii.gz",
        )


def test_file_resample_to_spacing(tmp_path, image):
    source = tmp_path / "source.nii.gz"
    output = tmp_path / "nested/output.nii.gz"
    nib.save(image, source)
    assert resample_file_to_spacing(source, output, (1, 1, 1)) == output
    assert output.is_file()
