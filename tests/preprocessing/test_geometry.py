import nibabel as nib
import numpy as np
import pytest

from fcd_segmentation.preprocessing.geometry import (
    maximum_corner_displacement_mm,
    same_grid,
    validate_3d_image,
    validate_image_and_mask,
    validate_image_geometry,
)


def test_valid_geometry_passes(image):
    assert validate_image_geometry(image) is None


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4, 1), (1,), ()])
def test_non_3d_image_is_rejected(shape):
    image = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4))
    with pytest.raises(ValueError, match="3D"):
        validate_image_geometry(image)


def test_empty_dimension_is_rejected():
    with pytest.raises(ValueError):
        image = nib.Nifti1Image(
            np.empty((0, 2, 2), dtype=np.float32),
            np.eye(4),
        )
        validate_image_geometry(image)


def test_singular_affine_is_rejected():
    image = nib.Nifti1Image(
        np.ones((2, 2, 2), dtype=np.float32),
        np.eye(4),
    )

    singular_affine = np.eye(4)
    singular_affine[2, :3] = 0

    # Bypass NiBabel's constructor validation so that our
    # geometry validator receives the singular affine.
    image._affine = singular_affine

    with pytest.raises(ValueError):
        validate_image_geometry(image)


def test_nonfinite_affine_is_rejected():
    affine = np.eye(4)
    affine[0, 3] = np.nan
    image = nib.Nifti1Image(np.ones((2, 2, 2)), affine)
    with pytest.raises(ValueError, match="non-finite"):
        validate_image_geometry(image)


def test_same_grid_for_equal_grids(image):
    other = nib.Nifti1Image(np.ones(image.shape), image.affine.copy())
    assert same_grid(image, other)


def test_same_grid_ignores_voxel_values(image):
    other = nib.Nifti1Image(
        np.full(image.shape, -99, dtype=np.float32),
        image.affine.copy(),
    )

    assert same_grid(image, other)


def test_same_grid_detects_shape_difference(image):
    other = nib.Nifti1Image(np.zeros((3, 5, 6)), image.affine)
    assert not same_grid(image, other)


def test_same_grid_detects_affine_difference(image):
    affine = image.affine.copy()
    affine[0, 3] += 1
    other = nib.Nifti1Image(np.zeros(image.shape), affine)
    assert not same_grid(image, other)


def test_same_grid_tolerance_must_be_nonnegative(image):
    with pytest.raises(ValueError, match="non-negative"):
        same_grid(image, image, tolerance_mm=-0.1)


def test_corner_displacement_is_zero_for_same_affine(image):
    other = nib.Nifti1Image(np.zeros(image.shape), image.affine)
    assert maximum_corner_displacement_mm(image, other) == pytest.approx(0)


def test_corner_displacement_rejects_different_shapes(image):
    other = nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4))
    with pytest.raises(ValueError, match="shapes differ"):
        maximum_corner_displacement_mm(image, other)


def test_validate_image_and_mask_accepts_matching_grid(image, mask):
    validate_image_and_mask(image, mask)


def test_validate_image_and_mask_rejects_shape_mismatch(image):
    other = nib.Nifti1Image(np.zeros((2, 2, 2)), image.affine)
    with pytest.raises(ValueError, match="shapes differ"):
        validate_image_and_mask(image, other)


def test_validate_image_and_mask_rejects_grid_mismatch(image, mask):
    affine = mask.affine.copy()
    affine[0, 3] += 0.5
    shifted = nib.Nifti1Image(np.asarray(mask.dataobj), affine)
    with pytest.raises(ValueError, match="do not match"):
        validate_image_and_mask(image, shifted)

def test_validate_3d_image_compatibility_wrapper():
    valid_image = nib.Nifti1Image(
        np.zeros((2, 3, 4), dtype=np.float32),
        np.eye(4),
    )
    validate_3d_image(valid_image)

    invalid_image = nib.Nifti1Image(
        np.zeros((2, 3), dtype=np.float32),
        np.eye(4),
    )
    with pytest.raises(ValueError):
        validate_3d_image(invalid_image)
