"""Spatial geometry utilities for MRI preprocessing."""

from __future__ import annotations

from itertools import product

import nibabel as nib
import numpy as np


def validate_image_geometry(
    image: nib.spatialimages.SpatialImage,
) -> None:
    """Validate the basic geometry of a 3D medical image.

    Raises:
        ValueError: If the image has invalid dimensions or spatial geometry.
    """
    if len(image.shape) != 3:
        raise ValueError(
            f"Expected a 3D image, received shape {image.shape}."
        )

    if any(size <= 0 for size in image.shape):
        raise ValueError(f"Image has an empty dimension: {image.shape}.")

    affine = np.asarray(image.affine, dtype=np.float64)

    if affine.shape != (4, 4):
        raise ValueError(
            f"Expected a 4x4 affine matrix, received {affine.shape}."
        )

    if not np.all(np.isfinite(affine)):
        raise ValueError("Image affine contains non-finite values.")

    linear = affine[:3, :3]

    if not np.all(np.isfinite(np.linalg.det(linear))):
        raise ValueError("Image affine has an invalid determinant.")

    if abs(np.linalg.det(linear)) < 1e-8:
        raise ValueError("Image affine is singular.")

    voxel_sizes = nib.affines.voxel_sizes(affine)

    if not np.all(np.isfinite(voxel_sizes)):
        raise ValueError("Voxel sizes contain non-finite values.")

    if np.any(voxel_sizes <= 0):
        raise ValueError(f"Invalid voxel sizes: {voxel_sizes}.")


def maximum_corner_displacement_mm(
    first: nib.spatialimages.SpatialImage,
    second: nib.spatialimages.SpatialImage,
) -> float:
    """Measure maximum physical displacement at corresponding volume corners.

    Both images must have the same array shape. The returned displacement
    is expressed in millimetres when the NIfTI spatial units are millimetres.

    Note:
        This function compares the images' affine mappings. It does not
        establish anatomical registration between independently acquired
        images.
    """
    validate_image_geometry(first)
    validate_image_geometry(second)

    if first.shape != second.shape:
        raise ValueError(
            "Cannot compare corresponding voxel corners when shapes differ: "
            f"{first.shape} versus {second.shape}."
        )

    shape = first.shape

    corners = np.asarray(
        list(
            product(
                (0, shape[0] - 1),
                (0, shape[1] - 1),
                (0, shape[2] - 1),
            )
        ),
        dtype=np.float64,
    )

    homogeneous = np.column_stack(
        (corners, np.ones(len(corners), dtype=np.float64))
    ).T

    first_world = first.affine @ homogeneous
    second_world = second.affine @ homogeneous

    displacements = np.linalg.norm(
        first_world[:3] - second_world[:3],
        axis=0,
    )

    return float(np.max(displacements))


def same_grid(
    first: nib.spatialimages.SpatialImage,
    second: nib.spatialimages.SpatialImage,
    *,
    tolerance_mm: float = 0.01,
) -> bool:
    """Check whether two images use approximately the same voxel grid.

    Images must have identical array shapes, and their corresponding voxel
    coordinates must differ by no more than tolerance_mm at the volume
    corners.

    This is a grid comparison, not a registration algorithm.
    """
    if tolerance_mm < 0:
        raise ValueError("tolerance_mm must be non-negative.")

    validate_image_geometry(first)
    validate_image_geometry(second)

    if first.shape != second.shape:
        return False

    displacement = maximum_corner_displacement_mm(first, second)

    return displacement <= tolerance_mm


def validate_image_and_mask(
    image: nib.spatialimages.SpatialImage,
    mask: nib.spatialimages.SpatialImage,
    *,
    tolerance_mm: float = 0.01,
) -> None:
    """Validate that an MRI image and its mask share approximately one grid.

    Raises:
        ValueError: If the geometry is invalid or the grids differ.
    """
    validate_image_geometry(image)
    validate_image_geometry(mask)

    if image.shape != mask.shape:
        raise ValueError(
            "Image and mask shapes differ: "
            f"{image.shape} versus {mask.shape}."
        )

    displacement = maximum_corner_displacement_mm(image, mask)

    if displacement > tolerance_mm:
        raise ValueError(
            "Image and mask spatial grids do not match within tolerance. "
            f"Maximum corner displacement: {displacement:.6f} mm; "
            f"allowed: {tolerance_mm:.6f} mm."
        )

def validate_3d_image(
    image: nib.spatialimages.SpatialImage,
) -> None:
    """Validate that an image is a valid 3D medical image.

    Compatibility wrapper for validate_image_geometry().
    Raises:
        ValueError: If the image dimensions or spatial geometry are invalid.
    """
    validate_image_geometry(image)
