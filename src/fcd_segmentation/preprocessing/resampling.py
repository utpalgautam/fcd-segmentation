"""Resampling utilities for MRI images and segmentation masks."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from fcd_segmentation.preprocessing.geometry import (
    validate_image_geometry,
)


def resample_to_reference(
    image: nib.spatialimages.SpatialImage,
    reference: nib.spatialimages.SpatialImage,
    *,
    is_mask: bool = False,
) -> nib.Nifti1Image:
    """Resample an image onto the spatial grid of a reference image.

    Args:
        image: Source 3D MRI image or segmentation mask.
        reference: Image defining the target shape and affine.
        is_mask: If True, use nearest-neighbour interpolation to preserve
            discrete label values. Otherwise, use linear interpolation.

    Returns:
        A NIfTI image with the reference image's spatial grid.

    Notes:
        MRI intensities use linear interpolation (order=1).
        Segmentation masks use nearest-neighbour interpolation (order=0).
        This function aligns voxel grids but does not perform registration.
    """
    validate_image_geometry(image)
    validate_image_geometry(reference)

    interpolation_order = 0 if is_mask else 1

    source_data = np.asarray(image.dataobj)

    if not np.all(np.isfinite(source_data)):
        raise ValueError("Source image contains NaN or infinite values.")

    resampled = resample_from_to(
        image,
        (reference.shape, reference.affine),
        order=interpolation_order,
        mode="constant",
        cval=0.0,
    )

    output_data = np.asarray(resampled.dataobj)

    if not np.all(np.isfinite(output_data)):
        raise ValueError("Resampled image contains NaN or infinite values.")

    if is_mask:
        # Nearest-neighbour interpolation should preserve label values.
        # Restore the source data type to avoid unintended floating labels.
        output_data = np.rint(output_data).astype(source_data.dtype)

    header = reference.header.copy()
    header.set_data_dtype(output_data.dtype)

    return nib.Nifti1Image(
        output_data,
        reference.affine,
        header=header,
    )


def resample_to_spacing(
    image: nib.spatialimages.SpatialImage,
    target_spacing: Sequence[float],
    *,
    is_mask: bool = False,
) -> nib.Nifti1Image:
    """Resample a 3D image to a target voxel spacing in millimetres.

    The image's physical orientation and approximate field of view are
    preserved. The output grid is constructed from the source image's
    orientation, origin, and requested voxel spacing.

    Args:
        image: Source 3D MRI image or segmentation mask.
        target_spacing: Three positive target voxel sizes in millimetres,
            e.g. (1.0, 1.0, 1.0).
        is_mask: If True, use nearest-neighbour interpolation.

    Returns:
        A NIfTI image with the requested voxel spacing.

    Important:
        This changes the voxel grid, not the anatomical registration
        between modalities. Use resample_to_reference when the output must
        match a particular reference image exactly.
    """
    validate_image_geometry(image)

    spacing = np.asarray(target_spacing, dtype=np.float64)

    if spacing.shape != (3,):
        raise ValueError(
            "target_spacing must contain exactly three values."
        )

    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError(
            f"target_spacing must contain three positive finite values; "
            f"received {target_spacing}."
        )

    source_spacing = np.asarray(
        nib.affines.voxel_sizes(image.affine),
        dtype=np.float64,
    )

    source_shape = np.asarray(image.shape, dtype=np.float64)

    # Approximate physical extent using the distance between the first
    # and last voxel centres along each image axis.
    new_shape = np.maximum(
        1,
        np.round(
            (source_shape - 1.0) * source_spacing / spacing
        ).astype(np.int64) + 1,
    )

    source_affine = np.asarray(image.affine, dtype=np.float64)
    target_affine = source_affine.copy()

    # Preserve each axis direction while changing voxel size.
    for axis in range(3):
        direction = (
            source_affine[:3, axis] / source_spacing[axis]
        )
        target_affine[:3, axis] = direction * spacing[axis]

    source_data = np.asarray(image.dataobj)

    if not np.all(np.isfinite(source_data)):
        raise ValueError("Source image contains NaN or infinite values.")

    interpolation_order = 0 if is_mask else 1

    resampled = resample_from_to(
        image,
        (tuple(int(value) for value in new_shape), target_affine),
        order=interpolation_order,
        mode="constant",
        cval=0.0,
    )

    output_data = np.asarray(resampled.dataobj)

    if not np.all(np.isfinite(output_data)):
        raise ValueError("Resampled image contains NaN or infinite values.")

    if is_mask:
        output_data = np.rint(output_data).astype(source_data.dtype)

    header = image.header.copy()
    header.set_data_dtype(output_data.dtype)
    header.set_zooms(tuple(float(value) for value in spacing))

    return nib.Nifti1Image(
        output_data,
        target_affine,
        header=header,
    )


def resample_file_to_reference(
    input_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    *,
    is_mask: bool = False,
) -> Path:
    """Resample a NIfTI file to a reference image and save the result.

    Args:
        input_path: Path to the source NIfTI image.
        reference_path: Path to the reference NIfTI image.
        output_path: Destination path for the resampled NIfTI image.
        is_mask: Whether the source file contains discrete segmentation
            labels.

    Returns:
        The destination path.
    """
    input_path = Path(input_path)
    reference_path = Path(reference_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Reference image not found: {reference_path}"
        )

    image = nib.load(str(input_path))
    reference = nib.load(str(reference_path))

    result = resample_to_reference(
        image,
        reference,
        is_mask=is_mask,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, str(output_path))

    return output_path


def resample_file_to_spacing(
    input_path: str | Path,
    output_path: str | Path,
    target_spacing: Sequence[float],
    *,
    is_mask: bool = False,
) -> Path:
    """Resample a NIfTI file to a target spacing and save the result.

    Args:
        input_path: Path to the source NIfTI image.
        output_path: Destination path.
        target_spacing: Desired voxel spacing in millimetres.
        is_mask: Whether the source file contains discrete labels.

    Returns:
        The destination path.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    image = nib.load(str(input_path))

    result = resample_to_spacing(
        image,
        target_spacing,
        is_mask=is_mask,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, str(output_path))

    return output_path