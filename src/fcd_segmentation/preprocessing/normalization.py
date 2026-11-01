"""
Intensity normalization utilities for FCD MRI segmentation.

Supports NumPy arrays and NIfTI images. The default method performs
foreground-based percentile clipping followed by z-score normalization.

Background voxels are kept at zero. Input images are never modified
in place.
"""

from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
from numpy.typing import ArrayLike, NDArray


def normalize_intensity(
    data: ArrayLike,
    mask: ArrayLike | None = None,
    clip_percentiles: Sequence[float] | None = (0.5, 99.5),
    eps: float = 1e-8,
    allow_constant: bool = False,
) -> NDArray[np.float32]:
    """
    Normalize MRI intensities using foreground-based z-score normalization.

    Parameters
    ----------
    data:
        Input 3D MRI array. The input is not modified.
    mask:
        Optional foreground mask. Nonzero voxels are treated as foreground.
        If omitted, finite, nonzero input voxels define the foreground.
    clip_percentiles:
        Optional lower and upper percentiles for clipping foreground
        intensities before calculating the mean and standard deviation.
        Use None to disable clipping.
    eps:
        Minimum standard deviation accepted for normalization.
    allow_constant:
        If True, returns an all-zero array without error when foreground is
        empty or has near-zero variance. If False, raises ValueError.

    Returns
    -------
    numpy.ndarray
        Float32 normalized array. Background voxels are zero.

    Raises
    ------
    ValueError
        If the input is not 3D, contains no usable foreground, the
        foreground has effectively constant intensity, or parameters
        are invalid.
    """
    array = np.asarray(data, dtype=np.float32)

    if array.ndim != 3:
        raise ValueError(
            f"Expected a 3D MRI volume, received an array with "
            f"{array.ndim} dimensions."
        )

    if array.size == 0:
        raise ValueError("Cannot normalize an empty MRI volume.")

    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite positive number.")

    # Work on a copy so that the input is never modified.
    output = array.copy()

    finite_mask = np.isfinite(array)

    # Replace invalid intensities before any statistical calculations.
    output[~finite_mask] = 0.0

    if mask is None:
        foreground = finite_mask & (array != 0)
    else:
        supplied_mask = np.asarray(mask)

        if supplied_mask.shape != array.shape:
            raise ValueError(
                "The foreground mask and MRI volume must have identical "
                f"shapes; got {supplied_mask.shape} and {array.shape}."
            )

        foreground = (supplied_mask != 0) & finite_mask

    if not np.any(foreground):
        if allow_constant:
            return np.zeros(array.shape, dtype=np.float32)
        raise ValueError("Cannot normalize an MRI volume with no foreground voxels.")

    foreground_values = output[foreground]

    if not np.all(np.isfinite(foreground_values)):
        if allow_constant:
            foreground_values = np.nan_to_num(foreground_values, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raise ValueError("Foreground contains non-finite values.")

    # Clip extreme foreground intensities if requested.
    if clip_percentiles is not None:
        if len(clip_percentiles) != 2:
            raise ValueError(
                "clip_percentiles must contain exactly two values."
            )

        lower, upper = map(float, clip_percentiles)

        if not (
            np.isfinite(lower)
            and np.isfinite(upper)
            and 0.0 <= lower < upper <= 100.0
        ):
            raise ValueError(
                "Percentiles must satisfy 0 <= lower < upper <= 100."
            )

        lower_bound, upper_bound = np.percentile(
            foreground_values,
            [lower, upper],
        )

        foreground_values = np.clip(
            foreground_values,
            lower_bound,
            upper_bound,
        )

    # Calculate statistics from the clipped foreground values.
    mean = float(np.mean(foreground_values, dtype=np.float64))
    std = float(np.std(foreground_values, dtype=np.float64))

    if not np.isfinite(mean) or not np.isfinite(std):
        if allow_constant:
            return np.zeros(array.shape, dtype=np.float32)
        raise ValueError("Foreground statistics are not finite.")

    if std < eps:
        if allow_constant:
            return np.zeros(array.shape, dtype=np.float32)
        raise ValueError(
            "Foreground intensity standard deviation is too small "
            "for reliable z-score normalization."
        )

    # Normalize the clipped foreground values.
    normalized = (foreground_values - mean) / std

    # Use a zero-filled output to keep background voxels at zero.
    result = np.zeros(array.shape, dtype=np.float32)
    result[foreground] = normalized.astype(np.float32)

    return result


def normalize_image(
    image: nib.spatialimages.SpatialImage,
    mask: ArrayLike | None = None,
    clip_percentiles: Sequence[float] | None = (0.5, 99.5),
    eps: float = 1e-8,
) -> nib.Nifti1Image:
    """
    Normalize a NIfTI MRI image while preserving its spatial geometry.

    Parameters
    ----------
    image:
        Loaded NIfTI image.
    mask:
        Optional foreground mask array in the same voxel grid as the image.
    clip_percentiles:
        Percentile limits passed to normalize_intensity.
    eps:
        Minimum accepted standard deviation.

    Returns
    -------
    nibabel.Nifti1Image
        Normalized image with the original affine and copied header.
    """
    if len(image.shape) != 3:
        raise ValueError(
            f"Expected a 3D NIfTI image, received shape {image.shape}."
        )

    data = np.asarray(image.dataobj, dtype=np.float32)

    normalized = normalize_intensity(
        data=data,
        mask=mask,
        clip_percentiles=clip_percentiles,
        eps=eps,
    )

    header = image.header.copy()
    header.set_data_dtype(np.float32)

    return nib.Nifti1Image(
        normalized,
        image.affine.copy(),
        header=header,
    )


def normalize_nifti(
    input_path: str | Path,
    output_path: str | Path,
    mask_path: str | Path | None = None,
    clip_percentiles: Sequence[float] | None = (0.5, 99.5),
    eps: float = 1e-8,
) -> Path:
    """
    Normalize a NIfTI image and save it to a separate output path.

    The input image is read-only. The output is saved to output_path.

    Parameters
    ----------
    input_path:
        Path to the source MRI NIfTI file.
    output_path:
        Destination path for the normalized NIfTI image.
    mask_path:
        Optional path to a foreground mask in the same voxel grid.
        Its shape and affine must match the MRI image.
    clip_percentiles:
        Percentile limits passed to normalize_intensity.
    eps:
        Minimum accepted standard deviation.

    Returns
    -------
    pathlib.Path
        Path to the saved normalized image.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input MRI file does not exist: {input_path}"
        )

    image = nib.load(str(input_path))

    mask_array = None

    if mask_path is not None:
        mask_path = Path(mask_path)

        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Foreground mask file does not exist: {mask_path}"
            )

        mask_image = nib.load(str(mask_path))

        if len(mask_image.shape) != 3:
            raise ValueError(
                f"Expected a 3D mask, received shape {mask_image.shape}."
            )

        if mask_image.shape != image.shape:
            raise ValueError(
                "The mask and MRI image must have identical shapes; "
                f"got {mask_image.shape} and {image.shape}."
            )

        if not np.allclose(
            mask_image.affine,
            image.affine,
            atol=1e-3,
            rtol=0.0,
        ):
            raise ValueError(
                "The mask and MRI image affines differ. Resample the mask "
                "to the MRI grid before using it for normalization."
            )

        mask_array = np.asarray(mask_image.dataobj)

    normalized_image = normalize_image(
        image=image,
        mask=mask_array,
        clip_percentiles=clip_percentiles,
        eps=eps,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(normalized_image, str(output_path))

    return output_path