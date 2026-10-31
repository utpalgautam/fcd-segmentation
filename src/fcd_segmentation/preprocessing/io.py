"""
Input/output utilities for NIfTI medical images.

Provides functions for loading, validating, and saving NIfTI images
without modifying source files.
"""

from pathlib import Path

import nibabel as nib
import numpy as np

from fcd_segmentation.preprocessing.geometry import validate_image_geometry

PathLike = str | Path


def load_nifti(
    path: PathLike,
    dtype: np.dtype | None = np.float32,
    validate: bool = True,
) -> nib.Nifti1Image:
    """
    Load a NIfTI image and optionally validate its geometry.

    Parameters
    ----------
    path:
        Path to a .nii or .nii.gz file.
    dtype:
        Optional dtype for checking/loading voxel data. The returned
        NIfTI object retains its original stored data representation.
        Use None to avoid the additional data conversion check.
    validate:
        If True, validate that the image has usable 3D geometry.

    Returns
    -------
    nibabel.Nifti1Image
        Loaded image.

    Raises
    ------
    FileNotFoundError
        If the input path does not exist.
    ValueError
        If the file is not a supported 3D NIfTI image or contains
        invalid geometry/data.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    try:
        image = nib.load(str(path))
    except Exception as exc:
        raise ValueError(
            f"Could not load NIfTI image '{path}': {exc}"
        ) from exc

    if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
        raise ValueError(
            f"Unsupported image type: {type(image).__name__}"
        )

    if len(image.shape) != 3:
        raise ValueError(
            f"Expected a 3D NIfTI image, got shape {image.shape}: {path}"
        )

    if validate:
        validate_image_geometry(image)

    if dtype is not None:
        try:
            data = np.asarray(image.dataobj, dtype=dtype)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Could not read image data as {dtype}: {path}"
            ) from exc

        if data.size == 0:
            raise ValueError(f"NIfTI image has no voxel data: {path}")

        if not np.all(np.isfinite(data)):
            raise ValueError(
                f"NIfTI image contains NaN or infinite values: {path}"
            )

    return image


def load_nifti_data(
    path: PathLike,
    dtype: np.dtype = np.float32,
    validate: bool = True,
) -> np.ndarray:
    """
    Load NIfTI voxel data as a NumPy array.

    Parameters
    ----------
    path:
        Path to a .nii or .nii.gz file.
    dtype:
        NumPy dtype for the returned array.
    validate:
        Whether to validate image geometry and finite voxel values.

    Returns
    -------
    numpy.ndarray
        Voxel data array.
    """
    image = load_nifti(
        path=path,
        dtype=dtype,
        validate=validate,
    )

    data = np.asarray(image.dataobj, dtype=dtype)

    # Return an independent array rather than a proxy into the file.
    return data.copy()


def save_nifti(
    image: nib.spatialimages.SpatialImage,
    output_path: PathLike,
    overwrite: bool = False,
    dtype: np.dtype | None = None,
) -> Path:
    """
    Save a NIfTI image to disk.

    The destination parent directory is created if necessary.
    Existing files are not overwritten unless overwrite=True.

    Parameters
    ----------
    image:
        NIfTI image to save.
    output_path:
        Destination .nii or .nii.gz path.
    overwrite:
        Whether to allow replacing an existing destination file.
    dtype:
        Optional output voxel dtype. If supplied, the voxel data are
        converted to this dtype before saving.

    Returns
    -------
    pathlib.Path
        Destination path.

    Raises
    ------
    FileExistsError
        If the destination exists and overwrite=False.
    ValueError
        If the image or destination is invalid.
    """
    output_path = Path(output_path)

    if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
        raise ValueError(
            "image must be a NIfTI-1 or NIfTI-2 image."
        )

    if len(image.shape) != 3:
        raise ValueError(
            f"Expected a 3D image, got shape {image.shape}."
        )

    if output_path.suffix not in {".nii", ".gz"}:
        raise ValueError(
            "Output path must end in .nii or .nii.gz."
        )

    if output_path.suffix == ".gz" and not output_path.name.endswith(
        ".nii.gz"
    ):
        raise ValueError(
            "Compressed NIfTI output must use the .nii.gz extension."
        )

    if output_path.suffix == ".nii" and not output_path.name.endswith(
        ".nii"
    ):
        raise ValueError("Output path must end in .nii or .nii.gz.")

    validate_image_geometry(image)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {output_path}"
        )

    data = np.asarray(image.dataobj)

    if data.size == 0:
        raise ValueError("Cannot save an image with empty voxel data.")

    if not np.all(np.isfinite(data)):
        raise ValueError(
            "Cannot save an image containing NaN or infinite values."
        )

    if dtype is not None:
        data = data.astype(dtype, copy=False)

    header = image.header.copy()

    if dtype is not None:
        header.set_data_dtype(data.dtype)

    output_image = image.__class__(
        data,
        image.affine.copy(),
        header=header,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        nib.save(output_image, str(output_path))
    except Exception as exc:
        raise OSError(
            f"Failed to save NIfTI image to '{output_path}': {exc}"
        ) from exc

    return output_path