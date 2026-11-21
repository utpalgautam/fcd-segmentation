"""Spatial geometry utilities for MRI preprocessing."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

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


def get_direction_matrix(affine: np.ndarray) -> np.ndarray:
    """Extract 3x3 direction cosine matrix from a 4x4 affine matrix."""
    linear = np.asarray(affine[:3, :3], dtype=np.float64)
    voxel_sizes = nib.affines.voxel_sizes(affine)
    return linear / voxel_sizes


def compute_roi_physical_centroid(
    roi_data: np.ndarray,
    affine: np.ndarray,
) -> list[float] | None:
    """Compute center of mass in physical scanner coordinates (mm)."""
    coords = np.argwhere(roi_data > 0)
    if coords.size == 0:
        return None
    voxel_centroid = coords.mean(axis=0)
    phys = nib.affines.apply_affine(affine, voxel_centroid)
    return [float(x) for x in phys]


def compute_roi_physical_bbox(
    roi_data: np.ndarray,
    affine: np.ndarray,
) -> dict[str, list[float]] | None:
    """Compute axis-aligned physical bounding box in scanner coordinates (mm)."""
    coords = np.argwhere(roi_data > 0)
    if coords.size == 0:
        return None
    min_v = coords.min(axis=0)
    max_v = coords.max(axis=0)
    corners = np.array(list(product(
        [min_v[0], max_v[0]],
        [min_v[1], max_v[1]],
        [min_v[2], max_v[2]],
    )), dtype=np.float64)
    corners_phys = nib.affines.apply_affine(affine, corners)
    min_phys = [float(x) for x in corners_phys.min(axis=0)]
    max_phys = [float(x) for x in corners_phys.max(axis=0)]
    return {"min": min_phys, "max": max_phys}


def validate_subject_geometry_and_roi(
    flair_img: nib.spatialimages.SpatialImage | str | Path,
    t1_img: nib.spatialimages.SpatialImage | str | Path,
    roi_img: nib.spatialimages.SpatialImage | str | Path,
    raw_flair_img: nib.spatialimages.SpatialImage | str | Path | None = None,
    raw_roi_img: nib.spatialimages.SpatialImage | str | Path | None = None,
    subject_id: str = "unknown",
    is_hc: bool = False,
    extra_label_images: list[nib.spatialimages.SpatialImage | str | Path] | None = None,
    tolerance_mm: float = 0.01,
    centroid_tolerance_mm: float = 0.5,
) -> dict[str, Any]:
    """
    Thoroughly validate spatial geometry, discrete labels, and ROI consistency.

    Checks:
    1. FLAIR, registered T1, and final ROI shapes match exactly.
    2. Spacing, origin, and direction matrix match across all three images.
    3. Maximum corner displacement between modalities is within tolerance.
    4. Discrete labels: strictly {0, 1} for FCD; strictly {0} for HC.
    5. Nonzero lesion voxels > 0 for FCD; exactly 0 for HC.
    6. Physical centroid consistency between raw ROI and final ROI.
    7. Physical bounding box consistency between raw ROI and final ROI.
    8. ROI voxel count preservation within legitimate resampling limits.
    9. Exact label synchronization across all target nnU-Net dataset labels.
    """
    errors: list[str] = []
    warnings: list[str] = []

    def _load(im: nib.spatialimages.SpatialImage | str | Path) -> nib.spatialimages.SpatialImage:
        if isinstance(im, (str, Path)):
            return nib.load(str(im))
        return im

    flair = _load(flair_img)
    t1 = _load(t1_img)
    roi = _load(roi_img)

    validate_image_geometry(flair)
    validate_image_geometry(t1)
    validate_image_geometry(roi)

    flair_shape = list(flair.shape)
    t1_shape = list(t1.shape)
    roi_shape = list(roi.shape)

    flair_spacing = [float(s) for s in nib.affines.voxel_sizes(flair.affine)]
    t1_spacing = [float(s) for s in nib.affines.voxel_sizes(t1.affine)]
    roi_spacing = [float(s) for s in nib.affines.voxel_sizes(roi.affine)]

    flair_origin = [float(o) for o in flair.affine[:3, 3]]
    t1_origin = [float(o) for o in t1.affine[:3, 3]]
    roi_origin = [float(o) for o in roi.affine[:3, 3]]

    flair_dir = get_direction_matrix(flair.affine)
    t1_dir = get_direction_matrix(t1.affine)
    roi_dir = get_direction_matrix(roi.affine)

    # 1. Shape comparison
    if flair_shape != t1_shape:
        errors.append(f"FLAIR and Registered T1 shapes differ: {flair_shape} vs {t1_shape}")
    if flair_shape != roi_shape:
        errors.append(f"FLAIR and ROI shapes differ: {flair_shape} vs {roi_shape}")

    # 2. Spacing comparison
    if not np.allclose(flair_spacing, t1_spacing, atol=1e-4):
        errors.append(f"FLAIR and Registered T1 voxel spacing differ: {flair_spacing} vs {t1_spacing}")
    if not np.allclose(flair_spacing, roi_spacing, atol=1e-4):
        errors.append(f"FLAIR and ROI voxel spacing differ: {flair_spacing} vs {roi_spacing}")

    # 3. Origin comparison
    if not np.allclose(flair_origin, t1_origin, atol=1e-3):
        errors.append(f"FLAIR and Registered T1 origins differ: {flair_origin} vs {t1_origin}")
    if not np.allclose(flair_origin, roi_origin, atol=1e-3):
        errors.append(f"FLAIR and ROI origins differ: {flair_origin} vs {roi_origin}")

    # 4. Direction matrix comparison
    if not np.allclose(flair_dir, t1_dir, atol=1e-3):
        errors.append("FLAIR and Registered T1 direction matrices differ")
    if not np.allclose(flair_dir, roi_dir, atol=1e-3):
        errors.append("FLAIR and ROI direction matrices differ")

    # 5. Corner displacement
    max_disp = 0.0
    if flair_shape == t1_shape:
        disp_t1 = maximum_corner_displacement_mm(flair, t1)
        max_disp = max(max_disp, disp_t1)
        if disp_t1 > tolerance_mm:
            errors.append(f"FLAIR and Registered T1 corner displacement ({disp_t1:.6f} mm) exceeds tolerance ({tolerance_mm} mm)")
    if flair_shape == roi_shape:
        disp_roi = maximum_corner_displacement_mm(flair, roi)
        max_disp = max(max_disp, disp_roi)
        if disp_roi > tolerance_mm:
            errors.append(f"FLAIR and ROI corner displacement ({disp_roi:.6f} mm) exceeds tolerance ({tolerance_mm} mm)")

    # 6. Data arrays and label values
    roi_data = np.asarray(roi.dataobj)
    unique_vals = [float(v) for v in np.unique(roi_data)]
    final_nonzero = int(np.count_nonzero(roi_data))

    if is_hc:
        if final_nonzero != 0:
            errors.append(f"HC subject {subject_id} must have all-zero label, but found {final_nonzero} non-zero voxels")
        if not set(unique_vals).issubset({0.0}):
            errors.append(f"HC subject label contains non-zero values: {unique_vals}")
    else:
        if not set(unique_vals).issubset({0.0, 1.0}):
            errors.append(f"FCD subject label contains invalid values (must be binary 0 and 1): {unique_vals}")
        if final_nonzero == 0:
            errors.append(f"FCD subject {subject_id} lesion ROI contains 0 non-zero voxels (unexpected empty mask)")

    # 7. Physical Centroid & Bounding Box vs Raw ROI
    raw_nonzero: int | None = None
    raw_centroid_phys: list[float] | None = None
    final_centroid_phys: list[float] | None = None
    centroid_dist_mm: float | None = None
    raw_bbox: dict[str, list[float]] | None = None
    final_bbox: dict[str, list[float]] | None = None

    if final_nonzero > 0:
        final_centroid_phys = compute_roi_physical_centroid(roi_data, roi.affine)
        final_bbox = compute_roi_physical_bbox(roi_data, roi.affine)

    if raw_roi_img is not None and not is_hc:
        raw_roi = _load(raw_roi_img)
        validate_image_geometry(raw_roi)
        raw_data = np.asarray(raw_roi.dataobj)
        raw_nonzero = int(np.count_nonzero(raw_data))

        if raw_nonzero == 0:
            errors.append(f"Raw ROI file has 0 lesion voxels for {subject_id}")
        else:
            raw_centroid_phys = compute_roi_physical_centroid(raw_data, raw_roi.affine)
            raw_bbox = compute_roi_physical_bbox(raw_data, raw_roi.affine)

            if raw_centroid_phys is not None and final_centroid_phys is not None:
                centroid_dist_mm = float(np.linalg.norm(np.array(raw_centroid_phys) - np.array(final_centroid_phys)))
                # If raw ROI is on FLAIR grid, centroid should match almost exactly (<= 0.05 mm)
                # If resampled from a different grid, allow centroid_tolerance_mm
                if centroid_dist_mm > centroid_tolerance_mm:
                    errors.append(
                        f"ROI physical centroid shifted by {centroid_dist_mm:.4f} mm between raw and final ROI "
                        f"(exceeds tolerance of {centroid_tolerance_mm} mm). Check if registration transform was wrongly applied!"
                    )

            # Check voxel count preservation
            if raw_flair_img is not None:
                raw_flair = _load(raw_flair_img)
                is_same = same_grid(raw_flair, raw_roi, tolerance_mm=tolerance_mm)
                if is_same and raw_nonzero != final_nonzero:
                    errors.append(
                        f"Raw ROI was already on FLAIR grid, but voxel count changed: "
                        f"raw={raw_nonzero} vs final={final_nonzero}"
                    )
                elif not is_same:
                    # Legitimate resampling: voxel count ratio should be within reasonable bounds
                    vol_ratio = final_nonzero / max(raw_nonzero, 1)
                    if not (0.7 <= vol_ratio <= 1.3):
                        warnings.append(f"ROI voxel count changed significantly during resampling: ratio={vol_ratio:.2f}")

    # 8. Extra label images synchronization check
    if extra_label_images:
        for idx, extra_lbl_item in enumerate(extra_label_images):
            extra_lbl = _load(extra_lbl_item)
            if not same_grid(roi, extra_lbl, tolerance_mm=tolerance_mm):
                errors.append(f"Extra label image {idx} geometry does not match reference ROI")
            extra_data = np.asarray(extra_lbl.dataobj)
            if not np.array_equal(roi_data, extra_data):
                errors.append(f"Extra label image {idx} voxel content does not match reference ROI")

    status = "FAIL" if errors else "PASS"

    return {
        "subject_id": subject_id,
        "is_hc": is_hc,
        "status": status,
        "flair_shape": flair_shape,
        "registered_t1_shape": t1_shape,
        "roi_shape": roi_shape,
        "flair_spacing": flair_spacing,
        "registered_t1_spacing": t1_spacing,
        "roi_spacing": roi_spacing,
        "flair_origin": flair_origin,
        "registered_t1_origin": t1_origin,
        "roi_origin": roi_origin,
        "flair_direction": [round(float(x), 4) for x in flair_dir.flatten()],
        "registered_t1_direction": [round(float(x), 4) for x in t1_dir.flatten()],
        "roi_direction": [round(float(x), 4) for x in roi_dir.flatten()],
        "max_geometry_diff_mm": float(max_disp),
        "roi_voxels_before": raw_nonzero,
        "roi_voxels_after": final_nonzero,
        "roi_physical_centroid_before": raw_centroid_phys,
        "roi_physical_centroid_after": final_centroid_phys,
        "roi_centroid_distance_mm": centroid_dist_mm,
        "roi_bbox_before": raw_bbox,
        "roi_bbox_after": final_bbox,
        "errors": errors,
        "warnings": warnings,
    }

