"""
MRI registration utilities for the FCD preprocessing pipeline.

Aligns moving T1-weighted MRI to fixed FLAIR reference space using SimpleITK.
Fixed image: FLAIR MRI
Moving image: T1-weighted MRI

Registration estimates the anatomical coordinate transformation from FLAIR physical
space to T1 space. Resampling then evaluates T1 intensities on the exact FLAIR
reference grid.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk


def register_to_reference(
    moving_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path | None = None,
    *,
    method: str = "rigid",
    transform_output_path: str | Path | None = None,
    number_of_iterations: int = 200,
    number_of_bins: int = 50,
    sampling_percentage: float = 0.20,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Register a moving MRI image (e.g. T1) to a fixed anatomical reference (e.g. FLAIR).

    In SimpleITK, the registration method maps coordinates from the fixed space
    to the moving space: x_moving = Transform(x_fixed).
    Resampling evaluates the moving image at transformed fixed-grid coordinates.

    Parameters
    ----------
    moving_path:
        Path to moving image (T1).
    reference_path:
        Path to fixed reference image (FLAIR).
    output_path:
        Destination for registered/resampled moving image. If None, array is kept in memory.
    method:
        "rigid" (Euler3DTransform) or "affine" (AffineTransform). Default: "rigid".
    transform_output_path:
        Optional path to save SimpleITK transform (.tfm).
    number_of_iterations:
        Optimizer iterations.
    number_of_bins:
        Mattes mutual information histogram bins.
    sampling_percentage:
        Spatial sampling percentage.

    Returns
    -------
    dict containing:
        - "resampled_image": SimpleITK image on reference grid
        - "transform": SimpleITK Transform
        - "output_path": Path or None
        - "transform_path": Path or None
        - "final_metric": float
        - "stop_condition": str
        - "method": str
    """
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected registration arguments: {unexpected}")

    moving_path = Path(moving_path)
    reference_path = Path(reference_path)

    if not moving_path.is_file():
        raise FileNotFoundError(f"Moving image not found: {moving_path}")
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference image not found: {reference_path}")

    if number_of_iterations <= 0:
        raise ValueError("number_of_iterations must be positive.")
    if number_of_bins < 5:
        raise ValueError("number_of_bins must be at least 5.")
    if not 0 < sampling_percentage <= 1:
        raise ValueError("sampling_percentage must be in (0, 1].")

    method = method.lower()
    if method not in {"rigid", "affine"}:
        raise NotImplementedError(
            f"Unsupported registration method: {method!r}. Supported are 'rigid' and 'affine'."
        )

    fixed = sitk.ReadImage(str(reference_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)

    if fixed.GetDimension() != 3 or moving.GetDimension() != 3:
        raise ValueError("Registration requires 3D MRI images.")

    # Initialize transform using geometry centers
    if method == "rigid":
        initial_transform = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
    else:
        initial_rigid = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        initial_transform = sitk.AffineTransform(3)
        initial_transform.SetCenter(initial_rigid.GetCenter())
        initial_transform.SetMatrix(initial_rigid.GetMatrix())
        initial_transform.SetTranslation(initial_rigid.GetTranslation())

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=number_of_bins)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling_percentage, seed=42)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=number_of_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    # Use a multiresolution pyramid adapted to image dimensions
    min_dim = min(list(fixed.GetSize()) + list(moving.GetSize()))
    if min_dim >= 64:
        shrink_factors = [4, 2, 1]
        smoothing_sigmas = [2, 1, 0]
    elif min_dim >= 32:
        shrink_factors = [2, 1]
        smoothing_sigmas = [1, 0]
    else:
        shrink_factors = [1]
        smoothing_sigmas = [0]

    reg.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = reg.Execute(fixed, moving)
    final_metric = float(reg.GetMetricValue())
    stop_condition = str(reg.GetOptimizerStopConditionDescription())

    # Resample moving image onto fixed reference grid
    resampled = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )

    out_p = None
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(resampled, str(out_p), True)

    tfm_p = None
    if transform_output_path is not None:
        tfm_p = Path(transform_output_path)
        tfm_p.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(final_transform, str(tfm_p))

    return {
        "resampled_image": resampled,
        "transform": final_transform,
        "output_path": str(out_p) if out_p else None,
        "transform_path": str(tfm_p) if tfm_p else None,
        "final_metric": final_metric,
        "stop_condition": stop_condition,
        "method": method,
        "fixed_size": list(fixed.GetSize()),
        "fixed_spacing": list(fixed.GetSpacing()),
        "moving_size": list(moving.GetSize()),
        "moving_spacing": list(moving.GetSpacing()),
        "resampled_size": list(resampled.GetSize()),
        "resampled_spacing": list(resampled.GetSpacing()),
    }


def apply_transform(
    moving_image_path: str | Path,
    reference_image_path: str | Path,
    transform_path_or_obj: str | Path | sitk.Transform,
    output_path: str | Path | None = None,
    is_mask: bool = False,
) -> sitk.Image:
    """
    Apply a pre-estimated spatial transformation to map a moving image into reference space.

    Uses linear interpolation for images, nearest-neighbor for masks.
    """
    reference = sitk.ReadImage(str(reference_image_path), sitk.sitkFloat32)
    moving = sitk.ReadImage(
        str(moving_image_path),
        sitk.sitkUInt8 if is_mask else sitk.sitkFloat32,
    )

    if isinstance(transform_path_or_obj, (str, Path)):
        transform = sitk.ReadTransform(str(transform_path_or_obj))
    else:
        transform = transform_path_or_obj

    interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    default_pixel = 0.0

    resampled = sitk.Resample(
        moving,
        reference,
        transform,
        interpolator,
        default_pixel,
        moving.GetPixelID(),
    )

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(resampled, str(out_p), True)

    return resampled


def create_registration_qc_visuals(
    fixed_path: str | Path,
    resampled_moving_path: str | Path,
    subject_id: str | None = None,
    output_png_path: str | Path | None = None,
    return_base64: bool = True,
    roi_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Generate side-by-side, overlay, and checkerboard QC views for registration.

    When ``roi_path`` is supplied the lesion ROI contour (drawn in red) is
    overlaid on both the Fixed-FLAIR panel and the Color-Overlay panel.  This
    is meaningful *after* registration because the ROI lives on the FLAIR grid
    and the resampled T1 has been mapped to the same grid, so the contour
    should align with anatomy in both images.

    Assesses geometry alignment and quantitative intensity correlation.
    Returns metadata and optional base64 data URI for embedded HTML reporting.
    """
    fixed_sitk = sitk.ReadImage(str(fixed_path), sitk.sitkFloat32)
    resampled_sitk = sitk.ReadImage(str(resampled_moving_path), sitk.sitkFloat32)

    fixed_arr = sitk.GetArrayFromImage(fixed_sitk)  # shape: (z, y, x)
    resampled_arr = sitk.GetArrayFromImage(resampled_sitk)

    # Load ROI mask if provided (must already be on the FLAIR grid)
    roi_arr: np.ndarray | None = None
    roi_lesion_voxels: int = 0
    if roi_path is not None and Path(roi_path).is_file():
        try:
            roi_nib = nib.load(str(roi_path))
            roi_arr = (np.asarray(roi_nib.dataobj) > 0).astype(np.uint8)
            roi_lesion_voxels = int(np.count_nonzero(roi_arr))
        except Exception:
            roi_arr = None

    # Validate geometry
    size_match = fixed_sitk.GetSize() == resampled_sitk.GetSize()
    spacing_match = np.allclose(fixed_sitk.GetSpacing(), resampled_sitk.GetSpacing(), atol=1e-5)
    origin_match = np.allclose(fixed_sitk.GetOrigin(), resampled_sitk.GetOrigin(), atol=1e-4)
    direction_match = np.allclose(fixed_sitk.GetDirection(), resampled_sitk.GetDirection(), atol=1e-4)

    # Compute CheckerBoard
    checker_sitk = sitk.CheckerBoard(fixed_sitk, resampled_sitk, checkerPattern=[4, 4, 4])
    checker_arr = sitk.GetArrayFromImage(checker_sitk)

    # Select central slices where brain signal is present
    fg = fixed_arr > 0
    if np.any(fg):
        coords = np.argwhere(fg)
        center = np.round(coords.mean(axis=0)).astype(int)
        cz, cy, cx = int(center[0]), int(center[1]), int(center[2])
    else:
        cz = fixed_arr.shape[0] // 2
        cy = fixed_arr.shape[1] // 2
        cx = fixed_arr.shape[2] // 2

    # Normalize slices for display
    def _norm(a: np.ndarray) -> np.ndarray:
        p1, p99 = np.percentile(a, [1, 99])
        if p99 > p1:
            return np.clip((a - p1) / (p99 - p1), 0, 1)
        return np.zeros_like(a)

    fig, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)

    views = [
        ("Axial", cz, 0),
        ("Coronal", cy, 1),
        ("Sagittal", cx, 2),
    ]

    def _get_roi_slice(axis_idx: int, sidx: int) -> np.ndarray | None:
        """Return the 2-D ROI binary slice for the given view axis."""
        if roi_arr is None:
            return None
        if axis_idx == 0:
            return roi_arr[sidx, :, :]
        if axis_idx == 1:
            return roi_arr[:, sidx, :]
        return roi_arr[:, :, sidx]

    for row, (vname, sidx, axis_idx) in enumerate(views):
        if axis_idx == 0:
            f_slice = _norm(fixed_arr[sidx, :, :])
            r_slice = _norm(resampled_arr[sidx, :, :])
            c_slice = _norm(checker_arr[sidx, :, :])
        elif axis_idx == 1:
            f_slice = _norm(fixed_arr[:, sidx, :])
            r_slice = _norm(resampled_arr[:, sidx, :])
            c_slice = _norm(checker_arr[:, sidx, :])
        else:
            f_slice = _norm(fixed_arr[:, :, sidx])
            r_slice = _norm(resampled_arr[:, :, sidx])
            c_slice = _norm(checker_arr[:, :, sidx])

        roi_slice = _get_roi_slice(axis_idx, sidx)

        # Col 0: Fixed FLAIR  (+ROI contour if available)
        axes[row, 0].imshow(f_slice, cmap="gray", origin="lower")
        if roi_slice is not None and roi_slice.any():
            axes[row, 0].contour(roi_slice, levels=[0.5], colors=["red"], linewidths=[0.8])
        axes[row, 0].set_title(f"Fixed FLAIR ({vname})", fontsize=9)
        axes[row, 0].axis("off")

        # Col 1: Resampled T1 (no ROI — shows T1 anatomy in FLAIR space)
        axes[row, 1].imshow(r_slice, cmap="gray", origin="lower")
        axes[row, 1].set_title(f"Resampled T1 ({vname})", fontsize=9)
        axes[row, 1].axis("off")

        # Col 2: Color Overlay (FLAIR=Green, T1=Magenta, ROI contour=Red)
        overlay = np.zeros((*f_slice.shape, 3), dtype=np.float32)
        overlay[..., 0] = r_slice  # Red channel (T1)
        overlay[..., 1] = f_slice  # Green channel (FLAIR)
        overlay[..., 2] = r_slice  # Blue channel (T1 → Magenta)
        axes[row, 2].imshow(overlay, origin="lower")
        if roi_slice is not None and roi_slice.any():
            axes[row, 2].contour(roi_slice, levels=[0.5], colors=["red"], linewidths=[1.0])
        axes[row, 2].set_title("Overlay (FLAIR=G, T1=M)", fontsize=9)
        axes[row, 2].axis("off")

        # Col 3: Checkerboard
        axes[row, 3].imshow(c_slice, cmap="gray", origin="lower")
        axes[row, 3].set_title(f"Checkerboard ({vname})", fontsize=9)
        axes[row, 3].axis("off")

    sid_str = subject_id or Path(fixed_path).parent.parent.name
    fig.suptitle(f"T1-to-FLAIR Registration QC — {sid_str}", fontsize=13)

    # Compute correlation within foreground
    mask = (fixed_arr > 0) & (resampled_arr > 0)
    correlation = 0.0
    if np.count_nonzero(mask) > 100:
        f_vals = fixed_arr[mask]
        r_vals = resampled_arr[mask]
        std_f, std_r = np.std(f_vals), np.std(r_vals)
        if std_f > 1e-5 and std_r > 1e-5:
            correlation = float(np.corrcoef(f_vals, r_vals)[0, 1])

    base64_str = ""
    if return_base64:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        base64_str = "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")
        buf.close()

    if output_png_path is not None:
        p = Path(output_png_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=120, bbox_inches="tight")

    plt.close(fig)

    # Assess review status
    flags: list[str] = []
    if not (size_match and spacing_match and origin_match and direction_match):
        flags.append("GEOMETRY_MISMATCH")
    if correlation < 0.2:
        flags.append("LOW_INTENSITY_CORRELATION")

    status = "REVIEW" if flags else "PASS"

    return {
        "subject_id": sid_str,
        "status": status,
        "flags": flags,
        "size_match": size_match,
        "spacing_match": spacing_match,
        "origin_match": origin_match,
        "direction_match": direction_match,
        "foreground_correlation": correlation,
        "roi_lesion_voxels": roi_lesion_voxels,
        "roi_overlay": roi_arr is not None,
        "base64_image": base64_str,
        "png_path": str(output_png_path) if output_png_path else None,
    }