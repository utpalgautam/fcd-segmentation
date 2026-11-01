"""
nnU-Net v2 dataset creation and validation utilities for FCD MRI cohorts.

Creates three synchronized nnU-Net v2 datasets:
1. FLAIR-only
2. T1-only
3. Fusion (FLAIR + T1)

Enforces:
- Consistent channel ordering: _0000 = FLAIR, _0001 = T1 (registered to FLAIR)
- Single shared subject split across all three datasets
- Atomic NIfTI and JSON writes
- Valid nnU-Net v2 dataset.json structure
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from fcd_segmentation.preprocessing.geometry import (
    maximum_corner_displacement_mm,
    same_grid,
    validate_image_geometry,
)
from fcd_segmentation.preprocessing.io import save_nifti


def write_json_atomic(data: dict[str, Any] | list[Any], destination: Path | str) -> Path:
    """Write JSON data to file atomically via a temporary sibling file."""
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(f"{destination.suffix}.tmp.{os.getpid()}")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_path, destination)
    return destination


def create_nnunet_dataset_json(
    output_dir: Path | str,
    channel_names: dict[str, str],
    labels: dict[str, int],
    num_training: int,
    file_ending: str = ".nii.gz",
    dataset_name: str | None = None,
    description: str | None = None,
    reference: str = "Joshi et al. 2025 Reproduction / FCD Preprocessing",
    release: str = "1.0",
) -> Path:
    """
    Generate and write a valid nnU-Net v2 dataset.json file.
    """
    output_dir = Path(output_dir).resolve()
    dataset_name = dataset_name or output_dir.name
    dataset_json_path = output_dir / "dataset.json"

    content = {
        "channel_names": {str(k): str(v) for k, v in channel_names.items()},
        "labels": {str(k): int(v) for k, v in labels.items()},
        "numTraining": int(num_training),
        "file_ending": file_ending,
        "name": dataset_name,
        "description": description or f"FCD MRI Segmentation: {dataset_name}",
        "reference": reference,
        "release": release,
    }

    return write_json_atomic(content, dataset_json_path)


def write_subject_to_nnunet(
    subject_id: str,
    flair_image: nib.spatialimages.SpatialImage,
    t1_image: nib.spatialimages.SpatialImage,
    roi_image: nib.spatialimages.SpatialImage,
    split: str,
    flair_dataset_dir: Path | str,
    t1_dataset_dir: Path | str,
    fusion_dataset_dir: Path | str,
    overwrite: bool = False,
    held_out_target: str = "imagesTr",
) -> dict[str, Any]:
    """
    Write a single validated subject to all three nnU-Net dataset folders.

    Parameters
    ----------
    subject_id:
        Subject identifier (e.g. 'sub-00001').
    flair_image:
        Preprocessed FLAIR image on reference grid.
    t1_image:
        Preprocessed T1 image registered and resampled onto FLAIR reference grid.
    roi_image:
        Lesion ROI mask on the reference grid (discrete integer labels).
    split:
        'train' or 'test'.
    flair_dataset_dir, t1_dataset_dir, fusion_dataset_dir:
        Output dataset roots.
    overwrite:
        Allow replacing existing files.
    held_out_target:
        'imagesTr' (all cases to imagesTr) or 'imagesTs' (test cases to imagesTs).

    Returns
    -------
    dict summarizing written files and validation status.
    """
    # Verify spatial alignment between FLAIR and T1 (Fusion channels)
    validate_image_geometry(flair_image)
    validate_image_geometry(t1_image)
    validate_image_geometry(roi_image)

    if flair_image.shape != t1_image.shape:
        raise ValueError(
            f"Fusion channels FLAIR and T1 do not share the exact same grid for {subject_id}: "
            f"shapes differ ({flair_image.shape} vs {t1_image.shape})."
        )
    fusion_match = same_grid(flair_image, t1_image, tolerance_mm=0.01)
    if not fusion_match:
        disp = maximum_corner_displacement_mm(flair_image, t1_image)
        raise ValueError(
            f"Fusion channels FLAIR and T1 do not share the exact same grid for {subject_id}. "
            f"Corner displacement: {disp:.6f} mm"
        )

    if flair_image.shape != roi_image.shape:
        raise ValueError(
            f"ROI does not share the reference FLAIR grid for {subject_id}: "
            f"shapes differ ({flair_image.shape} vs {roi_image.shape})."
        )
    roi_match = same_grid(flair_image, roi_image, tolerance_mm=0.01)
    if not roi_match:
        disp = maximum_corner_displacement_mm(flair_image, roi_image)
        raise ValueError(
            f"ROI does not share the reference FLAIR grid for {subject_id}. "
            f"Corner displacement: {disp:.6f} mm"
        )

    # Determine subfolder: train -> imagesTr / labelsTr
    # If split == 'test' and held_out_target == 'imagesTs':
    #   imagesTs / labelsTs (labelsTs preserved for evaluation)
    is_test = (split.lower() == "test")
    use_ts = is_test and (held_out_target == "imagesTs")

    img_folder_name = "imagesTs" if use_ts else "imagesTr"
    lbl_folder_name = "labelsTs" if use_ts else "labelsTr"

    flair_dir = Path(flair_dataset_dir).resolve()
    t1_dir = Path(t1_dataset_dir).resolve()
    fusion_dir = Path(fusion_dataset_dir).resolve()

    flair_img_path = flair_dir / img_folder_name / f"{subject_id}_0000.nii.gz"
    flair_lbl_path = flair_dir / lbl_folder_name / f"{subject_id}.nii.gz"

    t1_img_path = t1_dir / img_folder_name / f"{subject_id}_0000.nii.gz"
    t1_lbl_path = t1_dir / lbl_folder_name / f"{subject_id}.nii.gz"

    fusion_flair_path = fusion_dir / img_folder_name / f"{subject_id}_0000.nii.gz"
    fusion_t1_path = fusion_dir / img_folder_name / f"{subject_id}_0001.nii.gz"
    fusion_lbl_path = fusion_dir / lbl_folder_name / f"{subject_id}.nii.gz"

    # Write files atomically (using save_nifti)
    # Ensure float32 for images and uint8 for labels
    save_nifti(flair_image, flair_img_path, overwrite=overwrite, dtype=np.float32)
    save_nifti(roi_image, flair_lbl_path, overwrite=overwrite, dtype=np.uint8)

    save_nifti(t1_image, t1_img_path, overwrite=overwrite, dtype=np.float32)
    save_nifti(roi_image, t1_lbl_path, overwrite=overwrite, dtype=np.uint8)

    save_nifti(flair_image, fusion_flair_path, overwrite=overwrite, dtype=np.float32)
    save_nifti(t1_image, fusion_t1_path, overwrite=overwrite, dtype=np.float32)
    save_nifti(roi_image, fusion_lbl_path, overwrite=overwrite, dtype=np.uint8)

    return {
        "subject_id": subject_id,
        "split": split,
        "img_folder": img_folder_name,
        "lbl_folder": lbl_folder_name,
        "flair_image_path": str(flair_img_path),
        "flair_label_path": str(flair_lbl_path),
        "t1_image_path": str(t1_img_path),
        "t1_label_path": str(t1_lbl_path),
        "fusion_flair_path": str(fusion_flair_path),
        "fusion_t1_path": str(fusion_t1_path),
        "fusion_label_path": str(fusion_lbl_path),
        "fusion_geometry_match": True,
    }


def validate_nnunet_subject_outputs(
    subject_id: str,
    flair_dataset_dir: Path | str,
    t1_dataset_dir: Path | str,
    fusion_dataset_dir: Path | str,
    split: str = "train",
    held_out_target: str = "imagesTr",
) -> dict[str, Any]:
    """
    Validate that all expected output files exist, load correctly,
    share the reference grid, have finite intensities, and contain valid labels.
    """
    is_test = (split.lower() == "test")
    use_ts = is_test and (held_out_target == "imagesTs")
    img_folder = "imagesTs" if use_ts else "imagesTr"
    lbl_folder = "labelsTs" if use_ts else "labelsTr"

    flair_dir = Path(flair_dataset_dir)
    t1_dir = Path(t1_dataset_dir)
    fusion_dir = Path(fusion_dataset_dir)

    expected_files = {
        "flair_img": flair_dir / img_folder / f"{subject_id}_0000.nii.gz",
        "flair_lbl": flair_dir / lbl_folder / f"{subject_id}.nii.gz",
        "t1_img": t1_dir / img_folder / f"{subject_id}_0000.nii.gz",
        "t1_lbl": t1_dir / lbl_folder / f"{subject_id}.nii.gz",
        "fusion_ch0": fusion_dir / img_folder / f"{subject_id}_0000.nii.gz",
        "fusion_ch1": fusion_dir / img_folder / f"{subject_id}_0001.nii.gz",
        "fusion_lbl": fusion_dir / lbl_folder / f"{subject_id}.nii.gz",
    }

    missing = [k for k, p in expected_files.items() if not p.is_file()]
    if missing:
        return {
            "subject_id": subject_id,
            "status": "FAIL",
            "errors": [f"Missing expected output files: {missing}"],
            "flair_dataset_valid": False,
            "t1_dataset_valid": False,
            "fusion_dataset_valid": False,
            "fusion_geometry_match": False,
        }

    errors: list[str] = []
    warnings: list[str] = []

    try:
        f_img = nib.load(str(expected_files["flair_img"]))
        t1_img = nib.load(str(expected_files["t1_img"]))
        fus0 = nib.load(str(expected_files["fusion_ch0"]))
        fus1 = nib.load(str(expected_files["fusion_ch1"]))
        lbl = nib.load(str(expected_files["fusion_lbl"]))

        # Check fusion channel grid equality
        if not same_grid(fus0, fus1, tolerance_mm=0.01):
            errors.append("Fusion channels _0000 and _0001 do not share identical voxel grid.")

        if not same_grid(fus0, lbl, tolerance_mm=0.01):
            errors.append("Fusion channels and label do not share identical voxel grid.")

        # Check finite and statistics
        f_data = np.asarray(f_img.dataobj)
        t1_data = np.asarray(t1_img.dataobj)
        lbl_data = np.asarray(lbl.dataobj)

        if not np.all(np.isfinite(f_data)):
            errors.append("FLAIR dataset image contains non-finite values.")
        if not np.all(np.isfinite(t1_data)):
            errors.append("T1 dataset image contains non-finite values.")

        # Check label discrete values
        u_lbl = np.unique(lbl_data)
        if not np.all(np.isin(u_lbl, [0, 1])):
            errors.append(f"Label mask contains unexpected values: {u_lbl}")

        lesion_voxels = int(np.count_nonzero(lbl_data))
        if lesion_voxels == 0:
            warnings.append("Lesion mask has 0 non-zero voxels.")

        target_shape = list(f_img.shape)
        target_spacing = [round(float(s), 3) for s in nib.affines.voxel_sizes(f_img.affine)]
        flair_mean = float(np.mean(f_data[f_data != 0])) if np.any(f_data != 0) else 0.0
        t1_mean = float(np.mean(t1_data[t1_data != 0])) if np.any(t1_data != 0) else 0.0

        status = "FAIL" if errors else ("WARN" if warnings else "PASS")

        return {
            "subject_id": subject_id,
            "status": status,
            "split": split,
            "errors": errors,
            "warnings": warnings,
            "flair_dataset_valid": "flair_img" not in errors and "flair_lbl" not in errors,
            "t1_dataset_valid": "t1_img" not in errors and "t1_lbl" not in errors,
            "fusion_dataset_valid": len(errors) == 0,
            "fusion_geometry_match": "Fusion channels _0000 and _0001 do not share identical voxel grid." not in errors,
            "target_grid_shape": target_shape,
            "target_grid_spacing": target_spacing,
            "lesion_voxels": lesion_voxels,
            "flair_mean": flair_mean,
            "t1_mean": t1_mean,
            "unique_labels": [int(v) for v in u_lbl],
        }

    except Exception as exc:
        return {
            "subject_id": subject_id,
            "status": "FAIL",
            "errors": [f"Validation exception: {type(exc).__name__}: {exc}"],
            "flair_dataset_valid": False,
            "t1_dataset_valid": False,
            "fusion_dataset_valid": False,
            "fusion_geometry_match": False,
        }
