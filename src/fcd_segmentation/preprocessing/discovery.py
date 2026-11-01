"""
Dataset discovery and input validation for FCD MRI cohorts.

Discovers T1, FLAIR, and lesion ROI files per subject without modifying
raw files. Inspects geometry, affine, voxel spacing, orientation,
finite voxel status, and label values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from fcd_segmentation.preprocessing.geometry import (
    maximum_corner_displacement_mm,
    same_grid,
    validate_image_geometry,
)


def is_roi_file(path: Path | str) -> bool:
    """Return True if the file name matches ROI/label patterns."""
    name = Path(path).name.lower()
    return name.endswith(("_roi.nii", "_roi.nii.gz", "_mask.nii", "_mask.nii.gz", "_label.nii", "_label.nii.gz"))


def discover_subject_files(subject_anat_dir: Path | str) -> dict[str, list[Path]]:
    """
    Find candidate T1, FLAIR, and ROI files in a subject's anatomical directory.

    Strictly separates ROI files from FLAIR/T1 intensity images.
    """
    anat_dir = Path(subject_anat_dir)
    if not anat_dir.is_dir():
        return {"t1": [], "flair": [], "roi": []}

    all_files = sorted(
        p for p in anat_dir.iterdir()
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
    )

    t1_candidates: list[Path] = []
    flair_candidates: list[Path] = []
    roi_candidates: list[Path] = []

    for file_path in all_files:
        name = file_path.name
        name_lower = name.lower()

        # Check ROI first so it is never treated as intensity scan
        if is_roi_file(file_path):
            roi_candidates.append(file_path)
        elif "flair" in name_lower:
            flair_candidates.append(file_path)
        elif "t1w" in name_lower or (name_lower.endswith(("_t1.nii", "_t1.nii.gz")) and "flair" not in name_lower):
            t1_candidates.append(file_path)

    return {
        "t1": sorted(set(t1_candidates)),
        "flair": sorted(set(flair_candidates)),
        "roi": sorted(set(roi_candidates)),
    }


def inspect_nifti_metadata(
    path: Path | str,
    is_mask: bool = False,
    read_voxel_data: bool = False,
) -> dict[str, Any]:
    """
    Inspect a NIfTI file without altering it, returning geometric and intensity metadata.
    """
    path = Path(path)
    metadata: dict[str, Any] = {
        "path": str(path.resolve()),
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size if path.is_file() else 0,
        "readable": False,
        "shape": None,
        "spacing": None,
        "affine": None,
        "orientation": None,
        "all_finite": True,
        "min_val": None,
        "max_val": None,
        "unique_labels": None,
        "nonzero_voxels": None,
        "error": None,
    }

    if not path.is_file():
        metadata["error"] = f"File not found: {path}"
        return metadata

    try:
        img = nib.load(str(path))
        validate_image_geometry(img)
        metadata["readable"] = True
        metadata["shape"] = list(img.shape)
        metadata["spacing"] = [float(s) for s in nib.affines.voxel_sizes(img.affine)]
        metadata["affine"] = img.affine.tolist()
        metadata["orientation"] = "".join(nib.aff2axcodes(img.affine))

        if read_voxel_data or is_mask:
            data = np.asarray(img.dataobj)
            finite = bool(np.all(np.isfinite(data)))
            metadata["all_finite"] = finite

            if not finite:
                metadata["error"] = "Image contains NaN or infinite values."
                return metadata

            finite_vals = data[np.isfinite(data)]
            if finite_vals.size > 0:
                metadata["min_val"] = float(np.min(finite_vals))
                metadata["max_val"] = float(np.max(finite_vals))
                metadata["nonzero_voxels"] = int(np.count_nonzero(finite_vals))

            if is_mask:
                unique_vals = [float(v) for v in np.unique(data)]
                metadata["unique_labels"] = unique_vals
                is_discrete = all(float(v).is_integer() for v in unique_vals)
                metadata["is_discrete"] = is_discrete
                if not is_discrete:
                    metadata["error"] = f"Mask contains non-integer values: {unique_vals[:5]}"

    except Exception as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"

    return metadata


def validate_subject_inputs(
    subject_id: str,
    anat_dir: Path | str,
    require_roi: bool = True,
    expected_lesion_label: int = 1,
    read_voxel_data: bool = False,
) -> dict[str, Any]:
    """
    Validate T1, FLAIR, and ROI for a single subject.

    Returns full subject inventory record with eligibility and reasons.
    """
    anat_dir = Path(anat_dir)
    discovered = discover_subject_files(anat_dir)

    record: dict[str, Any] = {
        "subject_id": subject_id,
        "anat_dir": str(anat_dir.resolve()) if anat_dir.exists() else str(anat_dir),
        "status": "ELIGIBLE",
        "exclusion_reasons": [],
        "warnings": [],
        "t1_path": None,
        "flair_path": None,
        "roi_path": None,
        "t1_meta": None,
        "flair_meta": None,
        "roi_meta": None,
        "roi_on_flair_grid": None,
        "corner_displacement_flair_roi_mm": None,
    }

    # 1. Check FLAIR candidates
    flair_cands = discovered["flair"]
    if len(flair_cands) == 0:
        record["status"] = "EXCLUDED"
        record["exclusion_reasons"].append("Missing required FLAIR image")
    elif len(flair_cands) > 1:
        record["status"] = "EXCLUDED"
        record["exclusion_reasons"].append(
            f"Ambiguous FLAIR scans ({len(flair_cands)} found): {[p.name for p in flair_cands]}"
        )
    else:
        record["flair_path"] = str(flair_cands[0].resolve())
        flair_meta = inspect_nifti_metadata(flair_cands[0], is_mask=False, read_voxel_data=read_voxel_data)
        record["flair_meta"] = flair_meta
        if flair_meta.get("error"):
            record["status"] = "EXCLUDED"
            record["exclusion_reasons"].append(f"FLAIR invalid: {flair_meta['error']}")

    # 2. Check T1 candidates
    t1_cands = discovered["t1"]
    if len(t1_cands) == 0:
        record["status"] = "EXCLUDED"
        record["exclusion_reasons"].append("Missing required T1-weighted image")
    elif len(t1_cands) > 1:
        record["status"] = "EXCLUDED"
        record["exclusion_reasons"].append(
            f"Ambiguous T1 scans ({len(t1_cands)} found): {[p.name for p in t1_cands]}"
        )
    else:
        record["t1_path"] = str(t1_cands[0].resolve())
        t1_meta = inspect_nifti_metadata(t1_cands[0], is_mask=False, read_voxel_data=read_voxel_data)
        record["t1_meta"] = t1_meta
        if t1_meta.get("error"):
            record["status"] = "EXCLUDED"
            record["exclusion_reasons"].append(f"T1 invalid: {t1_meta['error']}")

    # 3. Check ROI candidates
    roi_cands = discovered["roi"]
    if len(roi_cands) == 0:
        if require_roi:
            record["status"] = "EXCLUDED"
            record["exclusion_reasons"].append("Missing required lesion ROI mask (e.g. Healthy Control)")
    elif len(roi_cands) > 1:
        record["status"] = "EXCLUDED"
        record["exclusion_reasons"].append(
            f"Ambiguous ROI masks ({len(roi_cands)} found): {[p.name for p in roi_cands]}"
        )
    else:
        record["roi_path"] = str(roi_cands[0].resolve())
        roi_meta = inspect_nifti_metadata(roi_cands[0], is_mask=True, read_voxel_data=True)
        record["roi_meta"] = roi_meta
        if roi_meta.get("error"):
            record["status"] = "EXCLUDED"
            record["exclusion_reasons"].append(f"ROI invalid: {roi_meta['error']}")
        else:
            labels = roi_meta.get("unique_labels") or []
            if float(expected_lesion_label) not in labels:
                record["warnings"].append(
                    f"Expected lesion label {expected_lesion_label} not found in unique labels: {labels}"
                )
            if roi_meta.get("nonzero_voxels", 0) == 0:
                record["warnings"].append("ROI contains zero non-zero voxels (empty mask)")

    # 4. Compare ROI to FLAIR grid if both exist and are readable
    if (
        record["flair_path"]
        and record["roi_path"]
        and record["flair_meta"]
        and record["flair_meta"].get("readable")
        and record["roi_meta"]
        and record["roi_meta"].get("readable")
    ):
        flair_img = nib.load(record["flair_path"])
        roi_img = nib.load(record["roi_path"])
        try:
            disp = maximum_corner_displacement_mm(flair_img, roi_img)
            record["corner_displacement_flair_roi_mm"] = float(disp)
            on_grid = same_grid(flair_img, roi_img, tolerance_mm=0.01)
            record["roi_on_flair_grid"] = bool(on_grid)
            if not on_grid:
                record["warnings"].append(
                    f"ROI corner displacement from FLAIR is {disp:.4f} mm (> 0.01 mm tolerance)"
                )
        except Exception as exc:
            record["roi_on_flair_grid"] = False
            record["warnings"].append(f"Grid comparison failed: {exc}")

    return record


def inventory_cohort(
    data_root: Path | str,
    participants_tsv: Path | str | None = None,
    require_roi: bool = True,
    expected_lesion_label: int = 1,
    read_voxel_data: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Inventory all subjects in data_root and cross-reference with participants_tsv if provided.

    Returns:
        tuple of (subject_records, summary_dict)
    """
    data_root = Path(data_root).resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root directory does not exist: {data_root}")

    # Discover subject folders (sub-*)
    subject_dirs = sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("sub-"))

    # Load participants.tsv if available
    participants_info: dict[str, dict[str, str]] = {}
    if participants_tsv is not None:
        pts_path = Path(participants_tsv)
        if pts_path.is_file():
            import csv
            with pts_path.open("r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    # Clean keys of any potential BOM or whitespace
                    cleaned_row = {k.strip().lstrip("\ufeff"): v for k, v in row.items() if k}
                    pid = cleaned_row.get("participant_id")
                    if pid:
                        participants_info[pid.strip()] = cleaned_row

    records: list[dict[str, Any]] = []

    for s_dir in subject_dirs:
        subject_id = s_dir.name
        anat_dir = s_dir / "anat"
        if not anat_dir.is_dir():
            anat_dir = s_dir

        record = validate_subject_inputs(
            subject_id=subject_id,
            anat_dir=anat_dir,
            require_roi=require_roi,
            expected_lesion_label=expected_lesion_label,
            read_voxel_data=read_voxel_data,
        )

        if subject_id in participants_info:
            p_data = participants_info[subject_id]
            record["group"] = p_data.get("group", "unknown")
            record["split"] = p_data.get("split", "unknown")
            record["sex"] = p_data.get("sex", "")
            record["age_scan"] = p_data.get("age_scan", "")
        else:
            record["group"] = "unknown"
            record["split"] = "unknown"

        records.append(record)

    total = len(records)
    eligible = sum(r["status"] == "ELIGIBLE" for r in records)
    excluded = total - eligible

    summary = {
        "data_root": str(data_root),
        "total_subjects": total,
        "eligible_subjects": eligible,
        "excluded_subjects": excluded,
        "with_t1": sum(bool(r["t1_path"]) for r in records),
        "with_flair": sum(bool(r["flair_path"]) for r in records),
        "with_roi": sum(bool(r["roi_path"]) for r in records),
        "roi_on_flair_grid_count": sum(bool(r.get("roi_on_flair_grid")) for r in records),
    }

    return records, summary
