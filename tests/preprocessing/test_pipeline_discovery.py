"""Unit tests for discovery and input validation module."""

from pathlib import Path

import nibabel as nib
import numpy as np

from fcd_segmentation.preprocessing.discovery import (
    discover_subject_files,
    inspect_nifti_metadata,
    is_roi_file,
    validate_subject_inputs,
)


def _make_dummy_nifti(path: Path, shape=(10, 10, 10), affine=None, data=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.eye(4)
    if data is None:
        data = np.ones(shape, dtype=np.float32)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))
    return path


def test_is_roi_file():
    assert is_roi_file("sub-001_acq-FLAIR_roi.nii.gz")
    assert is_roi_file("sub-001_roi.nii")
    assert is_roi_file("sub-001_mask.nii.gz")
    assert not is_roi_file("sub-001_acq-FLAIR.nii.gz")
    assert not is_roi_file("sub-001_T1w.nii")


def test_discover_subject_files_excludes_roi_from_flair(tmp_path):
    anat_dir = tmp_path / "anat"
    anat_dir.mkdir()

    _make_dummy_nifti(anat_dir / "sub-001_T1w.nii.gz")
    _make_dummy_nifti(anat_dir / "sub-001_FLAIR.nii.gz")
    _make_dummy_nifti(anat_dir / "sub-001_FLAIR_roi.nii.gz")

    discovered = discover_subject_files(anat_dir)
    assert len(discovered["t1"]) == 1
    assert len(discovered["flair"]) == 1
    assert len(discovered["roi"]) == 1

    assert discovered["flair"][0].name == "sub-001_FLAIR.nii.gz"
    assert discovered["roi"][0].name == "sub-001_FLAIR_roi.nii.gz"
    assert discovered["t1"][0].name == "sub-001_T1w.nii.gz"


def test_inspect_nifti_metadata_detects_geometry_and_labels(tmp_path):
    roi_data = np.zeros((10, 10, 10), dtype=np.uint8)
    roi_data[2:5, 2:5, 2:5] = 1
    p = _make_dummy_nifti(tmp_path / "roi.nii.gz", data=roi_data)

    meta = inspect_nifti_metadata(p, is_mask=True, read_voxel_data=True)
    assert meta["readable"] is True
    assert meta["shape"] == [10, 10, 10]
    assert meta["unique_labels"] == [0.0, 1.0]
    assert meta["is_discrete"] is True
    assert meta["nonzero_voxels"] == 27
    assert meta["error"] is None


def test_validate_subject_inputs_detects_missing_and_duplicates(tmp_path):
    anat_dir = tmp_path / "sub-001" / "anat"
    anat_dir.mkdir(parents=True)

    # Missing FLAIR
    _make_dummy_nifti(anat_dir / "sub-001_T1w.nii.gz")
    rec = validate_subject_inputs("sub-001", anat_dir, require_roi=True)
    assert rec["status"] == "EXCLUDED"
    assert any("Missing required FLAIR" in r for r in rec["exclusion_reasons"])

    # Duplicate FLAIR
    _make_dummy_nifti(anat_dir / "sub-001_acq1_FLAIR.nii.gz")
    _make_dummy_nifti(anat_dir / "sub-001_acq2_FLAIR.nii.gz")
    rec2 = validate_subject_inputs("sub-001", anat_dir, require_roi=True)
    assert rec2["status"] == "EXCLUDED"
    assert any("Ambiguous FLAIR" in r for r in rec2["exclusion_reasons"])
