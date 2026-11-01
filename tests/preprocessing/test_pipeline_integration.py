"""Integration tests for the complete end-to-end preprocessing pipeline."""

import csv
import json
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from fcd_segmentation.preprocessing.geometry import same_grid
from fcd_segmentation.preprocessing.pipeline import run_pipeline


def _create_mock_subject(anat_dir: Path, subject_id: str, with_roi: bool = True):
    anat_dir.mkdir(parents=True, exist_ok=True)
    shape = (32, 32, 32)
    z, y, x = np.ogrid[:32, :32, :32]
    dist = np.sqrt((x - 16)**2 + (y - 16)**2 + (z - 16)**2)
    flair_data = np.zeros(shape, dtype=np.float32)
    flair_data[dist <= 12] = 100.0 - dist[dist <= 12] * 4.0
    t1_data = np.zeros(shape, dtype=np.float32)
    t1_data[dist <= 12] = 120.0 - dist[dist <= 12] * 5.0
    affine = np.diag([1.0, 1.0, 1.0, 1.0])

    flair_p = anat_dir / f"{subject_id}_FLAIR.nii.gz"
    t1_p = anat_dir / f"{subject_id}_T1w.nii.gz"
    nib.save(nib.Nifti1Image(flair_data, affine), str(flair_p))
    nib.save(nib.Nifti1Image(t1_data, affine), str(t1_p))

    roi_p = None
    if with_roi:
        roi_data = np.zeros(shape, dtype=np.uint8)
        roi_data[14:18, 14:18, 14:18] = 1
        roi_p = anat_dir / f"{subject_id}_FLAIR_roi.nii.gz"
        nib.save(nib.Nifti1Image(roi_data, affine), str(roi_p))

    return flair_p, t1_p, roi_p


@pytest.fixture
def synthetic_cohort(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # sub-001 (FCD with ROI)
    _create_mock_subject(raw_dir / "sub-001" / "anat", "sub-001", with_roi=True)
    # sub-002 (HC without ROI)
    _create_mock_subject(raw_dir / "sub-002" / "anat", "sub-002", with_roi=False)

    pts_path = raw_dir / "participants.tsv"
    with pts_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["participant_id", "group", "split"])
        w.writerow(["sub-001", "fcd", "train"])
        w.writerow(["sub-002", "hc", "train"])

    config = {
        "paths": {
            "raw_data_dir": str(raw_dir),
            "processed_data_dir": str(tmp_path / "processed"),
            "reports_dir": str(tmp_path / "reports"),
            "participants_tsv": str(pts_path),
        },
        "datasets": {
            "flair_id": "Dataset101_FLAIR",
            "t1_id": "Dataset102_T1",
            "fusion_id": "Dataset103_Fusion",
            "file_ending": ".nii.gz",
        },
        "reference_modality": "FLAIR",
        "skull_stripping": {
            "enabled": True,
            "backend": "hd-bet",
            "device": "cpu",
            "tta": False,
        },
        "registration": {
            "enabled": True,
            "backend": "simpleitk",
            "method": "rigid",
            "number_of_iterations": 10,
            "number_of_bins": 10,
            "sampling_percentage": 0.5,
        },
        "normalization": {
            "method": "foreground_zscore",
            "clip_percentiles": [1.0, 99.0],
            "eps": 1e-8,
        },
        "label": {
            "expected_lesion_value": 1,
            "mapping": {"background": 0, "lesion": 1},
        },
        "splits": {
            "policy": "participants_tsv",
            "held_out_target": "imagesTr",
        },
        "execution": {
            "resume": False,
            "overwrite": True,
            "keep_failed_intermediates": False,
            "fail_fast": False,
        },
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    return {
        "tmp_path": tmp_path,
        "raw_dir": raw_dir,
        "processed_dir": tmp_path / "processed",
        "reports_dir": tmp_path / "reports",
        "config_path": cfg_path,
    }


def _mock_run_hd_bet(input_path, output_dir, subject_id, modality="FLAIR", output_image_path=None, output_mask_path=None, **kwargs):
    # Mock HD-BET by copying input image and making a dummy brain mask
    in_img = nib.load(str(input_path))
    data = np.asarray(in_img.dataobj)
    mask = (data > 0).astype(np.uint8)

    out_img_p = Path(output_image_path) if output_image_path else Path(output_dir) / f"{subject_id}_{modality}_skullstripped.nii.gz"
    out_mask_p = Path(output_mask_path) if output_mask_path else Path(output_dir) / f"{subject_id}_{modality}_brain_mask.nii.gz"

    nib.save(nib.Nifti1Image(data * mask, in_img.affine, header=in_img.header), str(out_img_p))
    nib.save(nib.Nifti1Image(mask, in_img.affine), str(out_mask_p))

    return {
        "subject_id": subject_id,
        "input_path": str(input_path),
        "skullstripped_path": str(out_img_p),
        "brain_mask_path": str(out_mask_p),
        "device": "cpu",
        "status": "success",
    }


@patch("fcd_segmentation.preprocessing.pipeline.run_hd_bet", side_effect=_mock_run_hd_bet)
def test_end_to_end_pipeline(mock_hdbet, synthetic_cohort):
    cfg_path = synthetic_cohort["config_path"]
    reports_dir = synthetic_cohort["reports_dir"]
    processed_dir = synthetic_cohort["processed_dir"]
    raw_dir = synthetic_cohort["raw_dir"]

    # Record mtime of raw files to verify they are never touched
    raw_files = list(raw_dir.rglob("*.nii.gz"))
    raw_mtimes = {str(p): p.stat().st_mtime_ns for p in raw_files}

    # Run pipeline
    results = run_pipeline(config_path=cfg_path)
    summary = results["summary"]
    assert summary["attempted"] == 1  # sub-001 (sub-002 excluded due to missing ROI)
    assert summary["passed"] == 1
    assert summary["failed"] == 0

    # 1. Verify 3 nnU-Net datasets
    for ds_id in ("Dataset101_FLAIR", "Dataset102_T1", "Dataset103_Fusion"):
        ds_dir = processed_dir / ds_id
        assert ds_dir.is_dir()
        assert (ds_dir / "dataset.json").is_file()
        assert (ds_dir / "imagesTr" / "sub-001_0000.nii.gz").is_file()
        assert (ds_dir / "labelsTr" / "sub-001.nii.gz").is_file()

    # 2. Verify Fusion has channel 1
    fusion_dir = processed_dir / "Dataset103_Fusion"
    assert (fusion_dir / "imagesTr" / "sub-001_0001.nii.gz").is_file()

    # 3. Verify fusion geometry matches exactly
    f0 = nib.load(str(fusion_dir / "imagesTr" / "sub-001_0000.nii.gz"))
    f1 = nib.load(str(fusion_dir / "imagesTr" / "sub-001_0001.nii.gz"))
    lbl = nib.load(str(fusion_dir / "labelsTr" / "sub-001.nii.gz"))
    assert same_grid(f0, f1, tolerance_mm=0.01)
    assert same_grid(f0, lbl, tolerance_mm=0.01)

    # 4. Verify discrete labels
    lbl_data = np.asarray(lbl.dataobj)
    assert set(np.unique(lbl_data)).issubset({0, 1})

    # 5. Verify HTML reports exist
    qc_dir = reports_dir / "qc"
    assert (qc_dir / "input_inventory.html").is_file()
    assert (qc_dir / "skull_stripping_FLAIR.html").is_file()
    assert (qc_dir / "skull_stripping_T1.html").is_file()
    assert (qc_dir / "registration_T1_to_FLAIR.html").is_file()
    assert (qc_dir / "final_dataset_qc.html").is_file()

    # Verify base64 images embedded in HTML
    reg_html = (qc_dir / "registration_T1_to_FLAIR.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in reg_html

    # 6. Verify manifests exist
    manifest_dir = reports_dir / "manifests"
    assert (manifest_dir / "preprocessing_manifest.json").is_file()
    assert (manifest_dir / "subject_splits.json").is_file()
    assert (manifest_dir / "failures.csv").is_file()

    # 7. Verify raw files were untouched
    for p_str, original_mtime in raw_mtimes.items():
        assert Path(p_str).stat().st_mtime_ns == original_mtime

    # 8. Test --resume skips completed subject
    resume_res = run_pipeline(config_path=cfg_path, resume=True, overwrite=False)
    assert resume_res["summary"]["skipped"] == 1
